# thanks to lucidrains!

import einops
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from dataclasses import field

from diff_music.models.attention import create_attention_blocks

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x
    
def binary_positional_encoding(length: int, dim: int):
    '''
    Returns (length, dim)
    '''
    res = []
    for i in range(length):
        res.append([int(x) for x in f"{i:0{dim}b}"][-dim:])
        # pad
        res[-1] += [0] * (dim - len(res[-1]))

    return torch.tensor(res, dtype=torch.float32).flip(dims=[1])

def binary_and_one_hot_encoding(length: int, dim: int):
    '''
    Returns (length, dim)
    '''
    min_bin_dim = int(np.ceil(np.log2(length)))
    one_hot_dim = length
    bin_dim = dim - one_hot_dim
    assert bin_dim > min_bin_dim, f"dim must be greater than ceil(log2(length))+length = {min_bin_dim + length}"
    return torch.cat([torch.eye(length, one_hot_dim), binary_positional_encoding(length, bin_dim)], dim=1)


class PianorollDenoiser(nn.Module):
    @dataclass
    class Params:
        dim: int = 256
        max_length: int = 16
        depth: int = 3
        heads: int = 8
        rotary_pos_emb: bool = True
        attn_type: str = 'vanilla'
        time_conv_dilations: list[int] = field(default_factory=lambda: [
            1, # one frame
            2, # eighth note
            4, # one beat
            8, # two beats
            16, # four beats
        ])
        pitch_conv_dilations: list[int] = field(default_factory=lambda: [
            1, # one semitone
            2, # major second
            7, # perfect fifth
            12, # octave
        ])

    def __init__(self, params: Params, in_channels: int):
        super().__init__()

        self.register_buffer('pe', binary_positional_encoding(length=params.max_length, dim=params.dim).unsqueeze(0)) # (b=1, t, f)

        self.in_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + 1,
                out_channels=params.dim,
                kernel_size=1,
                padding=0,
            ),
            nn.LeakyReLU(0.01),
            nn.GroupNorm(num_groups=32, num_channels=params.dim),
        )

        time_conv_layers = []
        for dilation in params.time_conv_dilations:
            time_conv_layers.append(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=params.dim,
                            out_channels=params.dim,
                            kernel_size=(1, 3),
                            padding=(0, dilation),
                            dilation=(1, dilation)
                        ),
                        nn.LeakyReLU(0.01),
                        nn.GroupNorm(num_groups=32, num_channels=params.dim),
                    )
                )
            )
        self.time_conv = nn.Sequential(*time_conv_layers)

        pitch_conv_layers = []
        for dilation in params.pitch_conv_dilations:
            pitch_conv_layers.append(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=params.dim,
                            out_channels=params.dim,
                            kernel_size=(3, 1),
                            padding=(dilation, 0),
                            dilation=(dilation, 1)
                        ),
                        nn.LeakyReLU(0.01),
                        nn.GroupNorm(num_groups=32, num_channels=params.dim),
                    )
                )
            )
        self.pitch_conv = nn.Sequential(*pitch_conv_layers)

        self.before_attn = nn.Sequential(
            nn.Linear(88, params.dim),
            nn.LeakyReLU(0.01),
        )

        self.attn_layers = create_attention_blocks(
            dim = params.dim,
            depth = params.depth,
            heads = params.heads,
            attn_type = params.attn_type,
            max_length = params.max_length,
            rotary_pos_emb = params.rotary_pos_emb,
        )
        
        self.after_attn1 = nn.Sequential(
            nn.Linear(params.dim, 4*88),
        )

        self.after_attn2 = nn.Sequential(
            nn.Linear(4, params.dim),
            nn.LeakyReLU(0.01),
        )

        self.after_attn_norm = nn.GroupNorm(num_groups=32, num_channels=params.dim)

        self.out_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=params.dim*2,
                out_channels=params.dim,
                kernel_size=1,
                padding=0,
            ),
            nn.LeakyReLU(0.01),
            nn.GroupNorm(num_groups=32, num_channels=params.dim),
            nn.Conv2d(
                in_channels=params.dim,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        '''
        x: (batch_size, in_channels, time, pitch)
        time_steps: (batch_size, time)
        '''
        time_steps = time_steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, time_steps], dim=1)
        x = self.in_layer(x)
        x = self.time_conv(x)
        skip = x = self.pitch_conv(x) # (batch_size, feature, time, pitch)

        x = x[:,0] # (batch_size, time, pitch), select the first feature of each time step as the input to the transformer
        x = self.before_attn(x) # (batch_size, time, feature)
        x = x + self.pe[:, :x.shape[1]]
        x = self.attn_layers(x)
        x = self.after_attn1(x) # (batch_size, time, feature)
        x = einops.rearrange(x, 'b t (f p) -> b t p f', p=88)
        x = self.after_attn2(x)
        x = einops.rearrange(x, 'b t p f -> b f t p')
        x = self.after_attn_norm(x)

        x = torch.cat([skip, x], dim=1)
        x = self.out_layer(x)
        
        return x

