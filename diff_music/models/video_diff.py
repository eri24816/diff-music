import einops
import torch.nn as nn
import torch
from dataclasses import dataclass

from .transformer import binary_positional_encoding
from .attention import create_attention_blocks, create_local_attention_blocks

class VideoDiffusion(nn.Module):
    @dataclass
    class Params:
        dim: int = 256
        max_length: int = 256
        depth: int = 1
        heads: int = 8
        attn_type: str = 'vanilla'
        rotary_pos_emb: bool = True
        frames_per_bar: int = 32
        local_attn_window_size: int = 96

    def __init__(self, params: Params, in_channels: int):
        
            super().__init__()
            self.in_channels = in_channels
            self.depth = params.depth
            self.frames_per_bar = params.frames_per_bar
            self.num_features = params.dim
            self.register_buffer('pe', binary_positional_encoding(length=params.max_length, dim=params.dim).unsqueeze(0)) # (b=1, t, f)

            self.in_layer = nn.Linear(in_channels*88 + 1, params.dim)

            self.frame_attn_blocks = nn.ModuleList()
            for _ in range(params.depth):
                self.frame_attn_blocks.append(
                    create_local_attention_blocks(
                        dim = params.dim,
                        heads = params.heads,
                        window_size = params.local_attn_window_size,
                        depth = 1,
                        residual = True
                    )
                )

            self.bar_attn_blocks = nn.ModuleList()
            for _ in range(params.depth):
                self.bar_attn_blocks.append(
                    create_attention_blocks(
                        dim = params.dim,
                        heads = params.heads,
                        depth = 1,
                        attn_type = params.attn_type,
                        max_length = params.max_length,
                        rotary_pos_emb = params.rotary_pos_emb,
                        residual = True
                    )
                )

            self.out_layer = nn.Linear(params.dim, in_channels*88)
            
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        '''
        x: (batch_size, in_channels, time, pitch)
        time_steps: (batch_size)
        '''

        batch_size = x.shape[0]
        frame_dim_size = self.frames_per_bar
        bar_dim_size = x.shape[2] // frame_dim_size
        
        x = einops.rearrange(x, 'b c t p -> b t (p c)') # (b, t, f)
        x = torch.cat([x, time_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, x.shape[1], -1)], dim=-1)
        x = self.in_layer(x)

        x = x + self.pe[:, :x.shape[1]]

        x = einops.rearrange(x, 'batch (bar frame) features -> (batch bar) frame features', frame=frame_dim_size, batch=batch_size, features=self.num_features)

        for i in range(self.depth):
            x = self.frame_attn_blocks[i](x) # (batch, bar) frame features
            x = einops.rearrange(x, '(batch bar) frame features -> (batch frame) bar features', frame=frame_dim_size, batch=batch_size, features=self.num_features)
            
            x = self.bar_attn_blocks[i](x) # (batch, frame) bar features
            x = einops.rearrange(x, '(batch frame) bar features -> (batch bar) frame features', frame=frame_dim_size, batch=batch_size, features=self.num_features)

        x = einops.rearrange(x, '(batch bar) frame features -> batch (bar frame) features', frame=frame_dim_size, batch=batch_size, features=self.num_features)

        x = self.out_layer(x)

        x = einops.rearrange(x, 'b t (p c) -> b c t p', c=self.in_channels)

        return x
