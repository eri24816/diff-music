from x_transformers.x_transformers import AttentionLayers
from linear_attention_transformer import LinearAttentionTransformer
from local_attention import LocalMHA
import torch
import torch.nn as nn

def FeedForward(dim, mult = 4, dropout = 0.):
    # copied from local_attention
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.LeakyReLU(0.01),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim)
    )

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x

def create_local_attention_blocks(dim: int, heads: int, window_size: int, residual: bool = False, depth: int=1, ff_mult: int=4, ff_dropout: float=0.0):
    layers = []
    for _ in range(depth):
        layer = nn.Sequential(
            LocalMHA(dim = dim, heads = heads, window_size = window_size),
            FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
        )
        if residual:
            layer = Residual(layer)

        layers.append(layer)
    return nn.Sequential(*layers)

def create_attention_blocks(dim: int, depth: int, heads: int, attn_type: str, max_length: int|None = None, rotary_pos_emb: bool = False, residual: bool = False):
    if attn_type == 'vanilla':
        attn_layers = AttentionLayers(
            dim = dim,
            depth = depth,
            heads = heads,
            rotary_pos_emb = rotary_pos_emb,
            causal = False,
            attn_flash = True,
        )
    elif attn_type == 'linear':
        assert rotary_pos_emb is False, "Rotary positional encoding is not supported for linear attention"
        assert max_length is not None, "max_length must be provided for linear attention"
        attn_layers = LinearAttentionTransformer(
            dim = dim,
            heads = heads,
            depth = depth,
            max_seq_len = max_length,
            n_local_attn_heads = 4,
            causal = False,
        )
    else:
        raise ValueError(f"Invalid attention type: {attn_type}")
    if residual:
        attn_layers = Residual(attn_layers)
    return attn_layers