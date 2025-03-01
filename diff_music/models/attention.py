from x_transformers.x_transformers import AttentionLayers
from linear_attention_transformer import LinearAttentionTransformer

def create_attention_blocks(dim: int, depth: int, heads: int, attn_type: str, max_length: int|None = None, rotary_pos_emb: bool = False):
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
    return attn_layers