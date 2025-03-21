from dataclasses import dataclass
import torch
import torch.nn as nn


from .representation import SymbolicRepresentation, cat_to_right
from .pe import sinusoidal_positional_encoding, binary_positional_encoding, one_hot_positional_encoding

# def promptable_feature_extractor_wrapper(func: Callable[[SymbolicRepresentation, torch.Tensor|None], torch.Tensor], x: SymbolicRepresentation, prompt: SymbolicRepresentation|None=None, condition: torch.Tensor|None=None):
#     '''
#     Wrapper for feature extractors to make them promptable.
#     '''
#     if prompt is not None:
#         # move pad tokens in prompt to the end of x so it doesn't affect the model's output
#         # i am too dumb to implement this in batch
#         prompt_and_x = []
#         prompt_lengths = (~prompt.is_pad).sum(dim=1)
#         for i in range(prompt.batch_size):
#             prompt_and_x.append(prompt[i:i+1, :prompt_lengths[i]] + x[i:i+1, :prompt_lengths[i]] + prompt[i:i+1, prompt_lengths[i]:])
    
#         # the new x is the concatenation of prompt and x, with no padding in between
#         x = SymbolicRepresentation.cat(prompt_and_x, dim=0)

#     result = func(x, condition)

#     if prompt is not None:
#         # only return the result of positions of x
#         actual_result = []
#         for i in range(x.batch_size):
#             actual_result.append(result[i, :prompt_lengths[i]])
#         actual_result = pad_and_stack(actual_result, pad_dim=1, stack_dim=0)

#     return actual_result

class FeatureExtractor(nn.Module):
    '''
    - dim (int):
        - The dimension of the network and the output feature
    - num_layers (int):
        - The number of transformer layers
    - pitch_range (list[int]):
        - The range of the pitch
    - max_len (int):
        - The maximum length of the input. Positional encodings are initialized with this length.
    - reduce (bool):
        - If True, the output is reduced to a single vector (batch_size, dim).
        - If False, the output is a sequence of vectors (batch_size, num_tokens, dim).
    '''
    
    @dataclass
    class Params:
        '''
        - dim (int):
            - The dimension of the network and the output feature
        - num_layers (int):
            - The number of transformer layers
        - pitch_range (list[int]):
            - The range of the pitch
        - max_len (int):
            - The maximum length of the input. Positional encodings are initialized with this length.
        - reduce (bool):
            - If True, the output is reduced to a single vector (batch_size, dim).
            - If False, the output is a sequence of vectors (batch_size, num_tokens, dim).
        '''
        dim: int
        num_layers: int
        pitch_range: list[int]
        num_pos: int
        reduce: bool
        condition_dim: int = 0
    
    def __init__(self, params: Params, is_causal: bool=True):
        super().__init__()
        self.is_causal = is_causal
        self.pitch_range = params.pitch_range
        self.reduce = params.reduce
        self.num_pitch = params.pitch_range[1] - params.pitch_range[0]
        self.dim = params.dim

        self.frame_emb = nn.Embedding(1, params.dim)
        self.pitch_emb = nn.Embedding(self.num_pitch, params.dim)
        self.velocity_emb = nn.Embedding(128, params.dim)

        if params.condition_dim > 0:
            self.condition_emb = nn.Linear(params.condition_dim, params.dim)
        else:
            self.condition_emb = None

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(params.dim, nhead=8, batch_first=True),
            num_layers=params.num_layers,
        )

        sinusoidal_pe = sinusoidal_positional_encoding(
            params.num_pos, params.dim - 5 - 32
        )
        binary_pe = binary_positional_encoding(params.num_pos, 5)
        one_hot_pe = one_hot_positional_encoding(params.num_pos, 32)

        pe = torch.cat([binary_pe, one_hot_pe, sinusoidal_pe], dim=1)  # (max_len, dim)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

        if params.reduce:
            self.out_token_emb = torch.nn.Parameter(torch.randn(params.dim))
        else:
            self.out_token_emb = None

    def to(self, device: torch.device):
        super().to(device)
        self.device = device

    def forward(
        self, input: SymbolicRepresentation, condition: torch.Tensor | None = None
    ):
        """
        x: SymbolicRepresentation (batch_size, num_tokens, ...)
        condition: (batch_size, length, dim)
        returns features extracted from the input. Used for downstream classification.
        return shape: (batch_size, num_tokens, dim)
        """

        pe = self.pe[input.pos]  # (batch_size, num_tokens, dim)

        x = input.is_note.unsqueeze(-1) * (
            self.pitch_emb(input.pitch) + self.velocity_emb(input.velocity)
        )  # (batch_size, num_tokens, dim)
        x = x + input.is_frame.unsqueeze(-1) * self.frame_emb(
            torch.zeros_like(input.pitch)
        )  # (batch_size, num_tokens, dim)
        x = x + pe  # (batch_size, num_tokens, dim)

        if condition is not None:
            condition = self.condition_emb(condition)
            x += condition

        if self.reduce:
            # add out token to last position
            assert self.out_token_emb is not None
            x = cat_to_right(x, self.out_token_emb, dim=1)

        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device) if self.is_causal else None
        src_key_padding_mask = (~input.is_pad).to(x.device)
        if self.reduce:
            # * IMPORTANT: the key mask of the out token must be true, otherwise the output will be zero when torch.no_grad(). I believe this is a bug in pytorch.
            src_key_padding_mask = cat_to_right(src_key_padding_mask, True, dim=1)

        # convert False and True to 0 and -inf.
        src_key_padding_mask = src_key_padding_mask.log()

        x = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=self.is_causal,
        )  # (batch_size, num_tokens, dim)

        if self.reduce:
            # get the last token
            x = x[:, -1, :] # (batch_size, dim)

        if self.reduce:
            assert x.shape == (input.batch_size, self.dim)
        else:
            assert x.shape == (input.batch_size, input.length, self.dim)

        return x
    
    
