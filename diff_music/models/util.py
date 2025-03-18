import torch
from torch import Tensor

def pad_to_length(x: Tensor, target_length: int, dim: int, pad_value: float):
    padding_shape = list(x.shape)
    padding_shape[dim] = target_length - x.shape[dim]
    return torch.cat([x, torch.full(padding_shape, pad_value)])

def pad_and_stack(batch: list[Tensor], pad_dim: int, pad_value: float=0, stack_dim: int=0, target_length: int|None=None) -> Tensor:
    if target_length is None:
        target_length = max(x.shape[pad_dim] for x in batch)
    return torch.stack([pad_to_length(x, target_length, pad_dim, pad_value) for x in batch], stack_dim)

def cat_to_right(x: torch.Tensor, value: torch.Tensor|int|float|list[int|float|Tensor]|torch.nn.Parameter, dim: int = -1):
    '''
    x: (1, ..., 1, length, d1, d2,...)
    value: (d1, d2,...)
    dim: int
    '''
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)

    if dim < 0:
        dim = x.ndim + dim

    assert x.shape[dim+1:] == value.shape, f"expected value.shape == {x.shape[dim+1:]} but got {value.shape}"
    n_unsqueeze = x.ndim - value.ndim
    for _ in range(n_unsqueeze):
        value = value.unsqueeze(0)

    return torch.cat([x, value], dim=dim)
