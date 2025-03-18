import torch
import math

def sinusoidal_positional_encoding(length, dim):
    '''
    Returns (length, dim)
    '''
    pe = torch.zeros(length, dim)
    n_effective_dim = dim - dim % 2
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, n_effective_dim, 2).float() * (-math.log(10000.0) / n_effective_dim))
    pe[:, 0:n_effective_dim:2] = torch.sin(position * div_term)
    pe[:, 1:n_effective_dim:2] = torch.cos(position * div_term)
    return pe

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

def one_hot_positional_encoding(length: int, dim: int):
    '''
    Returns (length, dim)
    '''
    return torch.eye(dim, dim).repeat(max((length)//dim+1,1),1)[:length]