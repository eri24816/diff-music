from dataclasses import dataclass
from music_data_analysis import Note, Pianoroll
import torch
import torch.nn as nn
import math
from torch import Tensor
from tqdm import tqdm

class Tokenizer:
    def __init__(self,n_pitch: int):
        '''
        Tokenize a patch into a sequence of tokens

        Example:
          Pitch(60), Velocity(100)
          Frame
          Pitch(64), Velocity(100)
          Pitch(67), Velocity(100)
          Frame
        '''
    def tokenize(self, pr: Pianoroll):
        '''
        pr: Pianoroll
        '''
        tokens: list[dict|str] = []
        current_frame = -1
        for note in pr.notes:
            if note.onset > current_frame:
                tokens.append("frame")
                current_frame = note.onset
            tokens.append({"type": "pitch", "value": note.pitch})
        return tokens

    # def tokenize(self, pr: torch.Tensor):
    #     '''
    #     pr: (patch_height(pitch), patch_width(time))
    #     '''
    #     tokens: list[dict|str] = []
    #     onsets = pr.nonzero(as_tuple=False)
    #     onsets = onsets[onsets[:, 1].argsort()] # sort by time
    #     for i in range(onsets.shape[0]):
    #         tokens.append({"type": "time", "value": onsets[i, 1].item()})
    #         tokens.append({"type": "pitch", "value": onsets[i, 0].item()})

    #     return tokens

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
    return torch.eye(dim, dim).repeat(max((length+1)//dim,1),1)[:length]

def append_to_right(x: torch.Tensor, value: torch.Tensor|int|float|list[int|float], dim: int = -1):
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

def nucleus_sample(logits: torch.Tensor, p: float):
    """
    logits: (classes)
    p: float close to 1
    return: scalar
    """
    probs = torch.softmax(logits, dim=0)
    sorted_probs, sorted_indices = torch.sort(probs, dim=0, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    selected_indices = []
    selected_probs = []
    for i in range(len(sorted_probs)):
        selected_indices.append(sorted_indices[i])
        selected_probs.append(sorted_probs[i])
        if cumulative_probs[i] > p:
            break
    # sample from selected_indices
    # normalize selected_probs
    selected_probs = torch.tensor(selected_probs)
    selected_probs = selected_probs / torch.sum(selected_probs)
    selected = torch.multinomial(selected_probs, 1)
    return selected_indices[selected]


class MidiLikeTransformer(nn.Module):
    '''
    Autoregressive decoder
    
    '''

    PAD = -1
    FRAME = 0
    NOTE = 1

    @dataclass
    class Params:
        hidden_dim: int
        num_layers: int
        pitch_range: list[int]
        max_len: int

    @dataclass
    class Loss:
        token_type_loss: Tensor
        pitch_loss: Tensor
        total_loss: Tensor
        pitch_acc: float
        token_type_acc: float

    def __init__(self, params: Params):
        super().__init__()
        self.hidden_dim = params.hidden_dim
        self.num_layers = params.num_layers
        self.pitch_range = params.pitch_range
        self.num_pitch = params.pitch_range[1] - params.pitch_range[0]
        self.frame_emb = nn.Embedding(1, params.hidden_dim)
        self.pitch_emb = nn.Embedding(self.num_pitch, params.hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(params.hidden_dim, nhead=8, batch_first=True),
            num_layers=params.num_layers,
        )
        self.token_type_classifier = nn.Linear(params.hidden_dim, 2)
        self.pitch_classifier = nn.Linear(params.hidden_dim, self.num_pitch)

        sinusoidal_pe = sinusoidal_positional_encoding(params.max_len, params.hidden_dim-5-32)
        binary_pe = binary_positional_encoding(params.max_len, 5)
        one_hot_pe = one_hot_positional_encoding(params.max_len, 32)

        pe = torch.cat([binary_pe, one_hot_pe, sinusoidal_pe], dim=1) # (max_len, hidden_dim)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def to(self, device: torch.device):
        super().to(device)
        self.device = device

    def forward(self, token: torch.Tensor, token_type: torch.Tensor, pos: torch.Tensor):
        '''
        token: (batch_size, num_tokens, data_dim)
        token_type: (batch_size, num_tokens), -1: pad, 0: frame, 1: note
        pos: (batch_size, num_tokens)

        returns features extracted from the input. Used for downstream classification.
        return shape: (batch_size, num_tokens, hidden_dim)
        '''

        is_frame = token_type == self.FRAME
        is_note = token_type == self.NOTE

        pe = self.pe[pos] # (batch_size, num_tokens, hidden_dim)

        x = is_note.unsqueeze(-1) * self.pitch_emb(token[:,:,0]) # (batch_size, num_tokens, hidden_dim)
        x = x + is_frame.unsqueeze(-1) * self.frame_emb(torch.zeros_like(token[:,:,0])) # (batch_size, num_tokens, hidden_dim)

        x = x + pe # (batch_size, num_tokens, hidden_dim)
        x = self.transformer.forward(x, nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device), is_causal=True) # (batch_size, num_tokens, hidden_dim)

        return x

    def calculate_loss(self, token: torch.Tensor, token_type: torch.Tensor, pos: torch.Tensor) -> Loss:
        B = token.shape[0]

        is_frame = token_type == self.FRAME
        is_note = token_type == self.NOTE

        feature = self.forward(token, token_type, pos)
        
        token_type_logits = self.token_type_classifier(feature) # (batch_size, num_tokens, 2)
        pitch_logits = self.pitch_classifier(feature) # (batch_size, num_tokens, vocab_size)

        assert token_type_logits.shape == (B, feature.shape[1], 2)
        assert pitch_logits.shape == (B, feature.shape[1], self.num_pitch)


        # shift the logits and tokens by 1
        token_type_logits = token_type_logits[:,:-1]
        pitch_logits = pitch_logits[:,:-1]
        token_type = token_type[:,1:]
        token = token[:,1:]
        is_note = is_note[:,1:]

        token_type_loss = torch.nn.functional.cross_entropy(token_type_logits.transpose(1, 2), token_type, ignore_index=-1, reduction="mean")
        pitch_loss = (torch.nn.functional.cross_entropy(pitch_logits.transpose(1, 2), token[:,:,0], ignore_index=-1, reduction="none") * is_note).mean()
        total_loss = token_type_loss + pitch_loss

        token_type_acc = (token_type_logits.detach().argmax(dim=2) == token_type).float().mean().item()
        pitch_acc = ((pitch_logits.detach().argmax(dim=2) == token[:,:,0]) * is_note).float().mean().item()
        return MidiLikeTransformer.Loss(token_type_loss=token_type_loss, pitch_loss=pitch_loss, total_loss=total_loss, pitch_acc=pitch_acc, token_type_acc=token_type_acc)

    def sample(self, length: int, max_iter: int|None=None):
        '''
        autoregressive sampling
        length: number of frames
        '''

        if max_iter is None:
            max_iter = length * 5

        token_type_seq = torch.zeros(1, 0, dtype=torch.long, device=self.device) # (b=1, length)
        token_seq = torch.zeros(1, 0, 2, dtype=torch.long, device=self.device) # (b=1, length, data_dim)
        pos_seq = torch.zeros(1, 0, dtype=torch.long, device=self.device) # (b=1, length)

        # add the first token, which is a frame
        token_type_seq = append_to_right(token_type_seq, self.FRAME, dim=1)
        token_seq = append_to_right(token_seq, [0,0], dim=1)
        pos_seq = append_to_right(pos_seq, 0, dim=1)

        current_pos = 0

        pbar = tqdm(range(length))
        for i in range(max_iter):
            feature = self.forward(token_seq, token_type_seq, pos_seq) # (b=1, length, hidden_dim)
            token_type_logits = self.token_type_classifier(feature[:, -1, :])[0] # (class)
            token_type_pred = nucleus_sample(token_type_logits, 0.95) # scalar
            
            if token_type_pred == self.FRAME: # frame
                current_pos += 1

                token_type_seq = append_to_right(token_type_seq, self.FRAME, dim=1)
                token_seq = append_to_right(token_seq, [0,0], dim=1)
                pos_seq = append_to_right(pos_seq, current_pos, dim=1)

                pbar.update(1)

                if current_pos >= length:
                    break

            elif token_type_pred == self.NOTE: # note
                # sample pitch
                pitch_logits = self.pitch_classifier(feature[:, -1, :])[0] # (class)
                pitch_pred = nucleus_sample(pitch_logits, 0.95) # scalar

                token_type_seq = append_to_right(token_type_seq, self.NOTE, dim=1)
                token_seq = append_to_right(token_seq, [pitch_pred, 100], dim=1)
                pos_seq = append_to_right(pos_seq, current_pos, dim=1)
            else:
                raise ValueError(f"What is this token type: {token_type_pred}")

        return token_type_seq, token_seq, pos_seq


    def sample_midi(self, length: int) -> Pianoroll:
        '''
        length: number of frames
        frame_rate: frames per second
        return: pretty_midi.PrettyMIDI
        '''
        notes: list[Note] = []
        token_type_seq, token_seq, pos_seq = [x[0] for x in self.sample(length)] # (length), (length, data_dim), (length)

        for i in range(token_type_seq.shape[0]):
            if token_type_seq[i] == self.NOTE:
                time = pos_seq[i].item()
                pitch = token_seq[i, 0].item()
                velocity = token_seq[i, 1].item()
                assert isinstance(time, int)
                assert isinstance(pitch, int)
                assert isinstance(velocity, int)
                notes.append(Note(onset=time, pitch=pitch+self.pitch_range[0], velocity=velocity))

        return Pianoroll(notes)


def tokenize(pr: Pianoroll, length: int|None=None, pitch_range: list[int]=[21, 109]):
    '''
    token type:
        -1: pad
        0: frame
        1: note
    '''
    current_frame = 0
    tokens = [[0,0]]
    token_types = [0]
    pos = [0]
    for note in pr.notes:
        if note.onset > current_frame:
            for _ in range(note.onset - current_frame):
                current_frame += 1
                tokens.append([0,0])
                token_types.append(0)
                pos.append(current_frame)
            current_frame = note.onset
        assert note.pitch >= pitch_range[0] and note.pitch <= pitch_range[1]
        tokens.append([note.pitch - pitch_range[0], note.velocity])
        token_types.append(1)
        pos.append(note.onset)

    if length is not None and length > len(tokens):
        for _ in range(length - len(tokens)):
            tokens.append([0,0])
            token_types.append(-1)
            pos.append(current_frame)
    tokens = torch.tensor(tokens[:length])
    token_types = torch.tensor(token_types[:length])
    pos = torch.tensor(pos[:length])

    return tokens, token_types, pos


def pad_to_length(x: Tensor, target_length: int, dim: int, pad_value: float):
    padding_shape = list(x.shape)
    padding_shape[dim] = target_length - x.shape[dim]
    return torch.cat([x, torch.full(padding_shape, pad_value)])

def pad_and_stack(batch: list[Tensor], pad_dim: int, pad_value: float=0, stack_dim: int=0, target_length: int|None=None) -> Tensor:
    if target_length is None:
        target_length = max(x.shape[pad_dim] for x in batch)
    return torch.stack([pad_to_length(x, target_length, pad_dim, pad_value) for x in batch], stack_dim)

def collate_fn(batch: list[Pianoroll]):
    tokens_batch = []
    token_types_batch = []
    pos_batch = []
    for pr in batch:
        tokens, token_types, pos = tokenize(pr)
        tokens_batch.append(tokens)
        token_types_batch.append(token_types)
        pos_batch.append(pos)
    tokens_batch = pad_and_stack(tokens_batch, 0)
    token_types_batch = pad_and_stack(token_types_batch, 0)
    pos_batch = pad_and_stack(pos_batch, 0)
    
    return tokens_batch, token_types_batch, pos_batch