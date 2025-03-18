from dataclasses import dataclass
from music_data_analysis import Note, Pianoroll
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from diff_music.models.util import pad_and_stack

from .feature_extractor import FeatureExtractor

from .pe import sinusoidal_positional_encoding, binary_positional_encoding, one_hot_positional_encoding
from .representation import SymbolicRepresentation

def nucleus_sample(logits: torch.Tensor, p: float) -> torch.Tensor:
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


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_hidden_layers):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
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
        velocity_loss: Tensor
        total_loss: Tensor
        pitch_acc: float
        velocity_acc: float
        token_type_acc: float

    def __init__(self, params: Params):
        super().__init__()
        self.hidden_dim = params.hidden_dim
        self.num_layers = params.num_layers
        self.pitch_range = params.pitch_range
        self.num_pitch = params.pitch_range[1] - params.pitch_range[0]
        self.feature_extractor = FeatureExtractor(FeatureExtractor.Params(
            dim=params.hidden_dim,
            num_layers=params.num_layers,
            pitch_range=params.pitch_range,
            max_len=params.max_len,
            reduce=False,
        ))
        self.token_type_classifier = nn.Linear(params.hidden_dim, 2)
        self.pitch_classifier = nn.Linear(params.hidden_dim, self.num_pitch)
        self.out_pitch_emb = nn.Embedding(self.num_pitch, params.hidden_dim)
        self.velocity_classifier = MLP(params.hidden_dim, 256, 128, 0)

        sinusoidal_pe = sinusoidal_positional_encoding(params.max_len, params.hidden_dim-5-32)
        binary_pe = binary_positional_encoding(params.max_len, 5)
        one_hot_pe = one_hot_positional_encoding(params.max_len, 32)

        pe = torch.cat([binary_pe, one_hot_pe, sinusoidal_pe], dim=1) # (max_len, hidden_dim)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def to(self, device: torch.device):
        super().to(device)
        self.device = device

    def forward(self, x: SymbolicRepresentation, prompt: SymbolicRepresentation|None=None, condition: torch.Tensor|None=None) -> Loss:

        if prompt is not None:
            feature_extractor_input = prompt + x[:,:-1]
        else:
            feature_extractor_input = x[:,:-1]

        feature = self.feature_extractor(feature_extractor_input, condition) # (batch_size, num_tokens-1, hidden_dim)

        if prompt is not None:
            feature = feature[:, prompt.length:]

        target = x[:,1:] # (batch_size, num_tokens-1)
        
        token_type_logits = self.token_type_classifier(feature) # (batch_size, num_tokens-1, 2)
        pitch_logits = self.pitch_classifier(feature) # (batch_size, num_tokens-1, vocab_size)

        # velocity classifier can see ground truth pitch
        velocity_logits = self.velocity_classifier(feature + self.out_pitch_emb(target.pitch)) # (batch_size, num_tokens-1, 128)

        assert token_type_logits.shape == target.token_type.shape
        assert pitch_logits.shape == target.pitch.shape
        assert velocity_logits.shape == target.velocity.shape

        token_type_loss = torch.nn.functional.cross_entropy(token_type_logits.transpose(1, 2), target.token_type, ignore_index=-1, reduction="mean")
        pitch_loss = (torch.nn.functional.cross_entropy(pitch_logits.transpose(1, 2), target.pitch, ignore_index=-1, reduction="none") * target.is_note).mean()
        velocity_loss = (torch.nn.functional.cross_entropy(velocity_logits.transpose(1, 2), target.velocity, ignore_index=-1, reduction="none", label_smoothing=0.05) * target.is_note).mean()
        total_loss = token_type_loss + pitch_loss + velocity_loss

        token_type_acc = (token_type_logits.detach().argmax(dim=2) == target.token_type).mean().item()
        pitch_acc = ((pitch_logits.detach().argmax(dim=2) == target.pitch) * target.is_note).mean().item()
        velocity_acc = ((velocity_logits.detach().argmax(dim=2) == target.velocity) * target.is_note).mean().item()

        return MidiLikeTransformer.Loss(
            total_loss=total_loss,
            token_type_loss=token_type_loss,
            pitch_loss=pitch_loss,
            velocity_loss=velocity_loss,
            token_type_acc=token_type_acc,
            pitch_acc=pitch_acc,
            velocity_acc=velocity_acc,
        )

    def sample(self, length: int, max_iter: int|None=None):
        '''
        autoregressive sampling
        length: number of frames
        '''

        if max_iter is None:
            max_iter = length * 5

        output = SymbolicRepresentation(device=self.device)

        current_pos = 0

        pbar = tqdm(range(length))
        for i in range(max_iter):
            feature = self.feature_extractor(output) # (b=1, length, hidden_dim)
            token_type_logits = self.token_type_classifier(feature[:, -1, :])[0] # (class)
            token_type_pred = nucleus_sample(token_type_logits, 0.95) # scalar
            
            if token_type_pred == self.FRAME: # frame
                current_pos += 1

                output.add_frame()

                pbar.update(1)

                if current_pos >= length:
                    break

            elif token_type_pred == self.NOTE: # note
                # sample pitch
                pitch_logits = self.pitch_classifier(feature[:, -1, :])[0] # (class)
                pitch_pred = nucleus_sample(pitch_logits, 0.95) # scalar

                # sample velocity
                velocity_logits = self.velocity_classifier(feature[:, -1, :] + self.out_pitch_emb(pitch_pred.unsqueeze(0)))[0] # (class)
                velocity_pred = nucleus_sample(velocity_logits, 0.95) # scalar

                output.add_note(pitch_pred, velocity_pred)

            else:
                raise ValueError(f"What is this token type: {token_type_pred}")

        return output


    def sample_pianoroll(self, length: int) -> Pianoroll:
        '''
        length: number of frames
        frame_rate: frames per second
        return: pretty_midi.PrettyMIDI
        '''
        notes: list[Note] = []
        result = self.sample(length)

        with torch.no_grad():
            for i in range(result.token_type.shape[0]):
                if result.token_type[i] == self.NOTE:
                    time = result.pos[i].item()
                    pitch = result.token[i, 0].item()
                    velocity = result.token[i, 1].item()
                    assert isinstance(time, int)
                    assert isinstance(pitch, int)
                    assert isinstance(velocity, int)
                    notes.append(Note(onset=time, pitch=pitch+self.pitch_range[0], velocity=velocity))

        return Pianoroll(notes)


def tokenize(pr: Pianoroll, max_length: int|None=None, pitch_range: list[int]=[21, 109], pad: bool=False):
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

    for current_frame in range(current_frame+1, pr.duration+1):
        tokens.append([0,0])
        token_types.append(0)

        # this is a special case that logically the last pos should equal to duration  
        # but in practice this may break the model's embedding module, so it's set to 0
        # it's okay to do this because this entry doesn't affect the model's output
        if current_frame == pr.duration: 
            current_frame = 0
            
        pos.append(current_frame)

    if pad and max_length is not None and max_length > len(tokens):
        for _ in range(max_length - len(tokens)):
            tokens.append([0,0])
            token_types.append(-1)
            pos.append(current_frame)

    if max_length is not None and len(tokens) > max_length:
        print(f"Truncating the input from {len(tokens)} to {max_length}")
        tokens = tokens[:max_length]
        token_types = token_types[:max_length]
        pos = pos[:max_length]

    tokens = torch.tensor(tokens)
    token_types = torch.tensor(token_types)
    pos = torch.tensor(pos)

    return tokens, token_types, pos


def collate_fn(batch: list[Pianoroll], max_tokens: int|None=None) -> SymbolicRepresentation:
    tokens_batch = []
    token_types_batch = []
    pos_batch = []
    for pr in batch:
        tokens, token_types, pos = tokenize(pr, max_length=max_tokens)
        tokens_batch.append(tokens)
        token_types_batch.append(token_types)
        pos_batch.append(pos)
    tokens_batch = pad_and_stack(tokens_batch, 0)
    token_types_batch = pad_and_stack(token_types_batch, 0, pad_value=-1)
    pos_batch = pad_and_stack(pos_batch, 0)
    
    return SymbolicRepresentation(tokens_batch, token_types_batch, pos_batch)