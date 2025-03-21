from music_data_analysis import Note, Pianoroll
import torch
from torch import Tensor

from diff_music.models.util import cat_to_right

Tensorable = Tensor | int | float | list[int | float]


class SymbolicRepresentation:
    PAD = -1
    FRAME = 0
    NOTE = 1

    @classmethod
    def cat(cls, batch: list["SymbolicRepresentation"], dim: int = 0):
        return cls(
            token=torch.cat([b.token for b in batch], dim=dim),
            token_type=torch.cat([b.token_type for b in batch], dim=dim),
            pos=torch.cat([b.pos for b in batch], dim=dim),
        )

    def __init__(
        self,
        token: Tensorable | None = None,
        token_type: Tensorable | None = None,
        pos: Tensorable | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        token: (batch_size, length, data_dim)
        token_type: (batch_size, length)
        pos: (batch_size, length)
        """
        self.device = device

        if token is None or token_type is None or pos is None:
            assert token is None and token_type is None and pos is None, (
                "if one argument is provided, all arguments must be provided"
            )
            token = torch.zeros(1, 1, dtype=torch.long, device=device)
            token_type = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
            pos = torch.tensor([[0]], dtype=torch.long, device=device)

        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token)
        if not isinstance(token_type, torch.Tensor):
            token_type = torch.tensor(token_type)
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos)

        assert token.shape[0:1] == token_type.shape[0:1] == pos.shape[0:1], (
            "batch size or length mismatch"
        )

        self.token = token
        self.token_type = token_type
        self.pos = pos

    @property
    def is_frame(self):
        return self.token_type == self.FRAME

    @property
    def is_note(self):
        return self.token_type == self.NOTE

    @property
    def is_pad(self):
        return self.token_type == self.PAD

    @property
    def pitch(self):
        return self.token[:, :, 0]

    @property
    def velocity(self):
        return self.token[:, :, 1]

    @property
    def length(self):
        return self.token.shape[1]

    @property
    def max_pos(self):
        return int(self.pos.max().item())

    @property
    def duration(self):
        if self.length == 0:
            return 0
        return int(self.pos.max().item() + 1)

    @property
    def batch_size(self):
        return self.token.shape[0]
    
    def __add__(self, other: 'SymbolicRepresentation'):
        return SymbolicRepresentation(
            cat_to_right(self.token, other.token, dim=1),
            cat_to_right(self.token_type, other.token_type, dim=1),
            cat_to_right(self.pos, other.pos + self.duration * ~other.is_pad, dim=1),
        )
    

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: slice|tuple[slice, ...]):
        new_token = self.token[index]
        new_token_type = self.token_type[index]
        new_pos = self.pos[index]
        
        assert new_token.ndim == self.token.ndim
        assert new_token_type.ndim == self.token_type.ndim
        assert new_pos.ndim == self.pos.ndim
    
        return SymbolicRepresentation(new_token, new_token_type, new_pos)
    
    def to(self, device: torch.device):
        self.token = self.token.to(device)
        self.token_type = self.token_type.to(device)
        self.pos = self.pos.to(device)
        return self
    
    def add_frame(self):
        self.pos = cat_to_right(self.pos, self.pos.max() + 1, dim=1)
        self.token = cat_to_right(self.token, [0,0], dim=1)
        self.token_type = cat_to_right(self.token_type, 0, dim=1)
        return self
    
    def add_note(self, pitch: int|Tensor, velocity: int|Tensor):
        self.pos = cat_to_right(self.pos, self.pos.max(), dim=1)
        self.token = cat_to_right(self.token, [pitch, velocity], dim=1)
        self.token_type = cat_to_right(self.token_type, self.NOTE, dim=1)
        return self
    
    def add_pad(self):
        self.pos = cat_to_right(self.pos, 0, dim=1)
        self.token = cat_to_right(self.token, [0,0], dim=1)
        self.token_type = cat_to_right(self.token_type, self.PAD, dim=1)
        return self
    
    def clone(self):
        return SymbolicRepresentation(self.token.clone(), self.token_type.clone(), self.pos.clone())
    
    def shift_pos(self, shift: int):
        self.pos = self.pos + shift * ~self.is_pad
        return self

    def to_pianoroll(self, min_pitch: int, batch_item: int = 0) -> Pianoroll:
        notes: list[Note] = []
        for i in range(self.token_type.shape[1]):
            if self.token_type[batch_item, i] == self.NOTE:
                time = self.pos[batch_item, i].item()
                pitch = self.token[batch_item, i, 0].item()
                velocity = self.token[batch_item, i, 1].item()
                assert isinstance(time, int)
                assert isinstance(pitch, int)
                assert isinstance(velocity, int)
                notes.append(
                    Note(onset=time, pitch=pitch + min_pitch, velocity=velocity)
                )

        return Pianoroll(notes)
