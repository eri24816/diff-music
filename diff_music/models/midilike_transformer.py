from typing import Generator
import einops
from music_data_analysis import Pianoroll
import torch
import torch.nn as nn
import math
import pretty_midi
from vqpiano.models.utils import nucleus_sample
from vqpiano.models.vqgan import VQEncoder
from vqpiano.utils.vocab import Vocabulary, WordArray


class Tokenizer:
    def __init__(self,n_pitch: int, n_time: int):
        '''
        Tokenize a patch into a sequence of tokens
        '''
        self.vocab = Vocabulary(
            [
                "pad",
                "frame",
                WordArray("pitch", {"type":["pitch"],"value": range(n_pitch)}),
            ]
        )

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

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self.get_pe(max_len, d_model))

    def get_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        return self.pe[:, :x.shape[1]]

class MidiLikeTransformer(nn.Module):
    '''
    Autoregressive decoder
    Target (token sequence) shape: (batch_size, num_tokens)
    Condition shape: (batch_size, hidden_dim)

    Forward pass:
    Input: target (x_0=[START], x_1, ..., x_T-1)
        shape: (batch_size, num_tokens)
    Condition: (c_0, c_1, ..., c_{n_patches})
        shape: (batch_size, num_patches, hidden_dim)
    Output: logits of x_1, x_2, ..., x_T
        shape: (batch_size, num_tokens, vocab_size)
    '''
    def __init__(self, hidden_dim, num_layers, cond_dim, tokenizer: Tokenizer, num_patches_h: int, num_patches_w: int, patch_height: int, patch_width: int, pitch_range: list[int], fs: int):
        super().__init__()
        self.vocab_size = len(tokenizer.vocab)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cond_dim = cond_dim
        self.token_emb = nn.Embedding(self.vocab_size, hidden_dim)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_dim)
        self.cond_input_proj = nn.Linear(cond_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers,
        )
        self.output_proj = nn.Linear(hidden_dim, self.vocab_size)
        self.tokenizer = tokenizer
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.pitch_range = pitch_range
        self.fs = fs

    def forward(self, x: torch.Tensor, pos: torch.Tensor, c: torch.Tensor):
        '''
        x: (batch_size, num_tokens)
        pos: (batch_size, num_tokens)
        c: (batch_size, num_patches, cond_dim)
        '''
        B = x.shape[0]


        assert self.cond_dim == c.shape[2], f"c.shape[ = {c.shape} dim 2 does not match self.cond_dim = {self.cond_dim}"
        c_to_stack = []
        for i in range(B):
            c_to_stack.append(c[i, pos[i]])
        c = torch.stack(c_to_stack) # (batch_size, num_tokens, cond_dim)

        c = self.cond_input_proj(c) # (batch_size, num_tokens, hidden_dim)
        x = self.token_emb(x) # (batch_size, num_tokens, hidden_dim)
        x = x + c # (batch_size, num_tokens, hidden_dim)
        x = x + self.pos_emb(x) # (batch_size, num_tokens, hidden_dim)
        x = self.transformer.forward(x, nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device), is_causal=True) # (batch_size, num_tokens, hidden_dim)
        x = self.output_proj(x) # (batch_size, num_tokens, vocab_size)

        assert x.shape == (B, x.shape[1], self.vocab_size)

        return x

    def sample(self, c: torch.Tensor, max_len: int):
        '''
        autoregressive sampling
        c: (num_patches, cond_dim)
        max_len: maximum length of the sequence
        return: (max_len)
        '''
        x = []
        pos = []
        tokens: list[dict|str] = []

        # start_patch 0
        x.append(self.tokenizer.vocab.get_idx({"type": "start_patch", "value": 0}))
        pos.append(0)
        tokens.append({"type": "start_patch", "value": 0})

        while len(tokens) < max_len:

            # return if pos >= num_patches
            if pos[-1] >= self.num_patches_h * self.num_patches_w:
                break

            logits = self(torch.tensor(x,device=c.device).unsqueeze(0),
                torch.tensor(pos,device=c.device).unsqueeze(0),
                c.unsqueeze(0)
            )[0, -1, :]
            if isinstance(tokens[-1], dict) and tokens[-1]["type"] == "time":
                possible_tokens = ["pitch"]
            else:
                possible_tokens = ["time", "start_patch"]
            logits_mask = self.tokenizer.vocab.get_mask([possible_tokens])[0].to(logits.device)

            logits = logits + logits_mask
            sampled_idx = nucleus_sample(logits, p=0.95).item()
            x.append(sampled_idx)
            tokens.append(self.tokenizer.vocab.get_token(sampled_idx))

            if isinstance(tokens[-1], dict) and tokens[-1]["type"] == "start_patch":
                pos.append(pos[-1] + 1)
                tokens[-1]["value"] = pos[-1] # the start_patch token may have wrong value sampled. correct it.
            else:
                pos.append(pos[-1])
        return tokens

    def sample_batch(self, c: torch.Tensor, max_len: int|None=None) -> list[list[dict|str]]:
        '''
        autoregressive sampling
        c: (batch_size, num_patches, cond_dim)
        max_len: maximum length of the sequence
        return: (batch_size, max_len)
        '''
        B = c.shape[0]
        if max_len is None:
            max_len = self.tokenizer.seq_len

        tokens_batch = []
        for i in range(B):
            tokens_batch.append(self.sample(einops.rearrange(c[i], "c h w -> (h w) c"), max_len))
        return tokens_batch

    def tokens_to_midi(self, tokens: list[dict|str]) -> pretty_midi.PrettyMIDI:
        '''
        tokens: list of tokens
        return: pretty_midi.PrettyMIDI
        '''
        midi = pretty_midi.PrettyMIDI()
        track = pretty_midi.Instrument(program=0)
        for time, pitch in self.tokens_to_notes(tokens):
            time_in_sec = time  / self.fs
            pitch = pitch + self.pitch_range[0]
            track.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=time_in_sec, end=time_in_sec+0.2))
        midi.instruments.append(track)
        return midi

    def tokens_to_image(self, tokens: list[dict|str]) -> torch.Tensor:
        '''
        tokens: list of tokens
        return: torch.Tensor (patch_height, patch_width)
        '''
        image = torch.zeros(self.patch_height*self.num_patches_h, self.patch_width*self.num_patches_w)
        for time, pitch in self.tokens_to_notes(tokens):
            image[pitch, time] = 1
        return image

    def tokens_to_notes(self, tokens: list[dict|str]) -> Generator[tuple[int,int], None, None]:
        '''
        tokens: list of tokens
        return: generator of (time, pitch) pairs in pixles
        '''
        patch_shift_h = [] # 0,0,0,1,1,1,2,2,2
        patch_shift_w = [] # 0,1,2,0,1,2,0,1,2
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                patch_shift_h.append(i*self.patch_height)
                patch_shift_w.append(j*self.patch_width)

        current_patch_idx = 0
        time = 0
        for token in tokens:
            assert isinstance(token, dict)
            if token["type"] == "start_patch":
                current_patch_idx = token["value"]
            elif token["type"] == "time":
                time = token["value"]
            elif token["type"] == "pitch":
                pitch = token["value"]
                real_time = time + patch_shift_w[current_patch_idx]
                real_pitch = pitch + patch_shift_h[current_patch_idx]
                yield (real_time, real_pitch)

class VQAutoEncoderWithARDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: tuple[int, int],
        num_features: int,
        channel_mults: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        codebook_emb_dim: int,
        decoder_hidden_dim: int,
        decoder_num_layers: int,
        pitch_range: list[int],
        sampling_frequency: int,
        seq_len: int,
        patch_config: dict,
        vq_args: dict = {},
        stride_vertical: list[int]|None=None,
        stride_horizontal: list[int]|None=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults
        self.in_size = in_size
        self.attn_resolutions = attn_resolutions

        num_patches_h = patch_config["num_h"]
        num_patches_w = patch_config["num_w"]
        patch_height = patch_config["height"]
        patch_width = patch_config["width"]
        num_patches = num_patches_h * num_patches_w


        self.encoder = VQEncoder(
            in_size=in_size,
            in_channels=in_channels,
            num_features=num_features,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            codebook_emb_dim=codebook_emb_dim,
            vq_args=vq_args,
            stride_vertical=stride_vertical,
            stride_horizontal=stride_horizontal
        )

        self.tokenizer = Tokenizer(n_pitch=patch_height, n_time=patch_width, num_patches=num_patches, seq_len=seq_len)

        self.decoder = ARDecoder(
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            cond_dim=codebook_emb_dim,
            tokenizer=self.tokenizer,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            patch_height=patch_height,
            patch_width=patch_width,
            pitch_range=pitch_range,
            fs=sampling_frequency,
        )


    def forward(self, pr: torch.Tensor, target_idx: torch.Tensor, pos: torch.Tensor):
        c, latent_idx, commitment_loss = self.encoder(pr)
        # c: (batch_size, codebook_emb_dim, num_patches_h, num_patches_w)
        c = einops.rearrange(c, "b c h w -> b (h w) c") # (batch_size, num_patches, codebook_emb_dim)
        logits = self.decoder(target_idx, pos, c)

        return logits, latent_idx, commitment_loss



if __name__ == "__main__":
    decoder = ARDecoder(hidden_dim=128,
                        cond_dim=128,
                        tokenizer=Tokenizer(n_pitch=12, n_time=64, num_patches=64, seq_len=200),
                        num_patches_h=8,
                        num_patches_w=8,
                        pitch_range=[17,113],
                        fs=128,
                        patch_height=12,
                        patch_width=64,
                        num_layers=4)
    c = torch.randn(1, 64, 128)

    samples = decoder.sample_batch(c)
    for sample in samples:
        midi = decoder.tokens_to_midi(sample)
        midi.write("sample.mid")
        image = decoder.tokens_to_image(sample)
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()
