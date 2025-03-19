from dataclasses import dataclass
from typing import Literal
import torch.nn as nn
from torch import Tensor
from .feature_extractor import FeatureExtractor
from .midilike_transformer import MidiLikeTransformer, SymbolicRepresentation
class VaeBottleneck(nn.Module):

    @dataclass
    class Params:
        beta: float
        dim: int

    def __init__(self, params: Params):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor):
        return x #TODO

class EncoderDecoder(nn.Module):
    '''
    input tokens: b, l
    output: b, d
    '''

    @dataclass
    class BottleneckParams:
        type: Literal['identity', 'vae']
        vae_params: VaeBottleneck.Params|None = None

    @dataclass
    class EncoderParams(FeatureExtractor.Params):
        max_len: int|None = None # will be set by EncoderDecoder
        reduce: bool = True

    @dataclass
    class DecoderParams(MidiLikeTransformer.Params):
        max_len: int|None = None # will be set by EncoderDecoder

    @dataclass
    class Params:
        target_length: int
        prompt_length: int
        encoder_params: 'EncoderDecoder.EncoderParams'
        decoder_params: 'EncoderDecoder.DecoderParams'
        bottleneck_params: 'EncoderDecoder.BottleneckParams'

    def __init__(self, params: Params):
        super().__init__()

        self.target_length = params.target_length
        self.prompt_length = params.prompt_length

        params.encoder_params.max_len = params.target_length
        self.encoder = FeatureExtractor(params.encoder_params, is_causal=False)

        params.decoder_params.max_len = params.target_length + params.prompt_length
        self.decoder = MidiLikeTransformer(params.decoder_params)
        
        if params.bottleneck_params.type == 'vae':
            assert params.bottleneck_params.vae_params is not None, "vae_params must be provided if bottleneck type is 'vae'"
            self.bottleneck = VaeBottleneck(params.bottleneck_params.vae_params)
        else:
            self.bottleneck = lambda x: x

    def forward(self, target: SymbolicRepresentation, prompt: SymbolicRepresentation):
        # first, encode target

        # target's pos starts at prompt.max_pos+1, but for the encoder, it should start at 0
        encoder_input = target.clone().shift_pos(-(prompt.max_pos+1))
        target_embed = self.encoder(encoder_input)
        latent = self.bottleneck(target_embed)

        # then, predict target
        loss = self.decoder.forward(x=target, prompt=prompt, condition=latent)
        return loss
