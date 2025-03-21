from dataclasses import dataclass
from typing import Callable, Literal, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .feature_extractor import FeatureExtractor
from .midilike_transformer import MidiLikeTransformer, SymbolicRepresentation


@dataclass
class BottleneckLoss:
    total_loss: Tensor


class VaeBottleneck(nn.Module):

    @dataclass
    class Loss:
        total_loss: Tensor
        kl_loss: Tensor
        beta: float
        rms_mu: Tensor
        rms_std: Tensor

    def cyclic_beta(self, step: int, beta_cycle_steps: int, beta_start_step: int):
        if step < beta_start_step:
            return 0
        return min(
            1, 2 * ((step - beta_start_step) % beta_cycle_steps / beta_cycle_steps)
        )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        beta: float,
        beta_cycle_steps: int,
        beta_start_step: int,
    ):
        super().__init__()
        self.beta = beta
        self.beta_cycle_steps = beta_cycle_steps
        self.beta_start_step = beta_start_step
        self.mu_proj = nn.Linear(input_dim, output_dim)
        self.logvar_proj = nn.Linear(input_dim, output_dim)
        self.step = 0

    def forward(self, x: Tensor):
        """
        x: b, d
        """
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        kl_loss = (0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(dim=-1)).mean()
        beta = (
            self.cyclic_beta(self.step, self.beta_cycle_steps, self.beta_start_step)
            * self.beta
        )
        total_loss = kl_loss * beta
        rms_mu = mu.pow(2).mean().sqrt()
        rms_std = std.pow(2).mean().sqrt()
        return mu + eps * std, self.Loss(
            total_loss=total_loss,
            kl_loss=kl_loss,
            beta=beta,
            rms_mu=rms_mu,
            rms_std=rms_std,
        )


def identity_bottleneck(x: Tensor) -> Tuple[Tensor, BottleneckLoss]:
    return x, BottleneckLoss(total_loss=torch.tensor(0, device=x.device))

class EncoderDecoder(nn.Module):
    """
    input tokens: b, l
    output: b, d
    """

    @dataclass
    class VaeParams:
        beta: float
        latent_dim: int
        beta_cycle_steps: int
        beta_start_step: int

    @dataclass
    class BottleneckParams:
        type: Literal["identity", "vae"]
        vae_params: "EncoderDecoder.VaeParams | None" = None

    @dataclass
    class EncoderParams(FeatureExtractor.Params):
        num_pos: int | None = None  # will be set by EncoderDecoder
        reduce: bool = True

    @dataclass
    class DecoderParams(MidiLikeTransformer.Params):
        num_pos: int | None = None  # will be set by EncoderDecoder
        condition_dim: int | None = None  # will be set by EncoderDecoder

    @dataclass
    class Params:
        target_length: int
        prompt_length: int
        encoder_params: "EncoderDecoder.EncoderParams"
        decoder_params: "EncoderDecoder.DecoderParams"
        bottleneck_params: "EncoderDecoder.BottleneckParams"

    @dataclass
    class Loss:
        total_loss: Tensor
        reconstruction: MidiLikeTransformer.Loss
        bottleneck: BottleneckLoss

    def __init__(self, params: Params):
        super().__init__()

        self.target_length = params.target_length
        self.prompt_length = params.prompt_length

        params.encoder_params.num_pos = params.target_length
        self.encoder = FeatureExtractor(params.encoder_params, is_causal=False)

        params.decoder_params.num_pos = params.target_length + params.prompt_length
        params.decoder_params.condition_dim = (
            params.bottleneck_params.vae_params.latent_dim
        )
        self.decoder = MidiLikeTransformer(params.decoder_params)

        if params.bottleneck_params.type == "vae":
            self.bottleneck: Callable[[Tensor], Tuple[Tensor, BottleneckLoss]]
            assert params.bottleneck_params.vae_params is not None, (
                "vae_params must be provided if bottleneck type is 'vae'"
            )
            self.bottleneck = VaeBottleneck(
                input_dim=params.encoder_params.dim,
                output_dim=params.bottleneck_params.vae_params.latent_dim,
                beta=params.bottleneck_params.vae_params.beta,
                beta_cycle_steps=params.bottleneck_params.vae_params.beta_cycle_steps,
                beta_start_step=params.bottleneck_params.vae_params.beta_start_step,
            )
        else:
            self.bottleneck = identity_bottleneck

    def set_step(self, step: int):
        if isinstance(self.bottleneck, VaeBottleneck):
            self.bottleneck.step = step

    def forward(self, target: SymbolicRepresentation, prompt: SymbolicRepresentation):
        # first, encode target

        target_embed = self.encoder(target)  # b, d
        latent, bottleneck_loss = self.bottleneck(target_embed)  # b, d

        # then, predict target
        reconstruction_loss = self.decoder.forward(
            x=target, prompt=prompt, condition=latent
        )
        return EncoderDecoder.Loss(
            total_loss=reconstruction_loss.total_loss + bottleneck_loss.total_loss,
            reconstruction=reconstruction_loss,
            bottleneck=bottleneck_loss,
        )

    @torch.no_grad()
    def reconstruct(
        self, target: SymbolicRepresentation, prompt: SymbolicRepresentation
    ):
        latent, _ = self.bottleneck(self.encoder(target))
        return self.decoder.sample(
            duration=prompt.duration + target.duration,
            prompt=prompt,
            condition=latent,
        )

    def encode(self, x: SymbolicRepresentation):
        return self.bottleneck(self.encoder(x))[0]

    @torch.no_grad()
    def sample(self, prompt: SymbolicRepresentation, duration: int, latent: Tensor):
        return self.decoder.sample(
            duration=prompt.duration + duration, prompt=prompt, condition=latent
        )
