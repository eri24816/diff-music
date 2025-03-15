from diff_music.diffusion_model import DiffusionModel
import torch
import torch.nn as nn

class GaussianDiffusionScheduler(nn.Module):
    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.beta = self.get_beta(num_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        

    def get_t(self, t: int) -> float:
        return t / self.num_steps

class GaussianDiffusion(DiffusionModel):
    def __init__(self, num_steps: int):
        super().__init__(num_steps)

    def sample_x_T(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape)
    
    def forward_one_step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_t + torch.randn_like(x_t) * torch.sqrt(t)

