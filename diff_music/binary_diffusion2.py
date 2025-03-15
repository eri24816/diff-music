# modified from https://github.com/ZeWang95/BinaryLatentDiffusion, MIT license

from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass


from diff_music.diffusion_model import DiffusionModel



class BinaryDiffusion(DiffusionModel):

    @dataclass
    class Params:
        total_steps: int
        loss_final: str
        use_softmax: bool
        beta_type: str
        p_flip: bool
        aux: float
        use_label: bool
        guidance: float
        channels: int
        focal_gamma: float = 0
        focal_alpha: float = 0.5
        gamma: float = 0.5

    def __init__(self, h: Params, denoise_fn: Callable):
        super().__init__(h.total_steps)

        self._denoise_fn = denoise_fn

        self.loss_final = h.loss_final
        self.use_softmax = h.use_softmax

        self.scheduler = NoiseScheduler(self.num_steps, beta_type=h.beta_type, gamma=h.gamma)
        self.p_flip = h.p_flip
        self.focal_gamma = h.focal_gamma
        self.focal_alpha = h.focal_alpha
        self.aux = h.aux
        self.use_label = h.use_label
        self.guidance = h.guidance
        self.channels = h.channels
        self.gamma = h.gamma

    def sample_x_T(self, shape: tuple[int, ...]) -> torch.Tensor:
        p_xT = self.gamma * torch.ones(shape, device=self.device)
        return torch.bernoulli(p_xT)

    def forward_from_x0(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta = self.scheduler.get_beta(t)
        p_xt = x_0 * (1-beta) + self.gamma * beta
        return torch.bernoulli(p_xt)
    
    def forward_one_step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        k = self.scheduler.get_k(t)
        b = self.scheduler.get_b(t)
        print(k, b)
        p_xt = x_t * k + b
        return torch.bernoulli(p_xt)
        

    def forward(self, x_0: torch.Tensor) -> dict:

        t = self.sample_t(x_0.shape[0])
        x_t = self.forward_from_x0(x_0, t)

        x_0_hat_logits = self._denoise_fn(x_t, time_steps=(t-1)/self.num_steps)

        if self.p_flip:
            if self.focal_gamma > 0:
                x_0_ = torch.logical_xor(x_0, x_t)*1.0
                kl_loss = focal_loss(x_0_hat_logits, x_0_, gamma=self.focal_gamma)
                x_0_hat_logits = x_t * ( - x_0_hat_logits) + (1 - x_t) * x_0_hat_logits
            else:
                x_0_hat_logits = x_t * ( - x_0_hat_logits) + (1 - x_t) * x_0_hat_logits
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        else:
            if self.focal_gamma > 0:
                kl_loss = focal_loss(x_0_hat_logits, x_0, self.focal_alpha, gamma=self.focal_gamma)
            else:
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        if self.loss_final == 'weighted':
            weight = (1 - ((t-1) / self.num_steps)).view(-1, 1, 1)
        elif self.loss_final == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
        loss = (weight * kl_loss).mean()
        kl_loss = kl_loss.mean()

        stats = {'loss': loss, 'bce_loss': kl_loss}

        return stats
    
    def backward_one_step(self, x_t: torch.Tensor, t: int, temp: float = 1.0) -> tuple[torch.Tensor, dict]:
        b = x_t.shape[0]
        t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)
        
        x_0_logits = self._denoise_fn(x_t, time_steps=(t_batch-1)/self.num_steps)
        x_0_logits = x_0_logits / temp
        x_0_probs = torch.sigmoid(x_0_logits)


        if self.p_flip:
            x_0_probs =  x_t * (1 - x_0_probs) + (1 - x_t) * x_0_probs

        if not t == 1:
            x_tm1_probs = get_p_xtm1_from_x0_prediction(x_0_probs, x_t, t_batch, self.scheduler)
            x_tm1_pred = torch.bernoulli(x_tm1_probs)
        
        else:
            x_tm1_probs = x_0_probs
            x_tm1_pred = (x_0_logits > 0) * 1.0

        return x_tm1_pred, {'x_0_prob': x_0_probs, 'x_tm1_prob': x_tm1_probs}

def get_p_xtm1_from_x0_prediction(x_0_prediction: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, scheduler: 'NoiseScheduler'):

    beta = scheduler.get_beta(t)
    k = scheduler.get_k(t)
    k_tm1 = scheduler.get_k(t-1)
    b = scheduler.get_b(t)
    b_tm1 = scheduler.get_b(t-1)
    gamma = scheduler.gamma

    def bernoulli_posterior(x, positive_prob):
        '''
        This function computes p(x|conditions) for a variable x that 
        follows a Bernoulli distribution B(x; positive_prob(conditions)), where positive_prob is a function of the conditions.
        '''
        return x*(positive_prob) + (1-x)*(1-positive_prob)
    
    
    def get_q_xtm1_xt_x0(x_t, x_0):
        '''
        Return q(x^{t-1}=1|x^t, x^0) for given x^t and x^0
        '''
        # eq. 4
        # q(x^t=1|x^{t-1}) = B(xt; xtm1(1-betat)+gamma*betat) | xtm1 = 1
        q_xt_xtm1 = bernoulli_posterior(x_t, 1*(1-beta)+gamma*beta)
        
        # eq. 5
        # q(z^t|z^0) = B(zt; kt*z0+bt)
        q_xt_x0 = bernoulli_posterior(x_t, k*x_0+b)
        
        # eq. 5
        # q(x^{t-1}=1|x^0) = B(x^{t-1}; k^{t-1}*x^0+b^{t-1}) | x^{t-1} = 1
        q_xtm1_x0 = bernoulli_posterior(1, k_tm1*x_0+b_tm1)

        # q(x^{t-1}|x^t, x^0) =  q(x^t|x^{t-1}) * q(x^{t-1}|x^0) / q(x^t|x^0)
        q_xtm1_xt_x0 = q_xt_xtm1 * q_xtm1_x0 / q_xt_x0
        return q_xtm1_xt_x0
    
    q_xtm1_if_x0_eq_1 = get_q_xtm1_xt_x0(x_t, 1)
    q_xtm1_if_x0_eq_0 = get_q_xtm1_xt_x0(x_t, 0)

    p_xtm1_eq_1_given_xt = q_xtm1_if_x0_eq_1 * x_0_prediction + q_xtm1_if_x0_eq_0 * (1-x_0_prediction)
    return p_xtm1_eq_1_given_xt

def collect_batch(value_list: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return value_list.view(-1, *([1]*(t.ndim-1)))

class NoiseScheduler(nn.Module):
    def __init__(self, steps=40, beta_type='linear', gamma = 0.5):
        super().__init__()

        self.gamma = gamma
        if beta_type == 'linear':

            beta = 1 / (1*steps - np.arange(1, steps+1) + 1) 

            k_final = [1.0]
            b_final = [0.0]

            for i in range(steps):
                k_final.append(k_final[-1]*(1-beta[i]))
                b_final.append((1-beta[i]) * b_final[-1] + gamma * beta[i])

            k_final = k_final[1:]
            b_final = b_final[1:]

        elif beta_type == 'clip_linear':
        
            beta = np.clip(1 / (1*steps - np.arange(1, steps+1) + 1), 0, 0.3)
        
            k_final = [1.0]
            b_final = [0.0]
        
            for i in range(steps):
                k_final.append(k_final[-1]*(1-beta[i]))
                b_final.append((1-beta[i]) * b_final[-1] + gamma * beta[i])
        
            k_final = k_final[1:]
            b_final = b_final[1:]


        elif beta_type == 'cos':

            k_final = np.linspace(0.0, 1.0, steps+1)

            k_final = k_final * np.pi
            k_final = 0.5 + 0.5 * np.cos(k_final)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]
        
        elif beta_type == 'sigmoid':
            
            def sigmoid(x):
                z = 1/(1 + np.exp(-x))
                return z

            def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=0.0):
                # A gamma function based on sigmoid function.
                v_start = sigmoid(start / tau)
                v_end = sigmoid(end / tau)
                output = sigmoid((t * (end - start) + start) / tau)
                output = (v_end - output) / (v_end - v_start)
                return np.clip(output, clip_min, 1.)
            
            k_final = np.linspace(0.0, 1.0, steps+1)
            k_final = sigmoid_schedule(k_final, 0, 3, 0.8)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]


        else:
            raise NotImplementedError
        
        k_final = np.hstack([1, k_final])
        b_final = np.hstack([0, b_final])
        beta = np.hstack([0, beta])
        self.register_buffer('k_final', torch.Tensor(k_final))
        self.register_buffer('b_final', torch.Tensor(b_final))
        self.register_buffer('beta', torch.Tensor(beta))  
        self.register_buffer('cumbeta', torch.cumprod(self.beta, 0))  
        # pdb.set_trace()

        print(f'Noise scheduler with {beta_type}:')

        print('Diffusion 1.0 -> 0.5:')
        data = (1.0 * self.k_final + self.b_final).data.numpy()
        print(' '.join([f'{d:0.4f}' for d in data]))

        print('Diffusion 0.0 -> 0.5:')
        data = (0.0 * self.k_final + self.b_final).data.numpy()
        print(' '.join([f'{d:0.4f}' for d in data]))

        print('Beta:')
        print(' '.join([f'{d:0.4f}' for d in self.beta.data.numpy()]))

    
    
    def one_step(self, x, t):
        beta = collect_batch(self.beta[t], t)
        x = x * (1-beta) + self.gamma * beta
        return x

    def forward(self, x, t):
        k = collect_batch(self.k_final[t], t)
        b = collect_batch(self.b_final[t], t)
        out = k * x + b
        return out
    
    def get_beta(self, t):
        return collect_batch(self.beta[t], t)
    
    def get_b(self, t):
        return collect_batch(self.b_final[t], t)
    
    def get_k(self, t):
        return collect_batch(self.k_final[t], t)
    

def focal_loss(inputs:torch.Tensor, targets:torch.Tensor, alpha:float=-1, gamma:float=1):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets) # tp and tn
    p_t = (1 - p_t) # fp and fn
    p_t = p_t.clamp(min=1e-6, max=(1-1e-6)) # numerical safety
    loss = ce_loss * (p_t ** gamma)
    if alpha == -1:
        neg_weight = targets.sum((-1, -2))
        neg_weight = neg_weight / targets[0].numel()
        neg_weight = neg_weight.view(-1, 1, 1)
        alpha_t = (1 - neg_weight) * targets + neg_weight * (1 - targets)
        loss = alpha_t * loss
    elif alpha > 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss
