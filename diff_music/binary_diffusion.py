# modified from https://github.com/ZeWang95/BinaryLatentDiffusion, MIT license

import einops
import numpy as np
import torch
import torch.nn.functional as F
import pdb
from torch import Tensor, nn
from dataclasses import dataclass

from tqdm import tqdm



class BinaryDiffusion(nn.Module):

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
    def __init__(self, h: Params, denoise_fn):
        super().__init__()

        self.num_timesteps = h.total_steps

        self._denoise_fn = denoise_fn

        self.loss_final = h.loss_final
        self.use_softmax = h.use_softmax

        self.scheduler = NoiseScheduler(self.num_timesteps, beta_type=h.beta_type, gamma=h.gamma)
        self.p_flip = h.p_flip
        self.focal_gamma = h.focal_gamma
        self.focal_alpha = h.focal_alpha
        self.aux = h.aux
        self.use_label = h.use_label
        self.guidance = h.guidance
        self.channels = h.channels
        self.gamma = h.gamma

    def sample_time(self, b, device):
        t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
        return t

    def q_sample(self, x_0, t):
        x_t = self.scheduler(x_0, t) # t >= 1 <=T#
        return x_t

    def _train_loss(self, x_0, label=None, x_ct=None):
        x_0 = x_0 * 1.0
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t = self.sample_time(b, device)

        # make x noisy and denoise
        if x_ct is None:
            x_t = self.q_sample(x_0, t)
        else:
            x_t = self.scheduler.sr_forward(x_0, x_ct, t)

        x_t_in = torch.bernoulli(x_t)
        if label is not None:
            if self.guidance and np.random.random() < 0.1:
                label = None
            x_0_hat_logits = self._denoise_fn(x_t_in, label=label, time_steps=(t-1)/self.num_timesteps) 
        else:
            x_0_hat_logits = self._denoise_fn(x_t_in, time_steps=(t-1)/self.num_timesteps)


        if self.p_flip:
            if self.focal_gamma > 0:
                x_0_ = torch.logical_xor(x_0, x_t_in)*1.0
                kl_loss = focal_loss(x_0_hat_logits, x_0_, gamma=self.focal_gamma)
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
            else:
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        else:
            if self.focal_gamma > 0:
                kl_loss = focal_loss(x_0_hat_logits, x_0, self.focal_alpha, gamma=self.focal_gamma)
            else:
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        if torch.isinf(kl_loss).max():
            pdb.set_trace()

        if self.loss_final == 'weighted':
            weight = (1 - ((t-1) / self.num_timesteps)).view(-1, 1, 1)
        elif self.loss_final == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
        loss = (weight * kl_loss).mean()
        kl_loss = kl_loss.mean()

        # with torch.no_grad():
        #     if self.use_softmax:
        #         acc = (((x_0_hat_logits[..., 1] > x_0_hat_logits[..., 0]) * 1.0 == x_0.view(-1)) * 1.0).sum() / float(x_0.numel())
        #     else:
        #         acc = (((x_0_hat_logits > 0.0) * 1.0 == x_0) * 1.0).sum() / float(x_0.numel())

        if self.aux > 0:
            ftr = (((t-1)==0)*1.0).view(-1, 1, 1)

            x_0_l = torch.sigmoid(x_0_hat_logits)
            x_0_logits = torch.cat([x_0_l.unsqueeze(-1), (1-x_0_l).unsqueeze(-1)], dim=-1)

            x_t_logits = torch.cat([x_t_in.unsqueeze(-1), (1-x_t_in).unsqueeze(-1)], dim=-1)

            p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t-1)

            q_one_step = self.scheduler.one_step(x_t_logits, t)
            unnormed_probs = p_EV_qxtmin_x0 * q_one_step
            unnormed_probs = unnormed_probs / (unnormed_probs.sum(-1, keepdims=True)+1e-6)
            unnormed_probs = unnormed_probs[...,0]
            
            x_tm1_logits = unnormed_probs * (1-ftr) + x_0_l * ftr
            x_0_gt = torch.cat([x_0.unsqueeze(-1), (1-x_0).unsqueeze(-1)], dim=-1)
            p_EV_qxtmin_x0_gt = self.scheduler(x_0_gt, t-1)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            unnormed_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)
            unnormed_gt = unnormed_gt[...,0]

            x_tm1_gt = unnormed_gt

            if torch.isinf(x_tm1_logits).max() or torch.isnan(x_tm1_logits).max():
                pdb.set_trace()
            aux_loss = F.binary_cross_entropy(x_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), x_tm1_gt.clamp(min=0.0, max=1.0), reduction='none')

            aux_loss = (weight * aux_loss).mean()
            loss = self.aux * aux_loss + loss

        stats = {'loss': loss, 'bce_loss': kl_loss}

        if self.aux > 0:
            stats['aux loss'] = aux_loss
        return stats
    
    
    def sample(self, shape: tuple[int, int, int, int], temp=1.0, sample_steps=None, return_all=False, label=None, mask=None, guidance=None, full=False, pbar=False):
        with torch.no_grad():
            device = 'cuda'

            b = shape[0]
            x_t = torch.bernoulli(self.gamma * torch.ones(shape, device=device))

            if mask is not None:
                m = mask['mask'].unsqueeze(0)
                latent = mask['latent'].unsqueeze(0)
                x_t = latent * m + x_t * (1-m)
            sampling_steps = np.array(range(1, self.num_timesteps+1))

            if sample_steps is not None and sample_steps != self.num_timesteps:
                idx = np.linspace(0.0, 1.0, sample_steps)
                idx = np.array(idx * (self.num_timesteps-1), int)
                sampling_steps = sampling_steps[idx]

            if return_all:
                x_all = [x_t]

            if self.use_label:
                if label is None:
                    label = torch.arange(b, device=device) * 100
                    label = label.long()
                else:
                    label = torch.full((b,), label, device=device, dtype=torch.long)
            
            sampling_steps = sampling_steps[::-1]

            if pbar:
                iterable = tqdm(sampling_steps)
            else:
                iterable = sampling_steps

            for i, t in enumerate(iterable):
                t = torch.full((b,), t, device=device, dtype=torch.long)

                if self.use_label:
                    x_0_logits = self._denoise_fn(x_t, label, time_steps=(t-1)/self.num_timesteps)
                    x_0_logits = x_0_logits / temp
                    if guidance is not None:
                        x_0_logits_uncond = self._denoise_fn(x_t, None, time_steps=(t-1)/self.num_timesteps)
                        x_0_logits_uncond = x_0_logits_uncond / temp

                        x_0_logits = (1 + guidance) * x_0_logits - guidance * x_0_logits_uncond
                else:
                    x_0_logits = self._denoise_fn(x_t, time_steps=(t-1)/self.num_timesteps)
                    x_0_logits = x_0_logits / temp
                    # scale by temperature

                x_0_probs = torch.sigmoid(x_0_logits)


                if self.p_flip:
                    x_0_probs =  x_t * (1 - x_0_probs) + (1 - x_t) * x_0_probs

                if not t[0].item() == 1:
                    # t_p = torch.full((b,), sampling_steps[i+1], device=device, dtype=torch.long)
                    
                    # x_0_probs = torch.cat([x_0_probs.unsqueeze(-1), (1-x_0_probs).unsqueeze(-1)], dim=-1) 
                    # x_t_probs = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)


                    # p_EV_qxtmin_x0 = self.scheduler(x_0_probs, t_p)
                    # q_one_step = x_t_probs

                    # for mns in range(sampling_steps[i] - sampling_steps[i+1]):
                    #     q_one_step = self.scheduler.one_step(q_one_step, t - mns)

                    # unnormed_probs = p_EV_qxtmin_x0 * q_one_step
                    # unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)
                    # unnormed_probs = unnormed_probs[...,0]
                    
                    # x_tm1_logits = unnormed_probs
                    # x_tm1_p = torch.bernoulli(x_tm1_logits)

                    x_tm1_probs = get_p_xtm1_from_x0_prediction(x_0_probs, x_t, t, self.scheduler)
                    x_tm1_pred = torch.bernoulli(x_tm1_probs)
                
                else:
                    x_tm1_pred = (x_0_logits > 0) * 1.0
                    # x_tm1_probs = torch.sigmoid(x_0_logits)
                    # x_tm1_pred = torch.bernoulli(x_tm1_probs)

                # if (t[0].item()-1) % 10 == 0 or t[0].item() < 5:

                #     import matplotlib.pyplot as plt
                #     plt.figure()
                #     plt.imshow(x_0_probs[0][0].detach().cpu().numpy())
                #     plt.colorbar()
                #     plt.title(f't={t[0].item()}')

                x_t = x_tm1_pred

                if mask is not None:
                    m = mask['mask'].unsqueeze(0)
                    latent = mask['latent'].unsqueeze(0)
                    x_t = latent * m + x_t * (1-m)


                if return_all:
                    x_all.append(x_t)
            if return_all:
                return torch.stack(x_all[::-1], 1)
            else:
                return x_t
        
    def forward(self, x, label=None, x_t=None):
        return self._train_loss(x, label, x_t)
    
    def get_forwarding_x_t(self, x, return_steps):
        return_steps = [0] + list(return_steps)
        last_step = step = 0
        if x.ndim == 3:
            x = x[0]
        else:
            x = x
        forwarding_x_t = x
        forwarding_x_t_list = []
        for target_step in return_steps:
            for step in range(last_step, target_step):
                forwarding_x_t = torch.bernoulli(self.scheduler.one_step(forwarding_x_t, step))
            forwarding_x_t_list.append(forwarding_x_t)
            last_step = step
        forwarding_x_t_list = forwarding_x_t_list[1:]
        forwarding_x_t_list = torch.stack(forwarding_x_t_list, dim=0)
        forwarding_x_t_list = einops.rearrange(forwarding_x_t_list, 't h w -> (t w) h')
        return forwarding_x_t_list
    

def get_p_xtm1_from_x0_prediction(x_0_prediction: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, scheduler: 'NoiseScheduler'):

    beta = scheduler.get_beta(x_t, t)
    k = scheduler.get_k(x_t, t)
    k_tm1 = scheduler.get_k(x_t, t-1)
    b = scheduler.get_b(x_t, t)
    b_tm1 = scheduler.get_b(x_t, t-1)
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
        dim = x.ndim - 1
        beta = self.beta[t].view(-1, *([1]*dim))
        x = x * (1-beta) + self.gamma * beta
        return x

    def forward(self, x, t):
        dim = x.ndim - 1
        k = self.k_final[t].view(-1, *([1]*dim))
        b = self.b_final[t].view(-1, *([1]*dim))
        out = k * x + b
        return out
    
    def get_beta(self, shape_tensor: Tensor, t):
        dim = shape_tensor.ndim - 1
        return self.beta[t].view(-1, *([1]*dim))
    
    def get_b(self, shape_tensor: Tensor, t):
        dim = shape_tensor.ndim - 1
        return self.b_final[t].view(-1, *([1]*dim))
    
    def get_k(self, shape_tensor: Tensor, t):
        dim = shape_tensor.ndim - 1
        return self.k_final[t].view(-1, *([1]*dim))
    

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
