import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Generator, TypeVar, Generic
import torch

BackwardParams = TypeVar('BackwardParams', bound=dict)
class DiffusionModel(nn.Module, ABC, Generic[BackwardParams]):
    '''
    An abstract class for diffusion models.
    Defines the core interface that all diffusion models should implement.
    '''

    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.device = torch.device('cpu')

    def to(self, device: torch.device):
        self.device = device
        return super().to(device)
    
    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps for training. Uniformly sample from [0, num_steps) by default.
        Override this method to sample from a different distribution.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to put the sampled timesteps on
            
        Returns:
            Tensor of timesteps
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=self.device, dtype=torch.long)
    
    def get_batch_t(self, t: int, x: torch.Tensor) -> torch.Tensor:
        """
        Get the batch of timesteps with shape (x.shape[0], 1, 1, ..., 1) so x and the returned t can be broadcasted together.
        """
        return torch.full((x.shape[0],) + (1,) * (len(x.shape) - 1), t, device=self.device, dtype=torch.long)
    
    @abstractmethod
    def sample_x_T(self, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Sample from p(x_T)
        """
        pass
    
    @abstractmethod
    def forward_one_step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward one step of the diffusion process.
        """
        pass
    

    @abstractmethod
    def forward_from_x0(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x_0: Initial data
            t: Timesteps
            
        Returns:
            The noised data x_t
        """
        pass

    def forward_process(self, x_0: torch.Tensor) -> Generator[tuple[int, torch.Tensor], None, None]:
        """
        A generator that yields (t, x_t) with t from 0 to num_steps.
        """
        x_t = x_0
        for t in range(self.num_steps):
            yield t, x_t
            x_t = self.forward_from_x0(x_t, self.get_batch_t(t, x_t))

        yield self.num_steps, x_t


    @abstractmethod
    def backward_one_step(self, x_t: torch.Tensor, t: int, **kwargs: BackwardParams) -> tuple[torch.Tensor, dict]:
        """
        Sample from p(x_{t-1} | x_t).
        """
        pass

    def backward(self, x_t: torch.Tensor, t2: int, t1: int, **kwargs: BackwardParams) -> tuple[torch.Tensor, dict]:
        """
        Sample from  p(x_{t1} | x_{t2}).

        Optionally implement this method for a more efficient backward pass across multiple timesteps.
        """
        raise NotImplementedError("Backward is not implemented")

    def _backward(self, x_t: torch.Tensor, t2: int, t1: int, **kwargs: BackwardParams) -> tuple[torch.Tensor, dict]:
        """
        Sample from p(x_{t1} | x_{t2}). If backward is not implemented, this method falls back
        to multiple calls to backward_one_step.
        """
        try:
            return self.backward(x_t, t2, t1, **kwargs)
        except NotImplementedError:
            for t in range(t2, t1, -1):
                x_t, info = self.backward_one_step(x_t, t, **kwargs)
            return x_t, info
        
    def backward_process(self, x_t: torch.Tensor, **kwargs: BackwardParams) -> Generator[tuple[int, torch.Tensor, dict], None, None]:
        """
        A generator that yields (t, x_{t-1}, info) with t from num_steps to 0.
        """
        info = {}
        for t in range(self.num_steps, 0, -1):
            yield t, x_t, info
            x_t, info = self.backward_one_step(x_t, t, **kwargs)
        yield 0, x_t, info

    def sample(self, 
              shape: tuple[int, ...],
              steps_to_return: int|list[int] = 0,
              **kwargs: BackwardParams) -> torch.Tensor:
        """
        - Sample from p(x_0) if steps_to_return is 0 (default). Return shape is the same as the shape argument.
        - Sample from p(x_t) if steps_to_return is specified to t. Return shape is the same as the shape argument.
        - Return intermediate steps if steps_to_return is a list of timesteps. Return shape is (len(steps_to_return), *shape).
        
        Args:
            shape: Shape of samples to generate
            steps_to_return: Specify to return intermediate steps
            **kwargs: Additional sampling arguments
            
        Returns:
            Generated samples.
        """
        
        if isinstance(steps_to_return, int):
            steps_to_return = [steps_to_return]

        x_t = self.sample_x_T(shape)

        last_t = self.num_steps

        result = []
        for t in steps_to_return:
            x_t, info = self._backward(x_t, last_t, t, **kwargs)
            last_t = t
            result.append(x_t)

        if isinstance(steps_to_return, int):
            return result[0]
        else:
            return torch.stack(result, dim=0)



    @abstractmethod
    def forward(self, x0: torch.Tensor, **kwargs) -> dict:
        """
        Training forward pass. Return a dictionary containing loss and other metrics.
        
        Args:
            x0: Input data
            **kwargs: Additional forward pass arguments
            
        Returns:
            Dictionary with required 'loss' field and optional additional metrics
        """
        pass

class BinaryDiffusion(DiffusionModel):
    def __init__(self, num_steps: int):
        super().__init__(num_steps)

    