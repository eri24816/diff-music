import torch
import matplotlib.pyplot as plt
import numpy as np
import einops
from pathlib import Path
def save_img(img: torch.Tensor | np.ndarray, path: Path):
    '''
    image shape: (c, h, w) or (h, w)
    '''
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)
    img = img.detach().cpu()

    if img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3:
        img = einops.rearrange(img, 'c h w -> h w c')
    plt.imsave(path, img)