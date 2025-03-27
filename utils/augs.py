from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as F_t


def torch_rgb2hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def torch_hsv2rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


def augment_intensity(x: torch.Tensor, global_local_strength: Tuple[float]) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor, [N, 3, H, W], RGB image
        global_local_strength:
            tuple, (global_min, global_max, local_strength)
            image level global intensity change: U(global_min, global_max)
            pixel level local intensity change: N(1, local_strength)
    """
    global_min, global_max, local_strength = global_local_strength
    N, C, H, W = x.shape
    assert C == 3, f"Input shape {x.shape} is not RGB"
    x = torch_rgb2hsv(x)
    # global change per image
    if global_min < global_max:
        global_factor = torch.rand(N, device=x.device) * (global_max - global_min) + global_min  # U(global_min, global_max)
        x[:, 2] = torch.clamp(x[:, 2] * global_factor[:, None, None], 0, 1)
    # local change per pixel
    if local_strength > 0:
        local_factor = 1 + torch.randn(N, H, W, device=x.device) * local_strength  # N(1, local_strength)
        x[:, 2] = torch.clamp(x[:, 2] * local_factor, 0, 1)
    x = torch_hsv2rgb(x)
    return x


def augment_grid_distortion(images, min_max: Tuple[float], n_steps: Tuple[int] = (8, 17)) -> torch.Tensor:
    """
    Args:
        images: torch.Tensor, [N, C, H, W], input images
        min_max: tuple, (min, max) of strength of grid distortion
        n_steps: tuple, (min, max) of number of steps in grid distortion, default (8, 17)
    """
    min_strength, max_strength = min_max
    N, C, H, W = images.shape
    strengths = np.random.uniform(min_strength, max_strength, N).tolist()

    num_steps = np.random.randint(*n_steps)
    grid_steps = torch.linspace(-1, 1, num_steps)

    grids = []
    for _, strength in zip(range(N), strengths):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps)  # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1)  # perturb
        x_steps = (x_steps * W).long()  # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0)  # [W]

        y_steps = torch.linspace(0, 1, num_steps)  # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1)  # perturb
        y_steps = (y_steps * H).long()  # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0)  # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grids.append(grid)

    grids = torch.stack(grids, dim=0).to(images.device)  # [B, H, W, 2]
    images = F.grid_sample(images, grids, align_corners=False)
    return images


def augment_blur(images: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Args:
        images: torch.Tensor, [N, C, H, W], input images
        rate: float, rate of greatest downsampling
    """
    N, C, H, W = images.shape
    execute_blur_rate = rate + random.random() * (1 - rate)
    blur_size = int(H * execute_blur_rate), int(W * execute_blur_rate)
    images = F_t.resize(images, size=blur_size, interpolation=F_t.InterpolationMode.BICUBIC).clamp(0, 1)
    images = F_t.resize(images, size=(H, W), interpolation=F_t.InterpolationMode.BICUBIC).clamp(0, 1)
    return images
