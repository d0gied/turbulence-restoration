
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(-3, -2, -1)).clamp_min(1e-12)
    max_tensor = torch.tensor(max_val, device=pred.device, dtype=pred.dtype)
    return 20.0 * torch.log10(max_tensor) - 10.0 * torch.log10(mse)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    window = window / window.sum()
    return window.reshape(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, C, H, W = pred.shape
    window = _gaussian_window(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu12

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean(dim=(-3, -2, -1))
