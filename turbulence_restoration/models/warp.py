
from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_features(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """
    x:    [B,C,H,W]
    flow: [B,2,H,W], flow[:,0]=dx, flow[:,1]=dy in pixels.
    Output samples x at output coordinates + flow.
    """
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx[None].expand(B, H, W)
    yy = yy[None].expand(B, H, W)
    grid_x = xx + flow[:, 0]
    grid_y = yy + flow[:, 1]
    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def upsample_flow(flow: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    flow = F.interpolate(flow, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    return flow * float(scale_factor)
