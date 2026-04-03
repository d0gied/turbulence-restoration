
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbulence_restoration.models.warp import warp_features


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.reshape(1, 1, 3, 3))
        self.register_buffer("ky", ky.reshape(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor):
        _, C, _, _ = x.shape
        kx = self.kx.to(x).repeat(C, 1, 1, 1)
        ky = self.ky.to(x).repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        px, py = self._grad(pred)
        tx, ty = self._grad(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


class RestorationLoss(nn.Module):
    def __init__(self, edge_weight: float = 0.05):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = SobelEdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pix = self.charbonnier(pred, target)
        edge = self.edge(pred, target)
        loss = pix + self.edge_weight * edge
        return loss, {"pix": pix.detach(), "edge": edge.detach()}


def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return dx + dy


def alignment_loss(frames: torch.Tensor, target: torch.Tensor, flow0: torch.Tensor) -> torch.Tensor:
    B, K, C, H, W = frames.shape
    center_idx = K // 2
    losses = []
    for i in range(K):
        if i == center_idx:
            continue
        aligned = warp_features(frames[:, i], flow0[:, i])
        losses.append(F.l1_loss(aligned, target))
    if not losses:
        return torch.zeros((), device=frames.device, dtype=frames.dtype)
    return torch.stack(losses).mean()


def oracle_weight_loss(frames: torch.Tensor, target: torch.Tensor, flow0: torch.Tensor,
                       pred_weights: torch.Tensor, temperature: float = 0.03) -> torch.Tensor:
    """
    Teaches fusion to prefer locally best aligned frames.
    """
    B, K, C, H, W = frames.shape
    aligned_rgb = []
    for i in range(K):
        aligned_rgb.append(warp_features(frames[:, i], flow0[:, i]))
    aligned_rgb = torch.stack(aligned_rgb, dim=1)

    err = torch.mean(torch.abs(aligned_rgb - target[:, None]), dim=2, keepdim=True)
    oracle = torch.softmax(-err / temperature, dim=1)
    eps = 1e-8
    return torch.mean(oracle * torch.log((oracle + eps) / (pred_weights + eps)))
