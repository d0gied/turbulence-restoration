
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


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt(x * x + eps * eps))


def lowpass_seq(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """
    x: [B,S,C,H,W]. Spatial low-pass keeps temporal losses focused on visible jitter.
    """
    if x.dim() != 5:
        raise ValueError(f"Expected sequence tensor [B,S,C,H,W], got {tuple(x.shape)}")

    B, S, C, H, W = x.shape
    y = x.reshape(B * S, C, H, W)
    if scale > 1 and H >= scale and W >= scale:
        y = F.avg_pool2d(y, kernel_size=int(scale), stride=int(scale))

    _, _, h, w = y.shape
    return y.reshape(B, S, C, h, w)


def temporal_velocity_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    lowpass_scale: int = 2,
) -> torch.Tensor:
    if pred_seq.shape[1] < 2:
        return pred_seq.new_zeros(())

    pred_lp = lowpass_seq(pred_seq, lowpass_scale)
    target_lp = lowpass_seq(target_seq, lowpass_scale)
    pred_delta = pred_lp[:, 1:] - pred_lp[:, :-1]
    target_delta = target_lp[:, 1:] - target_lp[:, :-1]
    return charbonnier(pred_delta - target_delta)


def temporal_acceleration_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    lowpass_scale: int = 2,
) -> torch.Tensor:
    if pred_seq.shape[1] < 3:
        return pred_seq.new_zeros(())

    pred_lp = lowpass_seq(pred_seq, lowpass_scale)
    target_lp = lowpass_seq(target_seq, lowpass_scale)
    pred_acc = pred_lp[:, 2:] - 2.0 * pred_lp[:, 1:-1] + pred_lp[:, :-2]
    target_acc = target_lp[:, 2:] - 2.0 * target_lp[:, 1:-1] + target_lp[:, :-2]
    return charbonnier(pred_acc - target_acc)


def residual_acceleration_loss(
    pred_seq: torch.Tensor,
    input_center_seq: torch.Tensor,
    lowpass_scale: int = 2,
) -> torch.Tensor:
    if pred_seq.shape[1] < 3:
        return pred_seq.new_zeros(())

    residual = pred_seq - input_center_seq
    residual_lp = lowpass_seq(residual, lowpass_scale)
    residual_acc = residual_lp[:, 2:] - 2.0 * residual_lp[:, 1:-1] + residual_lp[:, :-2]
    return charbonnier(residual_acc)


def fusion_weight_smoothness_loss(weights_seq: torch.Tensor) -> torch.Tensor:
    if weights_seq.shape[1] < 2:
        return weights_seq.new_zeros(())
    return charbonnier(weights_seq[:, 1:] - weights_seq[:, :-1])


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
