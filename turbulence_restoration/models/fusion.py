
from __future__ import annotations

import torch
import torch.nn as nn

from turbulence_restoration.models.blocks import ConvBlock


class TemporalFusionLevel(nn.Module):
    """Confidence-aware temporal fusion with softmax over frames."""

    def __init__(self, channels: int, use_time_bias: bool = True):
        super().__init__()
        self.use_time_bias = use_time_bias
        self.score_net = nn.Sequential(
            ConvBlock(channels * 3, channels),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        if use_time_bias:
            self.time_score = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.time_score[-1].weight)
            nn.init.zeros_(self.time_score[-1].bias)

    def forward(self, aligned: torch.Tensor, dt: torch.Tensor, valid_mask: torch.Tensor):
        B, K, C, H, W = aligned.shape
        center_idx = K // 2
        center = aligned[:, center_idx]
        center_rep = center[:, None].expand(-1, K, -1, -1, -1)

        x = torch.cat([aligned, center_rep, torch.abs(aligned - center_rep)], dim=2)
        score = self.score_net(x.reshape(B * K, 3 * C, H, W)).reshape(B, K, 1, H, W)

        if self.use_time_bias:
            score = score + self.time_score(dt.reshape(B * K, 1)).reshape(B, K, 1, 1, 1)

        mask = valid_mask.reshape(B, K, 1, 1, 1)
        score = score.masked_fill(mask < 0.5, -1e9)
        weights = torch.softmax(score, dim=1)
        fused = torch.sum(weights * aligned, dim=1)
        return fused, weights


class TemporalFusionPyramid(nn.Module):
    def __init__(self, c0: int = 32, c1: int = 64, c2: int = 128, use_time_bias: bool = True):
        super().__init__()
        self.fuse0 = TemporalFusionLevel(c0, use_time_bias)
        self.fuse1 = TemporalFusionLevel(c1, use_time_bias)
        self.fuse2 = TemporalFusionLevel(c2, use_time_bias)

    def forward(self, a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, dt: torch.Tensor, valid_mask: torch.Tensor):
        f0, w0 = self.fuse0(a0, dt, valid_mask)
        f1, w1 = self.fuse1(a1, dt, valid_mask)
        f2, w2 = self.fuse2(a2, dt, valid_mask)
        return (f0, f1, f2), {"weights0": w0, "weights1": w1, "weights2": w2}
