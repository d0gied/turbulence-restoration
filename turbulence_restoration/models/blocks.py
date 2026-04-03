
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel, stride=1, padding=pad),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TimeFiLM(nn.Module):
    """FiLM modulation from relative timestamp dt in seconds."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        feat: [B,K,C,H,W]
        dt:   [B,K]
        """
        B, K, C, H, W = feat.shape
        emb = self.mlp(dt.reshape(B * K, 1))
        gamma, beta = emb.chunk(2, dim=1)
        gamma = gamma.reshape(B, K, C, 1, 1)
        beta = beta.reshape(B, K, C, 1, 1)
        return feat * (1.0 + gamma) + beta
