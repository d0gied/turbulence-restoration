
from __future__ import annotations

import torch
import torch.nn as nn

from turbulence_restoration.models.blocks import ConvBlock, ResBlock, TimeFiLM


class SharedEncoder2D(nn.Module):
    def __init__(self, c0: int = 32, c1: int = 64, c2: int = 128, use_time_film: bool = True):
        super().__init__()
        self.use_time_film = use_time_film
        self.level0 = nn.Sequential(ConvBlock(3, c0), ResBlock(c0), ResBlock(c0))
        self.down1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.level1 = nn.Sequential(nn.GELU(), ResBlock(c1), ResBlock(c1))
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.level2 = nn.Sequential(nn.GELU(), ResBlock(c2), ResBlock(c2), ResBlock(c2))

        if use_time_film:
            self.time0 = TimeFiLM(c0)
            self.time1 = TimeFiLM(c1)
            self.time2 = TimeFiLM(c2)

    def forward(self, frames: torch.Tensor, dt: torch.Tensor):
        B, K, C, H, W = frames.shape
        x = frames.reshape(B * K, C, H, W)
        f0 = self.level0(x)
        f1 = self.level1(self.down1(f0))
        f2 = self.level2(self.down2(f1))

        _, C0, H0, W0 = f0.shape
        _, C1, H1, W1 = f1.shape
        _, C2, H2, W2 = f2.shape
        f0 = f0.reshape(B, K, C0, H0, W0)
        f1 = f1.reshape(B, K, C1, H1, W1)
        f2 = f2.reshape(B, K, C2, H2, W2)

        if self.use_time_film:
            f0 = self.time0(f0, dt)
            f1 = self.time1(f1, dt)
            f2 = self.time2(f2, dt)
        return f0, f1, f2
