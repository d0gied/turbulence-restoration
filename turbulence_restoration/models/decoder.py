
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbulence_restoration.models.blocks import ConvBlock, ResBlock


class RestorationDecoder(nn.Module):
    def __init__(self, c0: int = 32, c1: int = 64, c2: int = 128):
        super().__init__()
        self.bottleneck = nn.Sequential(ResBlock(c2), ResBlock(c2), ResBlock(c2), ResBlock(c2))
        self.up1 = nn.Sequential(ConvBlock(c2 + c1, c1), ResBlock(c1), ResBlock(c1))
        self.up0 = nn.Sequential(ConvBlock(c1 + c0, c0), ResBlock(c0), ResBlock(c0))
        self.out = nn.Sequential(nn.Conv2d(c0, c0, 3, padding=1), nn.GELU(), nn.Conv2d(c0, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        x2 = self.bottleneck(f2)
        x1 = F.interpolate(x2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.up1(torch.cat([x1, f1], dim=1))
        x0 = F.interpolate(x1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        x0 = self.up0(torch.cat([x0, f0], dim=1))
        return self.out(x0)
