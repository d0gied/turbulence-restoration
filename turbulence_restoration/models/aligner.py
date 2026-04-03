
from __future__ import annotations

import torch
import torch.nn as nn

from turbulence_restoration.models.blocks import ConvBlock
from turbulence_restoration.models.warp import upsample_flow, warp_features


class OffsetNet(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, max_delta: float = 8.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.body = nn.Sequential(ConvBlock(in_ch, hidden_ch), ConvBlock(hidden_ch, hidden_ch))
        self.head = nn.Conv2d(hidden_ch, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.body(x))) * self.max_delta


class PyramidAligner(nn.Module):
    """Coarse-to-fine feature alignment to the central frame."""

    def __init__(self, c0: int = 32, c1: int = 64, c2: int = 128):
        super().__init__()
        self.offset2 = OffsetNet(c2 * 3, c2, max_delta=8.0)
        self.offset1 = OffsetNet(c1 * 3 + 2, c1, max_delta=4.0)
        self.offset0 = OffsetNet(c0 * 3 + 2, c0, max_delta=2.0)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor):
        B, K, C0, H0, W0 = f0.shape
        center_idx = K // 2
        c0 = f0[:, center_idx]
        c1 = f1[:, center_idx]
        c2 = f2[:, center_idx]

        aligned0, aligned1, aligned2, flows0 = [], [], [], []
        for i in range(K):
            n0, n1, n2 = f0[:, i], f1[:, i], f2[:, i]
            if i == center_idx:
                zero0 = torch.zeros(B, 2, H0, W0, device=f0.device, dtype=f0.dtype)
                aligned0.append(n0)
                aligned1.append(n1)
                aligned2.append(n2)
                flows0.append(zero0)
                continue

            inp2 = torch.cat([n2, c2, torch.abs(n2 - c2)], dim=1)
            flow2 = self.offset2(inp2)
            a2 = warp_features(n2, flow2)

            flow1_up = upsample_flow(flow2, 2)
            n1_warp = warp_features(n1, flow1_up)
            inp1 = torch.cat([n1_warp, c1, torch.abs(n1_warp - c1), flow1_up], dim=1)
            flow1 = flow1_up + self.offset1(inp1)
            a1 = warp_features(n1, flow1)

            flow0_up = upsample_flow(flow1, 2)
            n0_warp = warp_features(n0, flow0_up)
            inp0 = torch.cat([n0_warp, c0, torch.abs(n0_warp - c0), flow0_up], dim=1)
            flow0 = flow0_up + self.offset0(inp0)
            a0 = warp_features(n0, flow0)

            aligned0.append(a0)
            aligned1.append(a1)
            aligned2.append(a2)
            flows0.append(flow0)

        return (
            torch.stack(aligned0, dim=1),
            torch.stack(aligned1, dim=1),
            torch.stack(aligned2, dim=1),
            torch.stack(flows0, dim=1),
        )
