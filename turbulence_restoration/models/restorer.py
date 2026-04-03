
from __future__ import annotations

import torch
import torch.nn as nn

from turbulence_restoration.models.aligner import PyramidAligner
from turbulence_restoration.models.decoder import RestorationDecoder
from turbulence_restoration.models.encoder import SharedEncoder2D
from turbulence_restoration.models.fusion import TemporalFusionPyramid


class TimeAwareGeoLuckyRestorer(nn.Module):
    """
    Input:
      frames:     [B,K,3,H,W]
      dt:         [B,K]
      valid_mask: [B,K]
    Output:
      restored central frame [B,3,H,W]
    """

    def __init__(self, c0: int = 32, c1: int = 64, c2: int = 128, use_time: bool = True):
        super().__init__()
        self.encoder = SharedEncoder2D(c0=c0, c1=c1, c2=c2, use_time_film=use_time)
        self.aligner = PyramidAligner(c0=c0, c1=c1, c2=c2)
        self.fusion = TemporalFusionPyramid(c0=c0, c1=c1, c2=c2, use_time_bias=use_time)
        self.decoder = RestorationDecoder(c0=c0, c1=c1, c2=c2)

    def forward(self, frames: torch.Tensor, dt: torch.Tensor, valid_mask: torch.Tensor, return_aux: bool = False):
        B, K, C, H, W = frames.shape
        center_idx = K // 2
        center = frames[:, center_idx]

        f0, f1, f2 = self.encoder(frames, dt)
        a0, a1, a2, flow0 = self.aligner(f0, f1, f2)
        (fused0, fused1, fused2), fusion_aux = self.fusion(a0, a1, a2, dt, valid_mask)
        residual = self.decoder(fused0, fused1, fused2)
        pred = (center + residual).clamp(0.0, 1.0)

        if not return_aux:
            return pred
        return pred, {"flow0": flow0, "aligned0": a0, **fusion_aux}
