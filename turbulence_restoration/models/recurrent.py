from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)

        self.conv_zr = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            2 * self.hidden_channels,
            kernel_size,
            padding=padding,
        )
        self.conv_h = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            self.hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"ConvGRUCell expects [B,C,H,W], got {tuple(x.shape)}")

        B, _, H, W = x.shape
        if h_prev is None:
            h_prev = x.new_zeros(B, self.hidden_channels, H, W)
        elif h_prev.shape != (B, self.hidden_channels, H, W):
            raise ValueError(
                "h_prev shape mismatch: "
                f"expected {(B, self.hidden_channels, H, W)}, got {tuple(h_prev.shape)}"
            )

        combined = torch.cat([x, h_prev], dim=1)
        zr = torch.sigmoid(self.conv_zr(combined))
        z, r = torch.chunk(zr, chunks=2, dim=1)
        candidate = torch.tanh(self.conv_h(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * candidate


def _flatten_aux_list(aux_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if not aux_list:
        return out

    for key in aux_list[0]:
        values = [aux[key] for aux in aux_list if key in aux and torch.is_tensor(aux[key])]
        if len(values) != len(aux_list):
            continue
        stacked = torch.stack(values, dim=1)
        B, S = stacked.shape[:2]
        out[key] = stacked.reshape(B * S, *stacked.shape[2:])
    return out


class RecurrentGeoLuckyRestorer(nn.Module):
    """
    Recurrent wrapper over TimeAwareGeoLuckyRestorer.

    Single-step input:
      frames: [B,K,3,H,W], dt: [B,K], valid_mask: [B,K]

    Sequence input:
      frames_seq: [B,S,K,3,H,W], dt_seq: [B,S,K], valid_seq: [B,S,K]
    """

    supports_sequence = True

    def __init__(
        self,
        base_model: nn.Module,
        bottleneck_channels: int = 128,
        hidden_damping: float = 1.0,
    ):
        super().__init__()
        self.encoder = base_model.encoder
        self.aligner = base_model.aligner
        self.fusion = base_model.fusion
        self.decoder = base_model.decoder
        self.hidden_damping = float(hidden_damping)

        self.gru2 = ConvGRUCell(
            input_channels=bottleneck_channels,
            hidden_channels=bottleneck_channels,
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.merge2[-1].weight)
        nn.init.zeros_(self.merge2[-1].bias)

    def forward_step(
        self,
        frames: torch.Tensor,
        dt: torch.Tensor,
        valid_mask: torch.Tensor,
        h2: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if frames.dim() != 5:
            raise ValueError(f"forward_step expects frames [B,K,3,H,W], got {tuple(frames.shape)}")

        _, K, _, _, _ = frames.shape
        center = frames[:, K // 2]

        f0, f1, f2 = self.encoder(frames, dt)
        return self.forward_step_from_features(
            center,
            f0,
            f1,
            f2,
            dt,
            valid_mask,
            h2=h2,
            return_aux=return_aux,
        )

    def forward_step_from_features(
        self,
        center: torch.Tensor,
        f0: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        dt: torch.Tensor,
        valid_mask: torch.Tensor,
        h2: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        a0, a1, a2, flow0 = self.aligner(f0, f1, f2)
        (fused0, fused1, fused2), fusion_aux = self.fusion(a0, a1, a2, dt, valid_mask)

        h2 = self.gru2(fused2, h2)
        if self.hidden_damping != 1.0:
            h2 = self.hidden_damping * h2

        recurrent_delta = self.merge2(torch.cat([fused2, h2], dim=1))
        fused2_recurrent = fused2 + recurrent_delta
        residual = self.decoder(fused0, fused1, fused2_recurrent)
        pred = (center + residual).clamp(0.0, 1.0)

        if not return_aux:
            return pred, h2

        aux = {"flow0": flow0, "h2": h2, **fusion_aux}
        return pred, h2, aux

    def forward(
        self,
        frames: torch.Tensor,
        dt: torch.Tensor,
        valid_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        if frames.dim() == 5:
            if return_aux:
                pred, _, aux = self.forward_step(frames, dt, valid_mask, h2=None, return_aux=True)
                return pred, aux
            pred, _ = self.forward_step(frames, dt, valid_mask, h2=None, return_aux=False)
            return pred

        if frames.dim() != 6:
            raise ValueError(f"Expected frames [B,K,3,H,W] or [B,S,K,3,H,W], got {tuple(frames.shape)}")

        _, S, _, _, _, _ = frames.shape
        h2 = None
        preds = []
        aux_list = []

        for seq_idx in range(S):
            if return_aux:
                pred, h2, aux = self.forward_step(
                    frames[:, seq_idx],
                    dt[:, seq_idx],
                    valid_mask[:, seq_idx],
                    h2=h2,
                    return_aux=True,
                )
                aux_list.append(aux)
            else:
                pred, h2 = self.forward_step(
                    frames[:, seq_idx],
                    dt[:, seq_idx],
                    valid_mask[:, seq_idx],
                    h2=h2,
                    return_aux=False,
                )
            preds.append(pred)

        pred_seq = torch.stack(preds, dim=1)
        if return_aux:
            return pred_seq, _flatten_aux_list(aux_list)
        return pred_seq
