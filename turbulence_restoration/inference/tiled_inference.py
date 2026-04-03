
from __future__ import annotations

from typing import Iterator, Tuple

import torch


def make_blend_mask(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    wy = torch.hann_window(h + 2, device=device, dtype=dtype)[1:-1]
    wx = torch.hann_window(w + 2, device=device, dtype=dtype)[1:-1]
    mask = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
    return mask[None]


def generate_tiles(H: int, W: int, tile: int, overlap: int) -> Iterator[Tuple[int, int, int, int]]:
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")

    if H <= tile:
        ys = [0]
    else:
        stride = tile - overlap
        ys = list(range(0, H - tile + 1, stride))
        if ys[-1] != H - tile:
            ys.append(H - tile)

    if W <= tile:
        xs = [0]
    else:
        stride = tile - overlap
        xs = list(range(0, W - tile + 1, stride))
        if xs[-1] != W - tile:
            xs.append(W - tile)

    for y in ys:
        for x in xs:
            yield y, x, min(tile, H - y), min(tile, W - x)


@torch.no_grad()
def infer_frame_tiled(
    model,
    frames: torch.Tensor,
    dt: torch.Tensor,
    valid: torch.Tensor,
    tile: int = 512,
    halo: int = 128,
    overlap: int = 128,
    amp: bool = True,
) -> torch.Tensor:
    """
    frames: [K,3,H,W], float in [0,1]
    dt:     [K]
    valid:  [K]
    returns [3,H,W]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    frames = frames.to(device=device, dtype=dtype)
    dt = dt.to(device=device, dtype=dtype)
    valid = valid.to(device=device, dtype=dtype)

    K, C, H, W = frames.shape
    output_sum = torch.zeros(3, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(1, H, W, device=device, dtype=dtype)

    for y, x, h, w in generate_tiles(H, W, tile, overlap):
        y0 = max(0, y - halo)
        x0 = max(0, x - halo)
        y1 = min(H, y + h + halo)
        x1 = min(W, x + w + halo)

        crop = frames[:, :, y0:y1, x0:x1].unsqueeze(0)
        dt_b = dt.unsqueeze(0)
        valid_b = valid.unsqueeze(0)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                pred_crop = model(crop, dt_b, valid_b)
        else:
            pred_crop = model(crop, dt_b, valid_b)

        pred_crop = pred_crop[0]
        inner_y0 = y - y0
        inner_x0 = x - x0
        inner_y1 = inner_y0 + h
        inner_x1 = inner_x0 + w
        pred_inner = pred_crop[:, inner_y0:inner_y1, inner_x0:inner_x1]

        mask = make_blend_mask(h, w, device, dtype)
        output_sum[:, y:y+h, x:x+w] += pred_inner * mask
        weight_sum[:, y:y+h, x:x+w] += mask

    return (output_sum / weight_sum.clamp_min(1e-6)).clamp(0.0, 1.0)
