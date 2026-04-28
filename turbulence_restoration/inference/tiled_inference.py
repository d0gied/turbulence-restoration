
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

def _pad_spatial_to_multiple(x: torch.Tensor, multiple: int = 4) -> Tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), h, w


def make_blend_mask(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    wy = torch.hann_window(h + 2, device=device, dtype=dtype)[1:-1]
    wx = torch.hann_window(w + 2, device=device, dtype=dtype)[1:-1]
    mask = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
    return mask[None]


class TileRecurrentState:
    def __init__(self):
        self._h2: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def get(self, bounds: Tuple[int, int, int, int]) -> torch.Tensor | None:
        return self._h2.get(bounds)

    def set(self, bounds: Tuple[int, int, int, int], h2: torch.Tensor) -> None:
        self._h2[bounds] = h2.detach()

    def clear(self) -> None:
        self._h2.clear()

    def __len__(self) -> int:
        return len(self._h2)


def generate_tiles(H: int, W: int, tile: int, overlap: int):
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


@torch.inference_mode()
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
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    frames = frames.to(device=device, dtype=dtype, non_blocking=True)
    dt = dt.to(device=device, dtype=dtype)
    valid = valid.to(device=device, dtype=dtype)
    dt_b = dt.unsqueeze(0)
    valid_b = valid.unsqueeze(0)

    _, C, H, W = frames.shape

    def autocast_context():
        return torch.amp.autocast("cuda") if amp and device.type == "cuda" else nullcontext()

    if H <= tile and W <= tile:
        padded_frames, crop_h, crop_w = _pad_spatial_to_multiple(frames)
        with autocast_context():
            pred = model(padded_frames.unsqueeze(0), dt_b, valid_b)[0]
        return pred[:, :crop_h, :crop_w].clamp(0.0, 1.0)

    output_sum = torch.zeros(C, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(1, H, W, device=device, dtype=dtype)
    mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    for y, x, h, w in generate_tiles(H, W, tile, overlap):
        y0 = max(0, y - halo)
        x0 = max(0, x - halo)
        y1 = min(H, y + h + halo)
        x1 = min(W, x + w + halo)

        crop, crop_h, crop_w = _pad_spatial_to_multiple(frames[:, :, y0:y1, x0:x1])

        with autocast_context():
            pred_crop = model(crop.unsqueeze(0), dt_b, valid_b)

        pred_crop = pred_crop[0, :, :crop_h, :crop_w]
        inner_y0 = y - y0
        inner_x0 = x - x0
        inner_y1 = inner_y0 + h
        inner_x1 = inner_x0 + w
        pred_inner = pred_crop[:, inner_y0:inner_y1, inner_x0:inner_x1]

        mask = mask_cache.get((h, w))
        if mask is None:
            mask = make_blend_mask(h, w, device, dtype)
            mask_cache[(h, w)] = mask
        output_sum[:, y:y+h, x:x+w] += pred_inner * mask
        weight_sum[:, y:y+h, x:x+w] += mask

    return (output_sum / weight_sum.clamp_min(1e-6)).clamp(0.0, 1.0)


@torch.inference_mode()
def infer_frame_tiled_recurrent(
    model,
    frames: torch.Tensor,
    dt: torch.Tensor,
    valid: torch.Tensor,
    recurrent_state: TileRecurrentState,
    tile: int = 512,
    halo: int = 128,
    overlap: int = 128,
    amp: bool = True,
) -> torch.Tensor:
    """
    Recurrent tiled inference for one output frame.

    Hidden state is kept per tile bounds, so consecutive output frames reuse the
    same spatial state without mixing unrelated tiles.
    """
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    frames = frames.to(device=device, dtype=dtype, non_blocking=True)
    dt = dt.to(device=device, dtype=dtype)
    valid = valid.to(device=device, dtype=dtype)
    dt_b = dt.unsqueeze(0)
    valid_b = valid.unsqueeze(0)

    _, C, H, W = frames.shape

    def autocast_context():
        return torch.amp.autocast("cuda") if amp and device.type == "cuda" else nullcontext()

    if H <= tile and W <= tile:
        padded_frames, crop_h, crop_w = _pad_spatial_to_multiple(frames)
        bounds = (0, 0, H, W)
        with autocast_context():
            pred, h2 = model.forward_step(
                padded_frames.unsqueeze(0),
                dt_b,
                valid_b,
                h2=recurrent_state.get(bounds),
                return_aux=False,
            )
        recurrent_state.set(bounds, h2)
        return pred[0, :, :crop_h, :crop_w].clamp(0.0, 1.0)

    output_sum = torch.zeros(C, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(1, H, W, device=device, dtype=dtype)
    mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    for y, x, h, w in generate_tiles(H, W, tile, overlap):
        y0 = max(0, y - halo)
        x0 = max(0, x - halo)
        y1 = min(H, y + h + halo)
        x1 = min(W, x + w + halo)
        bounds = (y0, x0, y1, x1)

        crop, crop_h, crop_w = _pad_spatial_to_multiple(frames[:, :, y0:y1, x0:x1])

        with autocast_context():
            pred_crop, h2 = model.forward_step(
                crop.unsqueeze(0),
                dt_b,
                valid_b,
                h2=recurrent_state.get(bounds),
                return_aux=False,
            )
        recurrent_state.set(bounds, h2)

        pred_crop = pred_crop[0, :, :crop_h, :crop_w]
        inner_y0 = y - y0
        inner_x0 = x - x0
        inner_y1 = inner_y0 + h
        inner_x1 = inner_x0 + w
        pred_inner = pred_crop[:, inner_y0:inner_y1, inner_x0:inner_x1]

        mask = mask_cache.get((h, w))
        if mask is None:
            mask = make_blend_mask(h, w, device, dtype)
            mask_cache[(h, w)] = mask
        output_sum[:, y:y+h, x:x+w] += pred_inner * mask
        weight_sum[:, y:y+h, x:x+w] += mask

    return (output_sum / weight_sum.clamp_min(1e-6)).clamp(0.0, 1.0)
