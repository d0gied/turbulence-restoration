from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch


def rgb_tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def scalar_to_colormap(
    value: torch.Tensor,
    vmin: float,
    vmax: float,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    array = value.detach().cpu().float().numpy()
    denom = max(vmax - vmin, 1e-12)
    normalized = np.clip((array - vmin) / denom, 0.0, 1.0)
    gray = np.round(normalized * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def write_rgb_png(path: Path, image_rgb: np.ndarray, png_compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr, params):
        raise RuntimeError(f"Could not write image: {path}")


def tile_images(images: Sequence[np.ndarray], cols: int = 0, pad: int = 4, fill: int = 255) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile")
    if cols <= 0:
        cols = len(images)

    rows = int(math.ceil(len(images) / cols))
    height, width, channels = images[0].shape
    canvas_h = rows * height + max(0, rows - 1) * pad
    canvas_w = cols * width + max(0, cols - 1) * pad
    canvas = np.full((canvas_h, canvas_w, channels), fill, dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        top = row * (height + pad)
        left = col * (width + pad)
        canvas[top : top + height, left : left + width] = image
    return canvas


def make_rgb_grid(frames: torch.Tensor, cols: int = 0, pad: int = 4) -> np.ndarray:
    if frames.dim() != 4:
        raise ValueError(f"Expected frames [K,C,H,W], got {tuple(frames.shape)}")
    return tile_images([rgb_tensor_to_uint8(frame) for frame in frames], cols=cols, pad=pad)


def make_captioned_tile(image_rgb: np.ndarray, caption: str, header_h: int = 26) -> np.ndarray:
    height, width, _ = image_rgb.shape
    canvas = np.full((height + header_h, width, 3), 255, dtype=np.uint8)
    canvas[header_h:] = image_rgb
    cv2.putText(
        canvas,
        caption,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_summary_row(
    frames: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_metrics: Dict[str, float],
) -> np.ndarray:
    center = frames.shape[0] // 2
    input_img = rgb_tensor_to_uint8(frames[center])
    pred_img = rgb_tensor_to_uint8(pred)
    target_img = rgb_tensor_to_uint8(target)
    error_map = scalar_to_colormap((pred - target).abs().mean(dim=0), vmin=0.0, vmax=1.0)

    tiles = [
        make_captioned_tile(input_img, "input(center)"),
        make_captioned_tile(pred_img, f"pred | PSNR {sample_metrics['psnr']:.2f}"),
        make_captioned_tile(target_img, f"target | SSIM {sample_metrics['ssim']:.4f}"),
        make_captioned_tile(error_map, "abs error"),
    ]
    return tile_images(tiles, cols=len(tiles), pad=6)


class ValidationVisualizer:
    def __init__(self, num_samples: int = 8, grid_cols: int = 0):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if grid_cols < 0:
            raise ValueError("grid_cols must be non-negative")

        self.num_samples = int(num_samples)
        self.grid_cols = int(grid_cols)
        self.sequence_images: list[np.ndarray] = []
        self.summary_rows: list[np.ndarray] = []
        self.samples: list[Dict[str, Any]] = []
        self.psnr_values: list[float] = []
        self.ssim_values: list[float] = []

    def is_full(self) -> bool:
        return len(self.samples) >= self.num_samples

    def add_batch(
        self,
        batch: Dict[str, Any],
        pred: torch.Tensor,
        batch_psnr: torch.Tensor,
        batch_ssim: torch.Tensor,
    ) -> None:
        for i in range(pred.shape[0]):
            if self.is_full():
                break

            sample_index = len(self.samples)
            sample_frames = batch["frames"][i].detach().cpu()
            sample_pred = pred[i].detach().cpu()
            sample_target = batch["target"][i].detach().cpu()
            sample_metrics = {
                "psnr": float(batch_psnr[i].item()),
                "ssim": float(batch_ssim[i].item()),
            }

            self.sequence_images.append(make_rgb_grid(sample_frames, cols=self.grid_cols))
            self.summary_rows.append(make_summary_row(sample_frames, sample_pred, sample_target, sample_metrics))
            self.samples.append(
                {
                    "index": int(sample_index),
                    "psnr": sample_metrics["psnr"],
                    "ssim": sample_metrics["ssim"],
                    "dt": batch["dt"][i].detach().cpu().tolist(),
                    "valid": batch["valid"][i].detach().cpu().tolist(),
                }
            )
            self.psnr_values.append(sample_metrics["psnr"])
            self.ssim_values.append(sample_metrics["ssim"])

    def summary_image(self) -> np.ndarray:
        if not self.summary_rows:
            raise RuntimeError("No validation samples were collected")
        return tile_images(self.summary_rows, cols=1, pad=10)

    def metrics_payload(self, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not self.samples:
            raise RuntimeError("No validation samples were collected")

        payload: Dict[str, Any] = dict(extra or {})
        payload["samples"] = list(self.samples)
        payload["mean"] = {
            "psnr": float(sum(self.psnr_values) / len(self.psnr_values)),
            "ssim": float(sum(self.ssim_values) / len(self.ssim_values)),
        }
        return payload

    def write_artifacts(
        self,
        out_dir: Path,
        png_compression: int = 3,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not 0 <= png_compression <= 9:
            raise ValueError("png_compression must be between 0 and 9")

        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(self.sequence_images):
            write_rgb_png(out_dir / f"sample_{idx:02d}_sequence.png", image, png_compression)

        write_rgb_png(out_dir / "summary.png", self.summary_image(), png_compression)
        payload = self.metrics_payload(extra=extra)
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
