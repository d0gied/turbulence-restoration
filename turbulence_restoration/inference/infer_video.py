
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from turbulence_restoration.inference.temporal_policy import InferenceTemporalPolicy
from turbulence_restoration.inference.tiled_inference import (
    TileRecurrentState,
    infer_frame_tiled,
    infer_frame_tiled_recurrent,
)
from turbulence_restoration.models import RecurrentGeoLuckyRestorer, TimeAwareGeoLuckyRestorer
from turbulence_restoration.utils.io import read_video_tensor, write_video_tensor


def _video_frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = torch.from_numpy(frame_rgb).float().div_(255.0)
    return arr.permute(2, 0, 1).contiguous()


def _tensor_to_video_frame(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    arr = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def build_model_from_checkpoint(checkpoint, device: torch.device):
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    c0 = int(model_cfg.get("c0", 32))
    c1 = int(model_cfg.get("c1", 64))
    c2 = int(model_cfg.get("c2", 128))
    use_time = bool(model_cfg.get("use_time", True))
    model_type = str(model_cfg.get("type", ""))
    has_recurrent_keys = any(key.startswith("gru2.") or key.startswith("merge2.") for key in state)

    base_model = TimeAwareGeoLuckyRestorer(c0=c0, c1=c1, c2=c2, use_time=use_time)
    if model_type in {"recurrent", "recurrent_geo_lucky"} or has_recurrent_keys:
        model = RecurrentGeoLuckyRestorer(
            base_model,
            bottleneck_channels=int(model_cfg.get("bottleneck_channels", c2)),
            hidden_damping=float(model_cfg.get("hidden_damping", 1.0)),
        )
    else:
        model = base_model

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _stream_indices_without_end_reflection(center_idx: int, offsets_idx: np.ndarray) -> np.ndarray:
    raw = center_idx + offsets_idx
    indices = raw.copy()
    negative = indices < 0
    indices[negative] = -indices[negative]
    return indices.astype(np.int64, copy=False)


def _drop_stale_frames(frame_buffer: Dict[int, torch.Tensor], next_center_idx: int, offsets_idx: np.ndarray) -> int:
    radius = int(np.max(np.abs(offsets_idx))) if offsets_idx.size else 0
    keep_from = max(0, next_center_idx - radius)
    for frame_idx in list(frame_buffer):
        if frame_idx < keep_from:
            del frame_buffer[frame_idx]
    return keep_from


@torch.inference_mode()
def infer_video(model, video: torch.Tensor, fps: float, temporal_policy: str = "default_200ms",
                tile: int = 512, halo: int = 128, overlap: int = 128, amp: bool = True) -> torch.Tensor:
    policy = InferenceTemporalPolicy(temporal_policy)
    outputs = []
    N = video.shape[0]
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    dt = torch.as_tensor(policy.offsets_sec, device=device, dtype=dtype)
    valid = torch.ones_like(dt)
    recurrent_state = TileRecurrentState() if getattr(model, "supports_sequence", False) else None
    for t in tqdm(range(N), desc="infer"):
        indices = policy.get_frame_indices(t, fps, N)
        frames = video[indices]
        if recurrent_state is None:
            pred = infer_frame_tiled(
                model,
                frames,
                dt,
                valid,
                tile=tile,
                halo=halo,
                overlap=overlap,
                amp=amp,
            )
        else:
            pred = infer_frame_tiled_recurrent(
                model,
                frames,
                dt,
                valid,
                recurrent_state,
                tile=tile,
                halo=halo,
                overlap=overlap,
                amp=amp,
            )
        outputs.append(pred.cpu())
    return torch.stack(outputs, dim=0)


@torch.inference_mode()
def infer_video_streaming(
    model,
    input_path: str,
    output_path: str,
    fps: float,
    temporal_policy: str = "default_200ms",
    tile: int = 512,
    halo: int = 128,
    overlap: int = 128,
    amp: bool = True,
) -> int:
    policy = InferenceTemporalPolicy(temporal_policy)
    offsets_idx = policy.get_offsets_idx(fps)
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    dt = torch.as_tensor(policy.offsets_sec, device=device, dtype=dtype)
    valid = torch.ones_like(dt)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total_frames if total_frames > 0 else None
    frame_buffer: Dict[int, torch.Tensor] = {}
    recurrent_state = TileRecurrentState() if getattr(model, "supports_sequence", False) else None
    writer: cv2.VideoWriter | None = None
    frames_read = 0
    next_output = 0
    written = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=total, desc="infer", unit="frame")

    def write_ready(final_num_frames: int | None = None) -> None:
        nonlocal next_output, writer, written

        while True:
            if final_num_frames is None:
                indices = _stream_indices_without_end_reflection(next_output, offsets_idx)
            else:
                if next_output >= final_num_frames:
                    break
                indices = policy.get_frame_indices(next_output, fps, final_num_frames)

            if any(int(idx) not in frame_buffer for idx in indices):
                break

            if recurrent_state is not None:
                frames = torch.stack([frame_buffer[int(idx)] for idx in indices], dim=0)
                pred = infer_frame_tiled_recurrent(
                    model,
                    frames,
                    dt,
                    valid,
                    recurrent_state,
                    tile=tile,
                    halo=halo,
                    overlap=overlap,
                    amp=amp,
                )
            else:
                frames = torch.stack([frame_buffer[int(idx)] for idx in indices], dim=0)
                pred = infer_frame_tiled(
                    model,
                    frames,
                    dt,
                    valid,
                    tile=tile,
                    halo=halo,
                    overlap=overlap,
                    amp=amp,
                )

            if writer is None:
                _, height, width = pred.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video writer: {output_path}")

            writer.write(_tensor_to_video_frame(pred))
            next_output += 1
            written += 1
            progress.update(1)
            _drop_stale_frames(frame_buffer, next_output, offsets_idx)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_buffer[frames_read] = _video_frame_to_tensor(frame_bgr)
            frames_read += 1
            write_ready()

        if frames_read == 0:
            raise RuntimeError(f"No frames read from {input_path}")

        write_ready(final_num_frames=frames_read)
        if written != frames_read:
            raise RuntimeError(f"Streaming inference wrote {written} frames, expected {frames_read}")
    finally:
        progress.close()
        cap.release()
        if writer is not None:
            writer.release()

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--temporal_policy", default="default_200ms")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--halo", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--in_memory", action="store_true", help="Load the full video tensor before inference.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location="cpu")
    model = build_model_from_checkpoint(ckpt, device)

    if args.in_memory:
        video = read_video_tensor(args.input)
        output = infer_video(
            model,
            video,
            args.fps,
            args.temporal_policy,
            args.tile,
            args.halo,
            args.overlap,
            not args.no_amp,
        )
        write_video_tensor(output, args.output, fps=args.fps)
    else:
        infer_video_streaming(
            model,
            args.input,
            args.output,
            args.fps,
            args.temporal_policy,
            args.tile,
            args.halo,
            args.overlap,
            not args.no_amp,
        )


if __name__ == "__main__":
    main()
