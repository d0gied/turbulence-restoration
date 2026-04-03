
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


def read_video_tensor(path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames: List[torch.Tensor] = []
    ok, frame = cap.read()
    while ok:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = torch.from_numpy(frame).float() / 255.0
        frames.append(arr.permute(2, 0, 1).contiguous())
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return torch.stack(frames, dim=0)


def write_video_tensor(frames: torch.Tensor, path: str, fps: float = 60.0) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames = frames.detach().cpu().clamp(0, 1)
    N, C, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
    for i in range(N):
        arr = (frames[i].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        writer.write(arr)
    writer.release()
