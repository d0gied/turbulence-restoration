
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from turbulence_restoration.data.temporal_sampler import TemporalSampler


class TurbulenceDataset(Dataset):
    """
    On-the-fly dataset:
      clean video sequence -> GPU simulator -> distorted window + clean center.
    clean_videos: list/sequence of tensors [N,3,H,W] in [0,1].
    """

    def __init__(self, clean_videos: Sequence[torch.Tensor], simulator, fps: float = 60.0, crop_size: int = 256,
                 temporal_policy: str = "k9_200ms", train: bool = True):
        self.clean_videos = list(clean_videos)
        self.simulator = simulator
        self.fps = float(fps)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.temporal_sampler = TemporalSampler(policy=temporal_policy)
        self.index = [(vid, t) for vid, video in enumerate(self.clean_videos) for t in range(video.shape[0])]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id, center_idx = self.index[idx]
        clean_video = self.clean_videos[video_id]
        sample = self.temporal_sampler.sample(center_idx, self.fps, clean_video.shape[0])
        clean_frames = clean_video[sample.indices]
        target = clean_video[center_idx]
        clean_frames, target = self._crop(clean_frames, target)

        timestamps = torch.tensor(sample.indices, dtype=torch.float32) / self.fps
        distorted, meta = self.simulator(clean_frames, timestamps, return_meta=True)

        return {
            "frames": distorted.float(),
            "target": target.float(),
            "dt": torch.tensor(sample.dt, dtype=torch.float32),
            "valid": torch.tensor(sample.valid, dtype=torch.float32),
            "meta": meta,
        }

    def _crop(self, frames: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = frames.shape
        cs = self.crop_size
        if cs <= 0 or (H == cs and W == cs):
            return frames, target
        if H < cs or W < cs:
            raise ValueError(f"crop_size={cs} is larger than frame size {(H, W)}")
        if self.train:
            y = torch.randint(0, H - cs + 1, (1,)).item()
            x = torch.randint(0, W - cs + 1, (1,)).item()
        else:
            y = (H - cs) // 2
            x = (W - cs) // 2
        return frames[:, :, y:y+cs, x:x+cs], target[:, y:y+cs, x:x+cs]


class SyntheticMovingShapesDataset(Dataset):
    """
    Debug dataset for smoke tests.
    It creates clean synthetic videos and optionally applies the simulator.
    """

    def __init__(self, num_samples: int = 128, num_clean_frames: int = 40, image_size: int = 128,
                 fps: float = 60.0, simulator=None, temporal_policy: str = "k9_200ms"):
        self.num_samples = int(num_samples)
        self.num_clean_frames = int(num_clean_frames)
        self.image_size = int(image_size)
        self.fps = float(fps)
        self.simulator = simulator
        self.temporal_sampler = TemporalSampler(temporal_policy)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video = self._make_video(idx)
        center_idx = int((idx * 7) % self.num_clean_frames)
        sample = self.temporal_sampler.sample(center_idx, self.fps, self.num_clean_frames)
        clean_frames = video[sample.indices]
        target = video[center_idx]
        timestamps = torch.tensor(sample.indices, dtype=torch.float32) / self.fps

        if self.simulator is None:
            distorted, meta = clean_frames, {}
        else:
            distorted, meta = self.simulator(clean_frames, timestamps, return_meta=True)

        return {
            "frames": distorted.float().cpu(),
            "target": target.float().cpu(),
            "dt": torch.tensor(sample.dt, dtype=torch.float32),
            "valid": torch.tensor(sample.valid, dtype=torch.float32),
            "meta": meta,
        }

    def _make_video(self, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        N = self.num_clean_frames
        H = W = self.image_size
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        yy = yy.float() / max(H - 1, 1)
        xx = xx.float() / max(W - 1, 1)
        base = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0)

        video = []
        for t in range(N):
            frame = base.clone()
            cx = int((0.2 + 0.6 * ((t / max(N - 1, 1) + 0.13 * (seed % 5)) % 1.0)) * W)
            cy = int((0.35 + 0.25 * torch.sin(torch.tensor(0.25 * t + seed)).item()) * H)
            size = max(8, H // 8)
            x0, x1 = max(0, cx - size), min(W, cx + size)
            y0, y1 = max(0, cy - size), min(H, cy + size)
            color = torch.rand(3, generator=g) * 0.5 + 0.5
            frame[:, y0:y1, x0:x1] = color[:, None, None]
            frame[:, H // 5:H // 5 + 4, W // 6:5 * W // 6] = 0.05
            frame[:, 2 * H // 3:2 * H // 3 + 4, W // 4:3 * W // 4] = 0.95
            video.append(frame.clamp(0, 1))
        return torch.stack(video, dim=0)
