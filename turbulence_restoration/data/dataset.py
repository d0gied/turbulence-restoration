
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import torch
from torch.utils.data import Dataset

from turbulence_restoration.data.temporal_sampler import TemporalSampler

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FRAME_CACHE_FORMAT = "turbulence_restoration.frame_cache.v1"


def _crop_frames(
    frames: torch.Tensor,
    target: torch.Tensor,
    crop_size: int,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = frames.shape
    if crop_size <= 0 or (height == crop_size and width == crop_size):
        return frames, target
    if height < crop_size or width < crop_size:
        raise ValueError(f"crop_size={crop_size} is larger than frame size {(height, width)}")

    if train:
        y = torch.randint(0, height - crop_size + 1, (1,)).item()
        x = torch.randint(0, width - crop_size + 1, (1,)).item()
    else:
        y = (height - crop_size) // 2
        x = (width - crop_size) // 2
    return (
        frames[:, :, y:y + crop_size, x:x + crop_size],
        target[:, y:y + crop_size, x:x + crop_size],
    )


def _read_frame_uint8(path: Path) -> torch.Tensor:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame)
    return tensor.permute(2, 0, 1).contiguous()


def _read_frame_tensor(path: Path) -> torch.Tensor:
    return _read_frame_uint8(path).float().div_(255.0)


def _uint8_to_float(frame: torch.Tensor) -> torch.Tensor:
    return frame.float().div_(255.0)


class TurbulenceDataset(Dataset):
    """
    On-the-fly dataset:
      clean video sequence -> clean window + clean center.
    clean_videos: list/sequence of tensors [N,3,H,W] in [0,1].
    """

    def __init__(
        self,
        clean_videos: Sequence[torch.Tensor],
        simulator=None,
        fps: float = 60.0,
        crop_size: int = 256,
        temporal_policy: str = "k9_200ms",
        train: bool = True,
        include_frames: bool = True,
    ):
        self.clean_videos = list(clean_videos)
        self.simulator = simulator
        self.fps = float(fps)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.include_frames = bool(include_frames)
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
        out = {
            "clean_frames": clean_frames.float(),
            "target": target.float(),
            "timestamps": timestamps.float(),
            "dt": torch.tensor(sample.dt, dtype=torch.float32),
            "valid": torch.tensor(sample.valid, dtype=torch.float32),
        }
        if self.include_frames or self.simulator is not None:
            out["frames"] = (
                clean_frames if self.simulator is None else self.simulator(clean_frames, timestamps, return_meta=False)
            ).float()
        return out

    def _crop(self, frames: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _crop_frames(frames, target, self.crop_size, self.train)


class RealVideoFramesDataset(Dataset):
    """
    Clean clips stored on disk as:
      root/video_0001/000000.png
      root/video_0002/000000.png
    """

    def __init__(
        self,
        root: str | Path,
        simulator=None,
        fps: float = 60.0,
        crop_size: int = 256,
        temporal_policy: str = "k9_200ms",
        train: bool = True,
        include_frames: bool = True,
        preload: bool = False,
        cache_path: str | Path | None = None,
    ):
        self.root = Path(root).expanduser()
        if not self.root.is_absolute():
            self.root = (Path.cwd() / self.root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.simulator = simulator
        self.fps = float(fps)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.include_frames = bool(include_frames)
        self.temporal_sampler = TemporalSampler(policy=temporal_policy)
        self.cache_path = self._resolve_path(cache_path) if cache_path is not None else None
        cached_video_frames = None
        if self.cache_path is None:
            self.video_frames = self._discover_videos(self.root)
        else:
            self.video_frames, cached_video_frames = self._load_frame_cache(self.cache_path)

        self.preload = bool(preload)
        self.cached_video_frames = (
            cached_video_frames
            if cached_video_frames is not None
            else self._preload_videos(self.video_frames) if self.preload else None
        )
        self.index = [(video_id, frame_id) for video_id, frames in enumerate(self.video_frames) for frame_id in range(len(frames))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id, center_idx = self.index[idx]
        frame_paths = self.video_frames[video_id]
        sample = self.temporal_sampler.sample(center_idx, self.fps, len(frame_paths))

        if self.cached_video_frames is None:
            frame_source = [_read_frame_uint8(frame_paths[i]) for i in sample.indices]
            target = _read_frame_uint8(frame_paths[center_idx])
        else:
            cached_frames = self.cached_video_frames[video_id]
            frame_source = [cached_frames[i] for i in sample.indices]
            target = cached_frames[center_idx]

        clean_frames = torch.stack(frame_source, dim=0)
        clean_frames, target = self._crop(clean_frames, target)
        clean_frames = _uint8_to_float(clean_frames)
        target = _uint8_to_float(target)

        timestamps = torch.tensor(sample.indices, dtype=torch.float32) / self.fps
        out = {
            "clean_frames": clean_frames.float(),
            "target": target.float(),
            "timestamps": timestamps.float(),
            "dt": torch.tensor(sample.dt, dtype=torch.float32),
            "valid": torch.tensor(sample.valid, dtype=torch.float32),
        }
        if self.include_frames or self.simulator is not None:
            out["frames"] = (
                clean_frames if self.simulator is None else self.simulator(clean_frames, timestamps, return_meta=False)
            ).float()
        return out

    def _crop(self, frames: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _crop_frames(frames, target, self.crop_size, self.train)

    @staticmethod
    def _discover_videos(root: Path) -> List[List[Path]]:
        videos: List[List[Path]] = []
        for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            frame_paths = sorted(
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if frame_paths:
                videos.append(frame_paths)

        if not videos:
            raise RuntimeError(
                f"No frame folders found in {root}. Expected directories like {root / 'video_0001'}"
            )
        return videos

    @staticmethod
    def _preload_videos(video_frames: List[List[Path]]) -> List[torch.Tensor]:
        cached_videos = []
        for frame_paths in video_frames:
            cached_videos.append(torch.stack([_read_frame_uint8(path) for path in frame_paths], dim=0))
        return cached_videos

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        return resolved

    @staticmethod
    def _load_frame_cache(cache_path: Path) -> Tuple[List[List[Path]], List[torch.Tensor]]:
        if not cache_path.is_file():
            raise FileNotFoundError(f"Frame cache does not exist: {cache_path}")

        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("format") != FRAME_CACHE_FORMAT:
            raise ValueError(f"Unsupported frame cache format in {cache_path}")

        video_frames: List[List[Path]] = []
        cached_videos: List[torch.Tensor] = []
        for video in cache.get("videos", []):
            frames = video["frames"]
            if frames.dtype != torch.uint8 or frames.dim() != 4 or frames.shape[1] != 3:
                raise ValueError(
                    f"Expected cached frames as uint8 [N,3,H,W], got {frames.dtype} {tuple(frames.shape)}"
                )
            filenames = video.get("filenames") or [f"{idx:06d}.png" for idx in range(frames.shape[0])]
            video_name = str(video.get("name", f"video_{len(video_frames):04d}"))
            video_frames.append([Path(video_name) / str(filename) for filename in filenames])
            cached_videos.append(frames.contiguous())

        if not cached_videos:
            raise RuntimeError(f"Frame cache contains no videos: {cache_path}")
        return video_frames, cached_videos


class SyntheticMovingShapesDataset(Dataset):
    """
    Debug dataset for smoke tests.
    It creates clean synthetic videos and optionally applies the simulator.
    """

    def __init__(
        self,
        num_samples: int = 128,
        num_clean_frames: int = 40,
        image_size: int = 128,
        fps: float = 60.0,
        simulator=None,
        temporal_policy: str = "k9_200ms",
        include_frames: bool = True,
    ):
        self.num_samples = int(num_samples)
        self.num_clean_frames = int(num_clean_frames)
        self.image_size = int(image_size)
        self.fps = float(fps)
        self.simulator = simulator
        self.include_frames = bool(include_frames)
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

        out = {
            "clean_frames": clean_frames.float().cpu(),
            "target": target.float().cpu(),
            "timestamps": timestamps.float().cpu(),
            "dt": torch.tensor(sample.dt, dtype=torch.float32),
            "valid": torch.tensor(sample.valid, dtype=torch.float32),
        }
        if self.include_frames or self.simulator is not None:
            frames = clean_frames if self.simulator is None else self.simulator(clean_frames, timestamps, return_meta=False)
            out["frames"] = frames.float().cpu()
        return out

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
