
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from turbulence_restoration.data.temporal_sampler import TemporalSampler

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FRAME_CACHE_FORMAT = "turbulence_restoration.frame_cache.v1"
FRAME_CACHE_INDEX_FILENAME = "index.json"


def _crop_frames(
    frames: torch.Tensor,
    target: torch.Tensor,
    crop_size: int,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = frames.shape[-2:]
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
        frames[..., y:y + crop_size, x:x + crop_size],
        target[..., y:y + crop_size, x:x + crop_size],
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


def _sequence_center_offsets(output_frames: int, center_stride: int) -> np.ndarray:
    output_frames = int(output_frames)
    center_stride = int(center_stride)
    if output_frames < 1:
        raise ValueError(f"output_frames must be >= 1, got {output_frames}")
    if center_stride < 1:
        raise ValueError(f"center_stride must be >= 1, got {center_stride}")
    if output_frames == 1:
        return np.array([0], dtype=np.int64)
    if output_frames % 2 == 0:
        raise ValueError("sequence output_frames must be odd so there is a central output frame")
    half = output_frames // 2
    return np.arange(-half, half + 1, dtype=np.int64) * center_stride


def _temporal_offsets_idx(temporal_sampler: TemporalSampler, fps: float) -> np.ndarray:
    return np.round(temporal_sampler.offsets_sec * float(fps)).astype(np.int64)


def _valid_sequence_centers(
    num_frames: int,
    temporal_sampler: TemporalSampler,
    fps: float,
    output_frames: int,
    center_stride: int,
) -> range:
    if output_frames <= 1:
        return range(num_frames)

    center_offsets = _sequence_center_offsets(output_frames, center_stride)
    temporal_offsets = _temporal_offsets_idx(temporal_sampler, fps)
    all_offsets = center_offsets[:, None] + temporal_offsets[None, :]
    margin_before = max(0, int(-all_offsets.min()))
    margin_after = max(0, int(all_offsets.max()))
    if num_frames > margin_before + margin_after:
        return range(margin_before, num_frames - margin_after)
    return range(num_frames)


def _sequence_layout(
    center_idx: int,
    num_frames: int,
    temporal_sampler: TemporalSampler,
    fps: float,
    output_frames: int,
    center_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_offsets = _sequence_center_offsets(output_frames, center_stride)
    temporal_offsets = _temporal_offsets_idx(temporal_sampler, fps)
    raw_centers = int(center_idx) + center_offsets
    raw_windows = raw_centers[:, None] + temporal_offsets[None, :]

    raw_min = int(raw_windows.min())
    raw_max = int(raw_windows.max())
    raw_clip = np.arange(raw_min, raw_max + 1, dtype=np.int64)

    clip_indices = TemporalSampler._reflect_indices(raw_clip, num_frames)
    target_indices = TemporalSampler._reflect_indices(raw_centers, num_frames)
    window_positions = raw_windows - raw_min
    dt_seq = np.broadcast_to(temporal_sampler.offsets_sec.astype(np.float32), raw_windows.shape).copy()
    valid_seq = np.ones(raw_windows.shape, dtype=np.float32)
    return clip_indices, target_indices, window_positions.astype(np.int64), dt_seq, valid_seq


def _gather_sequence_windows(clip: torch.Tensor, window_positions: np.ndarray | torch.Tensor) -> torch.Tensor:
    positions = torch.as_tensor(window_positions, device=clip.device, dtype=torch.long)
    return clip[positions]


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
        output_frames: int = 1,
        center_stride: int = 1,
    ):
        self.clean_videos = list(clean_videos)
        self.simulator = simulator
        self.fps = float(fps)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.include_frames = bool(include_frames)
        self.temporal_sampler = TemporalSampler(policy=temporal_policy)
        self.output_frames = int(output_frames)
        self.center_stride = int(center_stride)
        _sequence_center_offsets(self.output_frames, self.center_stride)
        self.index = [
            (vid, t)
            for vid, video in enumerate(self.clean_videos)
            for t in _valid_sequence_centers(
                video.shape[0],
                self.temporal_sampler,
                self.fps,
                self.output_frames,
                self.center_stride,
            )
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id, center_idx = self.index[idx]
        clean_video = self.clean_videos[video_id]
        if self.output_frames > 1:
            return self._getitem_sequence(clean_video, center_idx)

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

    def _getitem_sequence(self, clean_video: torch.Tensor, center_idx: int) -> Dict[str, Any]:
        clip_indices, target_indices, window_positions, dt_seq, valid_seq = _sequence_layout(
            center_idx,
            clean_video.shape[0],
            self.temporal_sampler,
            self.fps,
            self.output_frames,
            self.center_stride,
        )
        clean_clip = clean_video[clip_indices]
        target_seq = clean_video[target_indices]
        clean_clip, target_seq = self._crop(clean_clip, target_seq)

        timestamps = torch.tensor(clip_indices, dtype=torch.float32) / self.fps
        out = {
            "clean_clip": clean_clip.float(),
            "target_seq": target_seq.float(),
            "clip_timestamps": timestamps.float(),
            "window_positions": torch.tensor(window_positions, dtype=torch.long),
            "dt_seq": torch.tensor(dt_seq, dtype=torch.float32),
            "valid_seq": torch.tensor(valid_seq, dtype=torch.float32),
        }
        if self.include_frames or self.simulator is not None:
            distorted_clip = (
                clean_clip if self.simulator is None else self.simulator(clean_clip, timestamps, return_meta=False)
            )
            out["frames_seq"] = _gather_sequence_windows(distorted_clip.float(), window_positions)
        return out


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
        cache_lru_size: int = 1,
        output_frames: int = 1,
        center_stride: int = 1,
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
        self.output_frames = int(output_frames)
        self.center_stride = int(center_stride)
        _sequence_center_offsets(self.output_frames, self.center_stride)
        self.cache_path = self._resolve_path(cache_path) if cache_path is not None else None
        self.preload = bool(preload)
        self.cache_video_files: List[Path] | None = None
        self.cache_lru_size = max(1, int(cache_lru_size))
        self._cache_video_lru: OrderedDict[int, torch.Tensor] = OrderedDict()
        cached_video_frames = None
        if self.cache_path is None:
            self.video_frames = self._discover_videos(self.root)
        elif self.cache_path.is_dir() and not self.preload:
            self.video_frames, self.cache_video_files = self._load_frame_cache_metadata(self.cache_path)
        else:
            self.video_frames, cached_video_frames = self._load_frame_cache(self.cache_path)

        self.cached_video_frames = (
            cached_video_frames
            if cached_video_frames is not None
            else self._preload_videos(self.video_frames)
            if self.preload and self.cache_video_files is None
            else None
        )
        self.index = [
            (video_id, frame_id)
            for video_id, frames in enumerate(self.video_frames)
            for frame_id in _valid_sequence_centers(
                len(frames),
                self.temporal_sampler,
                self.fps,
                self.output_frames,
                self.center_stride,
            )
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id, center_idx = self.index[idx]
        frame_paths = self.video_frames[video_id]
        if self.output_frames > 1:
            return self._getitem_sequence(video_id, center_idx)

        sample = self.temporal_sampler.sample(center_idx, self.fps, len(frame_paths))

        if self.cached_video_frames is not None:
            cached_frames = self.cached_video_frames[video_id]
            frame_source = [cached_frames[i] for i in sample.indices]
            target = cached_frames[center_idx]
        elif self.cache_video_files is not None:
            cached_frames = self._get_cached_video(video_id)
            frame_source = [cached_frames[i] for i in sample.indices]
            target = cached_frames[center_idx]
        else:
            frame_source = [_read_frame_uint8(frame_paths[i]) for i in sample.indices]
            target = _read_frame_uint8(frame_paths[center_idx])

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

    def _get_video_frame_uint8(self, video_id: int, frame_idx: int) -> torch.Tensor:
        if self.cached_video_frames is not None:
            return self.cached_video_frames[video_id][frame_idx]
        if self.cache_video_files is not None:
            return self._get_cached_video(video_id)[frame_idx]
        return _read_frame_uint8(self.video_frames[video_id][frame_idx])

    def _getitem_sequence(self, video_id: int, center_idx: int) -> Dict[str, Any]:
        num_frames = len(self.video_frames[video_id])
        clip_indices, target_indices, window_positions, dt_seq, valid_seq = _sequence_layout(
            center_idx,
            num_frames,
            self.temporal_sampler,
            self.fps,
            self.output_frames,
            self.center_stride,
        )

        clean_clip = torch.stack(
            [self._get_video_frame_uint8(video_id, int(frame_idx)) for frame_idx in clip_indices],
            dim=0,
        )
        target_seq = torch.stack(
            [self._get_video_frame_uint8(video_id, int(frame_idx)) for frame_idx in target_indices],
            dim=0,
        )
        clean_clip, target_seq = self._crop(clean_clip, target_seq)
        clean_clip = _uint8_to_float(clean_clip)
        target_seq = _uint8_to_float(target_seq)

        timestamps = torch.tensor(clip_indices, dtype=torch.float32) / self.fps
        out = {
            "clean_clip": clean_clip.float(),
            "target_seq": target_seq.float(),
            "clip_timestamps": timestamps.float(),
            "window_positions": torch.tensor(window_positions, dtype=torch.long),
            "dt_seq": torch.tensor(dt_seq, dtype=torch.float32),
            "valid_seq": torch.tensor(valid_seq, dtype=torch.float32),
        }
        if self.include_frames or self.simulator is not None:
            distorted_clip = (
                clean_clip if self.simulator is None else self.simulator(clean_clip, timestamps, return_meta=False)
            )
            out["frames_seq"] = _gather_sequence_windows(distorted_clip.float(), window_positions)
        return out

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

    def _get_cached_video(self, video_id: int) -> torch.Tensor:
        cached_frames = self._cache_video_lru.get(video_id)
        if cached_frames is not None:
            self._cache_video_lru.move_to_end(video_id)
            return cached_frames

        if self.cache_video_files is None:
            raise RuntimeError("Lazy cache is not configured")

        video_frames, cached_videos = self._load_frame_cache_file(self.cache_video_files[video_id])
        if len(video_frames) != 1 or len(cached_videos) != 1:
            raise ValueError(
                f"Lazy cache file must contain exactly one video: {self.cache_video_files[video_id]}"
            )
        if len(video_frames[0]) != len(self.video_frames[video_id]):
            raise ValueError(
                f"Lazy cache metadata does not match frames in {self.cache_video_files[video_id]}"
            )

        cached_frames = cached_videos[0]
        self._cache_video_lru[video_id] = cached_frames
        self._cache_video_lru.move_to_end(video_id)
        while len(self._cache_video_lru) > self.cache_lru_size:
            self._cache_video_lru.popitem(last=False)
        return cached_frames

    @staticmethod
    def _load_frame_cache(cache_path: Path) -> Tuple[List[List[Path]], List[torch.Tensor]]:
        if cache_path.is_dir():
            index_path = cache_path / FRAME_CACHE_INDEX_FILENAME
            if index_path.is_file():
                with index_path.open("r", encoding="utf-8") as f:
                    index = json.load(f)
                if index.get("format") != FRAME_CACHE_FORMAT:
                    raise ValueError(f"Unsupported frame cache index format in {index_path}")
                cache_files = [
                    cache_path / str(video["file"])
                    for video in index.get("videos", [])
                ]
            else:
                cache_files = sorted(
                    path
                    for path in cache_path.iterdir()
                    if path.is_file() and path.suffix == ".pt"
                )
            if not cache_files:
                raise RuntimeError(f"Frame cache directory contains no .pt files: {cache_path}")
            missing_files = [path for path in cache_files if not path.is_file()]
            if missing_files:
                raise FileNotFoundError(f"Frame cache index references missing file: {missing_files[0]}")
        elif cache_path.is_file():
            cache_files = [cache_path]
        else:
            raise FileNotFoundError(f"Frame cache does not exist: {cache_path}")

        video_frames: List[List[Path]] = []
        cached_videos: List[torch.Tensor] = []
        for cache_file in cache_files:
            loaded_video_frames, loaded_cached_videos = RealVideoFramesDataset._load_frame_cache_file(cache_file)
            video_frames.extend(loaded_video_frames)
            cached_videos.extend(loaded_cached_videos)

        if not cached_videos:
            raise RuntimeError(f"Frame cache contains no videos: {cache_path}")
        return video_frames, cached_videos

    @staticmethod
    def _load_frame_cache_metadata(cache_path: Path) -> Tuple[List[List[Path]], List[Path]]:
        if not cache_path.is_dir():
            raise FileNotFoundError(f"Frame cache directory does not exist: {cache_path}")

        index_path = cache_path / FRAME_CACHE_INDEX_FILENAME
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            if index.get("format") != FRAME_CACHE_FORMAT:
                raise ValueError(f"Unsupported frame cache index format in {index_path}")

            video_frames: List[List[Path]] = []
            cache_files: List[Path] = []
            for video_idx, video in enumerate(index.get("videos", [])):
                cache_file = cache_path / str(video["file"])
                if not cache_file.is_file():
                    raise FileNotFoundError(f"Frame cache index references missing file: {cache_file}")
                frame_count = int(video["frames"])
                filenames = video.get("filenames") or [
                    f"{idx:06d}.png"
                    for idx in range(frame_count)
                ]
                if len(filenames) != frame_count:
                    raise ValueError(f"Frame cache index has invalid filenames count for {cache_file}")
                video_name = str(video.get("name", f"video_{video_idx:04d}"))
                video_frames.append([Path(video_name) / str(filename) for filename in filenames])
                cache_files.append(cache_file)

            if not cache_files:
                raise RuntimeError(f"Frame cache directory contains no indexed videos: {cache_path}")
            return video_frames, cache_files

        cache_files = sorted(
            path
            for path in cache_path.iterdir()
            if path.is_file() and path.suffix == ".pt"
        )
        if not cache_files:
            raise RuntimeError(f"Frame cache directory contains no .pt files: {cache_path}")

        video_frames = []
        for cache_file in cache_files:
            loaded_video_frames, loaded_cached_videos = RealVideoFramesDataset._load_frame_cache_file(cache_file)
            if len(loaded_video_frames) != 1 or len(loaded_cached_videos) != 1:
                raise ValueError(
                    f"Lazy cache directory without {FRAME_CACHE_INDEX_FILENAME} requires one video per file: {cache_file}"
                )
            video_frames.extend(loaded_video_frames)
        return video_frames, cache_files

    @staticmethod
    def _load_frame_cache_file(cache_path: Path) -> Tuple[List[List[Path]], List[torch.Tensor]]:
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
        output_frames: int = 1,
        center_stride: int = 1,
    ):
        self.num_samples = int(num_samples)
        self.num_clean_frames = int(num_clean_frames)
        self.image_size = int(image_size)
        self.fps = float(fps)
        self.simulator = simulator
        self.include_frames = bool(include_frames)
        self.temporal_sampler = TemporalSampler(temporal_policy)
        self.output_frames = int(output_frames)
        self.center_stride = int(center_stride)
        _sequence_center_offsets(self.output_frames, self.center_stride)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video = self._make_video(idx)
        valid_centers = _valid_sequence_centers(
            self.num_clean_frames,
            self.temporal_sampler,
            self.fps,
            self.output_frames,
            self.center_stride,
        )
        center_values = list(valid_centers)
        center_idx = center_values[int((idx * 7) % len(center_values))]
        if self.output_frames > 1:
            return self._getitem_sequence(video, center_idx)

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

    def _getitem_sequence(self, video: torch.Tensor, center_idx: int) -> Dict[str, torch.Tensor]:
        clip_indices, target_indices, window_positions, dt_seq, valid_seq = _sequence_layout(
            center_idx,
            self.num_clean_frames,
            self.temporal_sampler,
            self.fps,
            self.output_frames,
            self.center_stride,
        )
        clean_clip = video[clip_indices]
        target_seq = video[target_indices]
        timestamps = torch.tensor(clip_indices, dtype=torch.float32) / self.fps

        out = {
            "clean_clip": clean_clip.float().cpu(),
            "target_seq": target_seq.float().cpu(),
            "clip_timestamps": timestamps.float().cpu(),
            "window_positions": torch.tensor(window_positions, dtype=torch.long),
            "dt_seq": torch.tensor(dt_seq, dtype=torch.float32),
            "valid_seq": torch.tensor(valid_seq, dtype=torch.float32),
        }
        if self.include_frames or self.simulator is not None:
            distorted_clip = clean_clip if self.simulator is None else self.simulator(clean_clip, timestamps, return_meta=False)
            out["frames_seq"] = _gather_sequence_windows(distorted_clip.float(), window_positions).cpu()
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
