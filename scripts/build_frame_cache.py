from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import torch
from tqdm import tqdm

from turbulence_restoration.data.dataset import FRAME_CACHE_FORMAT, IMAGE_EXTENSIONS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "clean_videos"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "cache"
VIDEO_EXTENSIONS = (
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
)


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def discover_frame_videos(root: Path) -> List[List[Path]]:
    frame_videos: List[List[Path]] = []
    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        frame_paths = sorted(
            path for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if frame_paths:
            frame_videos.append(frame_paths)
    if not frame_videos:
        raise RuntimeError(f"No frame folders found in {root}")
    return frame_videos


def discover_splits(
    input_root: Path,
    requested_splits: Optional[Sequence[str]],
) -> List[str]:
    if requested_splits is not None:
        return list(requested_splits)

    splits = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    if not splits:
        raise RuntimeError(f"No split directories found in {input_root}")
    return splits


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def list_video_files(split_dir: Path, recursive: bool) -> List[Path]:
    iterator = split_dir.rglob("*") if recursive else split_dir.iterdir()
    return sorted(
        (path for path in iterator if is_video_file(path)),
        key=lambda path: str(path.relative_to(split_dir)).lower(),
    )


def resize_to_limit(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    scale = min(
        float(max_width) / float(width) if max_width > 0 else 1.0,
        float(max_height) / float(height) if max_height > 0 else 1.0,
        1.0,
    )
    if scale >= 1.0:
        return frame

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_frame_uint8(path: Path, max_width: int, max_height: int) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {path}")

    original_shape = frame.shape[:2]
    frame = resize_to_limit(frame, max_width=max_width, max_height=max_height)
    resized_shape = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
    return tensor, original_shape, resized_shape


def read_video_frames(
    video_path: Path,
    max_width: int,
    max_height: int,
) -> tuple[torch.Tensor, list[str], list[tuple[int, int]], list[tuple[int, int]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_tensors = []
    filenames = []
    original_shapes = []
    resized_shapes = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            original_shape = frame.shape[:2]
            frame = resize_to_limit(frame, max_width=max_width, max_height=max_height)
            resized_shape = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensors.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())
            filenames.append(f"{frame_index:06d}.png")
            original_shapes.append(tuple(int(v) for v in original_shape))
            resized_shapes.append(tuple(int(v) for v in resized_shape))
            frame_index += 1
    finally:
        capture.release()

    if not frame_tensors:
        raise RuntimeError(f"No frames decoded from {video_path}")

    return torch.stack(frame_tensors, dim=0), filenames, original_shapes, resized_shapes


def write_cache(
    *,
    root: Path,
    output: Path,
    payload_videos: list[dict],
    total_frames: int,
    total_bytes: int,
    max_width: int,
    max_height: int,
) -> None:
    payload = {
        "format": FRAME_CACHE_FORMAT,
        "root": str(root),
        "layout": "NCHW_RGB_UINT8",
        "max_width": int(max_width),
        "max_height": int(max_height),
        "num_videos": len(payload_videos),
        "num_frames": total_frames,
        "num_bytes": total_bytes,
        "videos": payload_videos,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        {
            "output": str(output),
            "videos": len(payload_videos),
            "frames": total_frames,
            "raw_gib": round(total_bytes / 1024**3, 2),
        }
    )


def build_cache_from_frame_folders(root: Path, output: Path, max_width: int, max_height: int) -> None:
    videos = discover_frame_videos(root)
    payload_videos = []
    total_frames = 0
    total_bytes = 0

    for frame_paths in tqdm(videos, desc=f"cache {root.name}"):
        frame_tensors = []
        original_shapes = []
        resized_shapes = []
        for path in frame_paths:
            frame, original_shape, resized_shape = read_frame_uint8(
                path,
                max_width=max_width,
                max_height=max_height,
            )
            frame_tensors.append(frame)
            original_shapes.append(tuple(int(v) for v in original_shape))
            resized_shapes.append(tuple(int(v) for v in resized_shape))

        frames = torch.stack(frame_tensors, dim=0)
        total_frames += int(frames.shape[0])
        total_bytes += int(frames.numel() * frames.element_size())
        payload_videos.append(
            {
                "name": frame_paths[0].parent.name,
                "filenames": [path.name for path in frame_paths],
                "original_shapes": original_shapes,
                "resized_shapes": resized_shapes,
                "frames": frames,
            }
        )

    write_cache(
        root=root,
        output=output,
        payload_videos=payload_videos,
        total_frames=total_frames,
        total_bytes=total_bytes,
        max_width=max_width,
        max_height=max_height,
    )


def build_cache_from_video_files(
    split_dir: Path,
    output: Path,
    video_paths: list[Path],
    max_width: int,
    max_height: int,
    start_index: int,
    video_digits: int,
) -> None:
    payload_videos = []
    total_frames = 0
    total_bytes = 0

    for offset, video_path in enumerate(tqdm(video_paths, desc=f"cache {split_dir.name}")):
        frames, filenames, original_shapes, resized_shapes = read_video_frames(
            video_path,
            max_width=max_width,
            max_height=max_height,
        )
        total_frames += int(frames.shape[0])
        total_bytes += int(frames.numel() * frames.element_size())
        video_index = start_index + offset
        payload_videos.append(
            {
                "name": f"video_{video_index:0{video_digits}d}",
                "source": str(video_path),
                "filenames": filenames,
                "original_shapes": original_shapes,
                "resized_shapes": resized_shapes,
                "frames": frames,
            }
        )

    write_cache(
        root=split_dir,
        output=output,
        payload_videos=payload_videos,
        total_frames=total_frames,
        total_bytes=total_bytes,
        max_width=max_width,
        max_height=max_height,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack clean videos or frame folders into torch uint8 cache files.")
    parser.add_argument("--root", type=Path, default=None, help="Legacy frame-folder root, e.g. datasets/clean/train")
    parser.add_argument("--output", type=Path, default=None, help="Output .pt path for --root mode")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root with split directories containing source videos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root where split cache files will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Split names to process. By default all subdirectories of input-root are used.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for videos recursively inside each split directory.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First video index inside each split.",
    )
    parser.add_argument(
        "--video-digits",
        type=int,
        default=4,
        help="Zero-padding width for cached video names.",
    )
    parser.add_argument("--max-width", type=int, default=1920, help="Downscale frames wider than this. Use 0 to disable.")
    parser.add_argument("--max-height", type=int, default=1080, help="Downscale frames taller than this. Use 0 to disable.")
    args = parser.parse_args()

    if args.max_width < 0:
        raise ValueError("--max-width must be non-negative")
    if args.max_height < 0:
        raise ValueError("--max-height must be non-negative")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.video_digits <= 0:
        raise ValueError("--video-digits must be positive")

    if args.root is not None:
        if args.output is None:
            raise ValueError("--output is required with --root")
        root = resolve_path(args.root)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        output = resolve_path(args.output)
        build_cache_from_frame_folders(root, output, max_width=args.max_width, max_height=args.max_height)
        return

    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    total_splits = 0
    for split_name in discover_splits(input_root, args.splits):
        split_dir = input_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        video_paths = list_video_files(split_dir, recursive=args.recursive)
        if not video_paths:
            print(f"{split_name}: no video files found, skipping")
            continue

        output = output_root / f"clean_{split_name}.pt"
        build_cache_from_video_files(
            split_dir=split_dir,
            output=output,
            video_paths=video_paths,
            max_width=args.max_width,
            max_height=args.max_height,
            start_index=args.start_index,
            video_digits=args.video_digits,
        )
        total_splits += 1

    if total_splits == 0:
        raise RuntimeError(f"No caches built from {input_root}")


if __name__ == "__main__":
    main()
