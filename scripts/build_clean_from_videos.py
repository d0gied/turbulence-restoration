from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "datasets" / "clean_videos"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "clean"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract videos from datasets/clean_videos/<split> into "
            "datasets/clean/<split>/video_0001/000000.png."
        )
    )
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
        help="Root where extracted frame directories will be written.",
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
        "--overwrite",
        action="store_true",
        help="Delete existing output split directories before writing frames.",
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
        help="Zero-padding width for video directories.",
    )
    parser.add_argument(
        "--frame-digits",
        type=int,
        default=6,
        help="Zero-padding width for frame file names.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="OpenCV PNG compression level from 0 to 9.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="Downscale extracted frames wider than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Downscale extracted frames taller than this. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


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


def prepare_output_split(split_output_dir: Path, overwrite: bool) -> None:
    if split_output_dir.exists():
        if overwrite:
            shutil.rmtree(split_output_dir)
        elif any(split_output_dir.iterdir()):
            raise FileExistsError(
                f"{split_output_dir} is not empty. Pass --overwrite to rebuild it."
            )
    split_output_dir.mkdir(parents=True, exist_ok=True)


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


def extract_video(
    video_path: Path,
    frame_dir: Path,
    frame_digits: int,
    png_compression: int,
    max_width: int,
    max_height: int,
) -> int:
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_index = 0
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = resize_to_limit(frame, max_width=max_width, max_height=max_height)
            frame_path = frame_dir / f"{frame_index:0{frame_digits}d}.png"
            if not cv2.imwrite(str(frame_path), frame, png_params):
                raise RuntimeError(f"Could not write frame: {frame_path}")
            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frame_index


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.video_digits <= 0:
        raise ValueError("--video-digits must be positive")
    if args.frame_digits <= 0:
        raise ValueError("--frame-digits must be positive")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be between 0 and 9")
    if args.max_width < 0:
        raise ValueError("--max-width must be non-negative")
    if args.max_height < 0:
        raise ValueError("--max-height must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist or is not a directory: {input_root}"
        )

    total_videos = 0
    total_frames = 0
    splits = discover_splits(input_root, args.splits)

    for split_name in splits:
        split_input_dir = input_root / split_name
        if not split_input_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_input_dir}")

        videos = list_video_files(split_input_dir, recursive=args.recursive)
        if not videos:
            print(f"{split_name}: no video files found, skipping")
            continue

        split_output_dir = output_root / split_name
        prepare_output_split(split_output_dir, overwrite=args.overwrite)
        print(f"{split_name}: extracting {len(videos)} video(s)")

        for offset, video_path in enumerate(videos):
            video_index = args.start_index + offset
            frame_dir = split_output_dir / f"video_{video_index:0{args.video_digits}d}"
            frame_count = extract_video(
                video_path=video_path,
                frame_dir=frame_dir,
                frame_digits=args.frame_digits,
                png_compression=args.png_compression,
                max_width=args.max_width,
                max_height=args.max_height,
            )
            total_videos += 1
            total_frames += frame_count
            print(
                f"  {video_path.name} -> {frame_dir.relative_to(output_root)} "
                f"({frame_count} frames)"
            )

    if total_videos == 0:
        raise RuntimeError(f"No videos extracted from {input_root}")

    print(
        f"Done: extracted {total_frames} frames from {total_videos} "
        f"video(s) into {output_root}"
    )


if __name__ == "__main__":
    main()
