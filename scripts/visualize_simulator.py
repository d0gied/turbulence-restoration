from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turbulence_restoration.data.temporal_sampler import TemporalSampler
from turbulence_restoration.simulator.gpu_turbulence import GPUTurbulenceSimulator, TurbulenceConfig


DEFAULT_OUTPUT_DIR = ROOT / "results" / "simulator_debug"
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one clean clip after the GPU turbulence simulator."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Clean video file or directory with extracted RGB/BGR frame images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for clean_t.png, distorted_t.png, distorted_sequence.png, flow_u.png, flow_v.png, alpha.png.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--policy", default="k9_200ms")
    parser.add_argument(
        "--center-index",
        type=int,
        default=None,
        help="Center frame index. Defaults to the middle frame of the input.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Center crop side length before simulation. Use 0 to keep the full frame.",
    )
    parser.add_argument(
        "--severity",
        choices=("weak", "medium", "strong", "stress"),
        default="medium",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=0,
        help="Number of columns in sequence grids. Defaults to all frames in one row.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="OpenCV PNG compression level from 0 to 9.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(frame_dir: Path) -> list[Path]:
    return sorted(
        (path for path in frame_dir.iterdir() if is_image_file(path)),
        key=lambda path: path.name.lower(),
    )


def find_first_frame_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    if list_image_files(root):
        return root
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: str(p).lower()):
        if list_image_files(path):
            return path
    return None


def find_first_video(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    videos = sorted((path for path in root.rglob("*") if is_video_file(path)), key=lambda p: str(p).lower())
    return videos[0] if videos else None


def discover_default_input() -> Path:
    for root in (
        ROOT / "datasets" / "clean" / "train",
        ROOT / "datasets" / "clean" / "val",
    ):
        frame_dir = find_first_frame_dir(root)
        if frame_dir is not None:
            return frame_dir

    for root in (
        ROOT / "datasets" / "clean_videos" / "train",
        ROOT / "datasets" / "clean_videos" / "val",
    ):
        video = find_first_video(root)
        if video is not None:
            return video

    raise FileNotFoundError(
        "Could not find a clean clip. Pass --input or prepare datasets/clean(_videos)."
    )


def count_video_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            return frame_count

        frame_count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frame_count += 1
        if frame_count == 0:
            raise RuntimeError(f"No frames read from {video_path}")
        return frame_count
    finally:
        capture.release()


def read_image_as_tensor(path: Path) -> torch.Tensor:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).float() / 255.0
    return tensor.permute(2, 0, 1).contiguous()


def read_video_frames(video_path: Path, indices: Sequence[int]) -> torch.Tensor:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[torch.Tensor] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame).float() / 255.0
            frames.append(tensor.permute(2, 0, 1).contiguous())
    finally:
        capture.release()

    return torch.stack(frames, dim=0)


def load_clean_clip(
    input_path: Path,
    center_index: int | None,
    fps: float,
    policy: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    sampler = TemporalSampler(policy)

    if input_path.is_dir():
        image_paths = list_image_files(input_path)
        if not image_paths:
            raise RuntimeError(f"No frame images found in {input_path}")
        num_frames = len(image_paths)
        center = num_frames // 2 if center_index is None else int(center_index)
        sample = sampler.sample(center, fps=fps, num_frames=num_frames)
        clean_clip = torch.stack([read_image_as_tensor(image_paths[int(i)]) for i in sample.indices], dim=0)
        return clean_clip, sample.indices, sample.dt

    if is_video_file(input_path):
        num_frames = count_video_frames(input_path)
        center = num_frames // 2 if center_index is None else int(center_index)
        sample = sampler.sample(center, fps=fps, num_frames=num_frames)
        return read_video_frames(input_path, sample.indices), sample.indices, sample.dt

    raise ValueError(f"--input must be a video file or a frame directory: {input_path}")


def center_crop(frames: torch.Tensor, crop_size: int) -> torch.Tensor:
    if crop_size <= 0:
        return frames
    _, _, height, width = frames.shape
    side = min(int(crop_size), height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frames[:, :, top : top + side, left : left + side]


def rgb_tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def write_rgb_png(path: Path, image_rgb: np.ndarray, png_compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr, params):
        raise RuntimeError(f"Could not write image: {path}")


def make_rgb_grid(frames: torch.Tensor, cols: int = 0, pad: int = 4) -> np.ndarray:
    if frames.dim() != 4:
        raise ValueError(f"Expected frames [K,C,H,W], got {tuple(frames.shape)}")

    frame_images = [rgb_tensor_to_uint8(frame) for frame in frames]
    return tile_images(frame_images, cols=cols, pad=pad)


def tile_images(images: Sequence[np.ndarray], cols: int = 0, pad: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile")
    if cols <= 0:
        cols = len(images)

    rows = int(math.ceil(len(images) / cols))
    height, width, channels = images[0].shape
    canvas_h = rows * height + max(0, rows - 1) * pad
    canvas_w = cols * width + max(0, cols - 1) * pad
    canvas = np.full((canvas_h, canvas_w, channels), 255, dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        top = row * (height + pad)
        left = col * (width + pad)
        canvas[top : top + height, left : left + width] = image
    return canvas


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


def make_scalar_grid(values: torch.Tensor, cols: int = 0, symmetric: bool = False) -> np.ndarray:
    values = values.detach().cpu().float()
    if values.dim() != 3:
        raise ValueError(f"Expected scalar values [K,H,W], got {tuple(values.shape)}")

    if symmetric:
        bound = float(values.abs().max().item())
        bound = max(bound, 1e-6)
        vmin, vmax = -bound, bound
    else:
        vmin = float(values.min().item())
        vmax = float(values.max().item())
        if abs(vmax - vmin) < 1e-6:
            center = 0.5 * (vmin + vmax)
            vmin, vmax = center - 0.05, center + 0.05

    images = [scalar_to_colormap(frame, vmin=vmin, vmax=vmax) for frame in values]
    return tile_images(images, cols=cols)


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.center_index is not None and args.center_index < 0:
        raise ValueError("--center-index must be non-negative")
    if args.crop_size < 0:
        raise ValueError("--crop-size must be non-negative")
    if args.grid_cols < 0:
        raise ValueError("--grid-cols must be non-negative")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be between 0 and 9")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    validate_args(args)

    input_path = resolve_path(args.input) if args.input is not None else discover_default_input()
    out_dir = resolve_path(args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    clean_clip, frame_indices, dt = load_clean_clip(
        input_path=input_path,
        center_index=args.center_index,
        fps=args.fps,
        policy=args.policy,
    )
    clean_clip = center_crop(clean_clip, args.crop_size).to(device)
    timestamps = torch.tensor(frame_indices, device=device, dtype=torch.float32) / float(args.fps)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    simulator = GPUTurbulenceSimulator(TurbulenceConfig.from_severity(args.severity)).to(device).eval()
    distorted, meta = simulator(clean_clip, timestamps, return_meta=True, generator=generator)

    center_pos = int(np.argmin(np.abs(dt)))
    flow = meta["flow"].detach().cpu()
    alpha = meta["alpha"].detach().cpu()

    write_rgb_png(out_dir / "clean_t.png", rgb_tensor_to_uint8(clean_clip[center_pos]), args.png_compression)
    write_rgb_png(out_dir / "distorted_t.png", rgb_tensor_to_uint8(distorted[center_pos]), args.png_compression)
    write_rgb_png(out_dir / "distorted_sequence.png", make_rgb_grid(distorted, cols=args.grid_cols), args.png_compression)
    write_rgb_png(out_dir / "flow_u.png", make_scalar_grid(flow[:, 0], cols=args.grid_cols, symmetric=True), args.png_compression)
    write_rgb_png(out_dir / "flow_v.png", make_scalar_grid(flow[:, 1], cols=args.grid_cols, symmetric=True), args.png_compression)
    write_rgb_png(out_dir / "alpha.png", make_scalar_grid(alpha[:, 0], cols=args.grid_cols), args.png_compression)

    print(f"input: {input_path}")
    print(f"frames: {frame_indices.tolist()}")
    print(f"clip: {tuple(clean_clip.shape)} device={device}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
