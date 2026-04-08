from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turbulence_restoration.experiments.metrics import psnr, ssim
from turbulence_restoration.models import TimeAwareGeoLuckyRestorer
from turbulence_restoration.simulator.gpu_turbulence import GPUTurbulenceSimulator, TurbulenceConfig
from turbulence_restoration.training.train import build_train_val_datasets, load_yaml, simulate_batch, to_device_batch
from turbulence_restoration.training.validation_visualization import ValidationVisualizer


DEFAULT_OUT_DIR = ROOT / "results" / "validation_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the model on validation data and save qualitative visualizations."
    )
    parser.add_argument("--config", required=True, help="Training config used to build the dataset and model.")
    parser.add_argument("--weights", required=True, help="Checkpoint path with model weights.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=8, help="How many validation samples to visualize.")
    parser.add_argument("--batch-size", type=int, default=1, help="Validation loader batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers from config.")
    parser.add_argument("--split-seed", type=int, default=123, help="Seed for train/val split.")
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=0,
        help="Columns for per-sample input sequence grids. Defaults to all frames in one row.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="OpenCV PNG compression level from 0 to 9.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.grid_cols < 0:
        raise ValueError("--grid-cols must be non-negative")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be between 0 and 9")


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def build_model(cfg: Dict[str, Any], device: torch.device) -> TimeAwareGeoLuckyRestorer:
    model_cfg = cfg.get("model", {})
    return TimeAwareGeoLuckyRestorer(
        c0=int(model_cfg.get("c0", 32)),
        c1=int(model_cfg.get("c1", 64)),
        c2=int(model_cfg.get("c2", 128)),
        use_time=bool(model_cfg.get("use_time", True)),
    ).to(device)


def build_validation_loader(
    cfg: Dict[str, Any],
    device: torch.device,
    split_seed: int,
    batch_size: int,
    num_workers: int | None,
) -> tuple[DataLoader, int, GPUTurbulenceSimulator]:
    sim_cfg = TurbulenceConfig.from_severity(cfg.get("simulator", {}).get("severity", "medium"))
    simulator = GPUTurbulenceSimulator(sim_cfg).to(device)
    _, val_ds = build_train_val_datasets(cfg, simulator, split_seed=int(split_seed))

    loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0) if num_workers is None else num_workers),
        pin_memory=device.type == "cuda",
    )
    return loader, len(val_ds), simulator


@torch.no_grad()
def main() -> None:
    args = parse_args()
    validate_args(args)

    cfg = load_yaml(args.config)
    out_dir = resolve_path(args.out_dir)
    weights_path = resolve_path(args.weights)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = build_model(cfg, device).eval()
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    val_loader, val_size, simulator = build_validation_loader(
        cfg=cfg,
        device=device,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    visualizer = ValidationVisualizer(num_samples=int(args.num_samples), grid_cols=int(args.grid_cols))

    for batch in val_loader:
        batch = to_device_batch(batch, device)
        batch = simulate_batch(batch, simulator)
        pred = model(batch["frames"], batch["dt"], batch["valid"])

        batch_psnr = psnr(pred, batch["target"]).detach().cpu()
        batch_ssim = ssim(pred, batch["target"]).detach().cpu()

        visualizer.add_batch(batch, pred, batch_psnr, batch_ssim)
        if visualizer.is_full():
            break

    metrics_payload = visualizer.write_artifacts(
        out_dir,
        png_compression=int(args.png_compression),
        extra={
            "config": str(resolve_path(args.config)),
            "weights": str(weights_path),
            "device": str(device),
            "split_seed": int(args.split_seed),
            "val_size": int(val_size),
        },
    )

    print(f"weights: {weights_path}")
    print(f"device: {device}")
    print(f"val size: {val_size}")
    print(f"visualized: {len(visualizer.samples)}")
    print(f"mean psnr: {metrics_payload['mean']['psnr']:.4f}")
    print(f"mean ssim: {metrics_payload['mean']['ssim']:.4f}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
