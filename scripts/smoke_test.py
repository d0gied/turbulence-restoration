
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turbulence_restoration.data.temporal_sampler import TemporalSampler
from turbulence_restoration.experiments.metrics import psnr, ssim
from turbulence_restoration.inference.tiled_inference import infer_frame_tiled
from turbulence_restoration.models import TimeAwareGeoLuckyRestorer
from turbulence_restoration.simulator.gpu_turbulence import GPUTurbulenceSimulator, TurbulenceConfig
from turbulence_restoration.training.losses import RestorationLoss


def make_clean_video(num_frames: int = 6, H: int = 16, W: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    yy = yy.float() / max(H - 1, 1)
    xx = xx.float() / max(W - 1, 1)
    base = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0)
    frames = []
    for t in range(num_frames):
        frame = base.clone()
        cx = int((0.2 + 0.6 * t / max(num_frames - 1, 1)) * W)
        cy = H // 2
        size = 2
        frame[:, max(0, cy - size):min(H, cy + size), max(0, cx - size):min(W, cx + size)] = torch.tensor([1.0, 0.2, 0.1])[:, None, None]
        frames.append(frame.clamp(0, 1))
    return torch.stack(frames, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy", default="k1_center", help="Use k1_center for quick CPU smoke; k9_200ms for full GPU smoke.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    fps = 60.0
    clean_video = make_clean_video().to(device)
    center_idx = 3

    sampler = TemporalSampler(args.policy)
    sample = sampler.sample(center_idx, fps=fps, num_frames=clean_video.shape[0])
    clean_window = clean_video[sample.indices]
    timestamps = torch.tensor(sample.indices, device=device, dtype=torch.float32) / fps

    cfg = TurbulenceConfig.from_severity("weak")
    cfg.octaves = 1
    cfg.spatial_scales = (16.0, 8.0, 4.0, 2.0)
    cfg.octave_weights = (1.0, 0.5, 0.25, 0.125)
    cfg.blur_sigma = (0.0, 0.5)
    cfg.scintillation_strength = (0.0, 0.01)

    sim = GPUTurbulenceSimulator(cfg).to(device)
    distorted, meta = sim(clean_window, timestamps, return_meta=True)

    frames = distorted.unsqueeze(0)
    target = clean_video[center_idx].unsqueeze(0)
    dt = torch.tensor(sample.dt, device=device).unsqueeze(0)
    valid = torch.tensor(sample.valid, device=device).unsqueeze(0)

    model = TimeAwareGeoLuckyRestorer(c0=1, c1=2, c2=4).to(device).eval()
    with torch.no_grad():
        pred, aux = model(frames, dt, valid, return_aux=True)
        loss_fn = RestorationLoss(edge_weight=0.05).to(device)
        loss, _ = loss_fn(pred, target)

    # Tiled check is useful but very slow on CPU in some environments.
    if device.type == "cuda":
        tiled = infer_frame_tiled(
            model,
            distorted,
            torch.tensor(sample.dt, device=device),
            torch.tensor(sample.valid, device=device),
            tile=8,
            halo=2,
            overlap=2,
            amp=False,
        )
        full_vs_tiled = float(torch.mean(torch.abs(pred[0] - tiled)).cpu())
    else:
        full_vs_tiled = None

    print("device:", device)
    print("policy:", args.policy)
    print("distorted:", tuple(distorted.shape), "flow:", tuple(meta["flow"].shape))
    print("pred:", tuple(pred.shape), "loss:", float(loss.detach().cpu()))
    print("psnr:", float(psnr(pred, target).mean().cpu()), "ssim:", float(ssim(pred, target).mean().cpu()))
    print("full_vs_tiled_l1:", full_vs_tiled)
    print("OK")


if __name__ == "__main__":
    main()
