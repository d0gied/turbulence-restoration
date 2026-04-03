
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from turbulence_restoration.inference.temporal_policy import InferenceTemporalPolicy
from turbulence_restoration.inference.tiled_inference import infer_frame_tiled
from turbulence_restoration.models import TimeAwareGeoLuckyRestorer
from turbulence_restoration.utils.io import read_video_tensor, write_video_tensor


@torch.no_grad()
def infer_video(model, video: torch.Tensor, fps: float, temporal_policy: str = "default_200ms",
                tile: int = 512, halo: int = 128, overlap: int = 128, amp: bool = True) -> torch.Tensor:
    policy = InferenceTemporalPolicy(temporal_policy)
    outputs = []
    N = video.shape[0]
    for t in tqdm(range(N), desc="infer"):
        indices, dt, valid = policy.get_indices(t, fps, N)
        frames = video[indices]
        pred = infer_frame_tiled(
            model,
            frames,
            torch.tensor(dt, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32),
            tile=tile,
            halo=halo,
            overlap=overlap,
            amp=amp,
        )
        outputs.append(pred.cpu())
    return torch.stack(outputs, dim=0)


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
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = TimeAwareGeoLuckyRestorer().to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)

    video = read_video_tensor(args.input)
    output = infer_video(model, video, args.fps, args.temporal_policy, args.tile, args.halo, args.overlap, not args.no_amp)
    write_video_tensor(output, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
