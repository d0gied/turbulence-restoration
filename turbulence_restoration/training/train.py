
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from turbulence_restoration.data.dataset import SyntheticMovingShapesDataset
from turbulence_restoration.experiments.metrics import psnr, ssim
from turbulence_restoration.models import TimeAwareGeoLuckyRestorer
from turbulence_restoration.simulator.gpu_turbulence import GPUTurbulenceSimulator, TurbulenceConfig
from turbulence_restoration.training.losses import (
    RestorationLoss,
    alignment_loss,
    flow_smoothness_loss,
    oracle_weight_loss,
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


def configure_stage(model, stage: str):
    unfreeze(model)
    if stage == "fusion":
        freeze(model.encoder)
        freeze(model.aligner)
        unfreeze(model.fusion)
        unfreeze(model.decoder)
    elif stage in {"single", "align", "e2e"}:
        pass
    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_dataset(cfg, simulator):
    data_cfg = cfg.get("data", {})
    return SyntheticMovingShapesDataset(
        num_samples=int(data_cfg.get("num_samples", 256)),
        num_clean_frames=int(data_cfg.get("num_clean_frames", 40)),
        image_size=int(data_cfg.get("image_size", 128)),
        fps=float(data_cfg.get("fps", 60.0)),
        simulator=simulator,
        temporal_policy=data_cfg.get("temporal_policy", "k9_200ms"),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device, stage: str, scaler=None, weights=None):
    model.train()
    logs = []
    weights = weights or {}
    pbar = tqdm(loader, desc=f"train/{stage}")
    for batch in pbar:
        batch = to_device_batch(batch, device)
        frames, target, dt, valid = batch["frames"], batch["target"], batch["dt"], batch["valid"]

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, aux = model(frames, dt, valid, return_aux=True)
            rec_loss, rec_logs = loss_fn(pred, target)
            loss = rec_loss

            if stage == "align" or float(weights.get("align", 0.0)) > 0:
                la = alignment_loss(frames, target, aux["flow0"])
                flow_flat = aux["flow0"].reshape(-1, 2, aux["flow0"].shape[-2], aux["flow0"].shape[-1])
                ls = flow_smoothness_loss(flow_flat)
                loss = loss + float(weights.get("align", 1.0 if stage == "align" else 0.0)) * la
                loss = loss + float(weights.get("smooth", 0.01)) * ls
            else:
                la = torch.zeros((), device=device)
                ls = torch.zeros((), device=device)

            if stage == "fusion" or float(weights.get("oracle", 0.0)) > 0:
                lw = oracle_weight_loss(frames, target, aux["flow0"], aux["weights0"])
                loss = loss + float(weights.get("oracle", 0.05)) * lw
            else:
                lw = torch.zeros((), device=device)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        item = {
            "loss": float(loss.detach().cpu()),
            "pix": float(rec_logs["pix"].cpu()),
            "edge": float(rec_logs["edge"].cpu()),
            "align": float(la.detach().cpu()),
            "smooth": float(ls.detach().cpu()),
            "oracle": float(lw.detach().cpu()),
            "psnr": float(psnr(pred.detach(), target).mean().detach().cpu()),
        }
        logs.append(item)
        pbar.set_postfix(loss=f"{item['loss']:.4f}", psnr=f"{item['psnr']:.2f}")

    return {k: sum(d[k] for d in logs) / max(len(logs), 1) for k in logs[0]}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    logs = []
    for batch in tqdm(loader, desc="val"):
        batch = to_device_batch(batch, device)
        pred = model(batch["frames"], batch["dt"], batch["valid"])
        loss, _ = loss_fn(pred, batch["target"])
        logs.append({
            "loss": float(loss.cpu()),
            "psnr": float(psnr(pred, batch["target"]).mean().cpu()),
            "ssim": float(ssim(pred, batch["target"]).mean().cpu()),
        })
    return {k: sum(d[k] for d in logs) / max(len(logs), 1) for k in logs[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    sim_cfg = TurbulenceConfig.from_severity(cfg.get("simulator", {}).get("severity", "medium"))
    simulator = GPUTurbulenceSimulator(sim_cfg).to(device)

    dataset = build_dataset(cfg, simulator)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(123))

    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True,
                              num_workers=int(cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=False,
                            num_workers=int(cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")

    model_cfg = cfg.get("model", {})
    model = TimeAwareGeoLuckyRestorer(
        c0=int(model_cfg.get("c0", 32)),
        c1=int(model_cfg.get("c1", 64)),
        c2=int(model_cfg.get("c2", 128)),
        use_time=bool(model_cfg.get("use_time", True)),
    ).to(device)

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    stage = cfg.get("stage", "e2e")
    configure_stage(model, stage)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    loss_fn = RestorationLoss(edge_weight=float(cfg.get("loss", {}).get("edge", 0.05))).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    out_dir = Path(cfg.get("out_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -1e9
    for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, device, stage, scaler, cfg.get("aux_weights", {}))
        val_log = validate(model, val_loader, loss_fn, device)
        print({"epoch": epoch, "train": train_log, "val": val_log})

        ckpt = {"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "val": val_log}
        torch.save(ckpt, out_dir / "last.pt")
        if val_log["psnr"] > best_psnr:
            best_psnr = val_log["psnr"]
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
