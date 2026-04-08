
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from turbulence_restoration.data.dataset import RealVideoFramesDataset, SyntheticMovingShapesDataset
from turbulence_restoration.experiments.metrics import psnr, ssim
from turbulence_restoration.models import TimeAwareGeoLuckyRestorer
from turbulence_restoration.simulator.gpu_turbulence import GPUTurbulenceSimulator, TurbulenceConfig
from turbulence_restoration.training.losses import (
    RestorationLoss,
    alignment_loss,
    flow_smoothness_loss,
    oracle_weight_loss,
)
from turbulence_restoration.training.validation_visualization import ValidationVisualizer


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


def simulate_batch(batch: Dict[str, Any], simulator) -> Dict[str, Any]:
    if "frames" in batch:
        return batch
    clean_frames = batch["clean_frames"] if "clean_frames" in batch else batch["frames"]
    timestamps = batch["timestamps"]
    batch["frames"] = simulator(clean_frames, timestamps, return_meta=False).float()
    return batch


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


def build_dataset(cfg, simulator, train: bool = True):
    data_cfg = cfg.get("data", {})
    simulate_on_device = bool(cfg.get("simulator", {}).get("on_device", data_cfg.get("simulate_on_device", False)))
    dataset_simulator = None if simulate_on_device else simulator
    include_frames = not simulate_on_device
    dataset_root = data_cfg.get("train_root" if train else "val_root")
    if dataset_root is None:
        dataset_root = data_cfg.get("root")

    if dataset_root:
        cache_key = "train_cache" if train else "val_cache"
        cache_path = data_cfg.get(cache_key, data_cfg.get("cache_path"))
        return RealVideoFramesDataset(
            root=dataset_root,
            simulator=dataset_simulator,
            fps=float(data_cfg.get("fps", 60.0)),
            crop_size=int(data_cfg.get("crop_size", data_cfg.get("image_size", 256))),
            temporal_policy=data_cfg.get("temporal_policy", "k9_200ms"),
            train=train,
            include_frames=include_frames,
            preload=bool(data_cfg.get("preload", False)),
            cache_path=cache_path,
        )

    return SyntheticMovingShapesDataset(
        num_samples=int(data_cfg.get("num_samples", 256)),
        num_clean_frames=int(data_cfg.get("num_clean_frames", 40)),
        image_size=int(data_cfg.get("image_size", 128)),
        fps=float(data_cfg.get("fps", 60.0)),
        simulator=dataset_simulator,
        temporal_policy=data_cfg.get("temporal_policy", "k9_200ms"),
        include_frames=include_frames,
    )


def build_train_val_datasets(cfg, simulator, split_seed: int = 123):
    data_cfg = cfg.get("data", {})
    train_dataset = build_dataset(cfg, simulator, train=True)

    if data_cfg.get("val_root"):
        val_dataset = build_dataset(cfg, simulator, train=False)
        return train_dataset, val_dataset

    val_dataset = build_dataset(cfg, simulator, train=False)
    n_val = max(1, int(0.1 * len(train_dataset)))
    n_train = len(train_dataset) - n_val
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(int(split_seed))).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    return (
        torch.utils.data.Subset(train_dataset, train_indices),
        torch.utils.data.Subset(val_dataset, val_indices),
    )


def maybe_add_scalars(writer, prefix: str, values: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in values.items():
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def create_summary_writer(cfg: Dict[str, Any], out_dir: Path):
    tb_cfg = cfg.get("tensorboard", {})
    if not bool(tb_cfg.get("enabled", True)):
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging is enabled, but the `tensorboard` package is not installed."
        ) from exc

    log_dir = tb_cfg.get("log_dir")
    if log_dir is None:
        log_dir = out_dir / "tensorboard"
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir), flush_secs=int(tb_cfg.get("flush_secs", 30)))


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    simulator,
    device,
    stage: str,
    scaler=None,
    weights=None,
    writer=None,
    global_step: int = 0,
    step_log_interval: int = 10,
):
    model.train()
    logs = []
    weights = weights or {}
    pbar = tqdm(loader, desc=f"train/{stage}")
    for batch in pbar:
        batch = to_device_batch(batch, device)
        batch = simulate_batch(batch, simulator)
        frames, target, dt, valid = batch["frames"], batch["target"], batch["dt"], batch["valid"]

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp):
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
        if writer is not None and step_log_interval > 0 and global_step % step_log_interval == 0:
            maybe_add_scalars(writer, "train_step", item, global_step)
        pbar.set_postfix(loss=f"{item['loss']:.4f}", psnr=f"{item['psnr']:.2f}")
        global_step += 1

    return {k: sum(d[k] for d in logs) / max(len(logs), 1) for k in logs[0]}, global_step


@torch.no_grad()
def validate(model, loader, loss_fn, simulator, device, visualizer: ValidationVisualizer | None = None):
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    for batch in tqdm(loader, desc="val"):
        batch = to_device_batch(batch, device)
        batch = simulate_batch(batch, simulator)
        pred = model(batch["frames"], batch["dt"], batch["valid"])
        loss, _ = loss_fn(pred, batch["target"])
        batch_psnr = psnr(pred, batch["target"]).detach().cpu()
        batch_ssim = ssim(pred, batch["target"]).detach().cpu()
        batch_size = pred.shape[0]

        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["psnr"] += float(batch_psnr.sum().item())
        totals["ssim"] += float(batch_ssim.sum().item())
        count += batch_size

        if visualizer is not None and not visualizer.is_full():
            visualizer.add_batch(batch, pred, batch_psnr, batch_ssim)

    if count == 0:
        raise RuntimeError("Validation loader produced no samples")

    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split-seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    sim_cfg = TurbulenceConfig.from_severity(cfg.get("simulator", {}).get("severity", "medium"))
    simulator = GPUTurbulenceSimulator(sim_cfg).to(device)

    train_ds, val_ds = build_train_val_datasets(cfg, simulator, split_seed=args.split_seed)

    num_workers = int(cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        **loader_kwargs,
    )

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
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    out_dir = Path(cfg.get("out_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(cfg, out_dir)
    global_step = 0
    tb_cfg = cfg.get("tensorboard", {})
    val_tb_cfg = tb_cfg.get("validation", {})
    val_viz_every = max(1, int(val_tb_cfg.get("every_n_epochs", 1)))

    best_psnr = -1e9
    try:
        for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
            train_log, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                simulator,
                device,
                stage,
                scaler,
                cfg.get("aux_weights", {}),
                writer=writer,
                global_step=global_step,
                step_log_interval=int(tb_cfg.get("train_step_interval", 10)),
            )

            collect_val_visuals = bool(val_tb_cfg.get("enabled", True)) and epoch % val_viz_every == 0
            visualizer = None
            if collect_val_visuals:
                visualizer = ValidationVisualizer(
                    num_samples=int(val_tb_cfg.get("num_samples", 8)),
                    grid_cols=int(val_tb_cfg.get("grid_cols", 0)),
                )

            val_log = validate(model, val_loader, loss_fn, simulator, device, visualizer=visualizer)
            print({"epoch": epoch, "train": train_log, "val": val_log})

            maybe_add_scalars(writer, "train", train_log, epoch)
            maybe_add_scalars(writer, "val", val_log, epoch)
            if writer is not None:
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)

            if visualizer is not None and writer is not None and visualizer.samples:
                writer.add_image("val/summary", visualizer.summary_image(), epoch, dataformats="HWC")
                for idx, image in enumerate(visualizer.sequence_images):
                    writer.add_image(f"val/sample_{idx:02d}_sequence", image, epoch, dataformats="HWC")

            ckpt = {"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "val": val_log}
            torch.save(ckpt, out_dir / "last.pt")
            if val_log["psnr"] > best_psnr:
                best_psnr = val_log["psnr"]
                torch.save(ckpt, out_dir / "best.pt")

            if visualizer is not None and bool(val_tb_cfg.get("save_to_disk", True)) and visualizer.samples:
                visualizer.write_artifacts(
                    out_dir / "validation" / f"epoch_{epoch:04d}",
                    png_compression=int(val_tb_cfg.get("png_compression", 3)),
                    extra={
                        "epoch": int(epoch),
                        "config": str(Path(args.config).resolve()),
                        "weights": str((out_dir / "last.pt").resolve()),
                        "device": str(device),
                        "split_seed": int(args.split_seed),
                        "val_size": int(len(val_ds)),
                    },
                )
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
