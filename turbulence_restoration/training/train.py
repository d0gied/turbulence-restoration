
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from turbulence_restoration.data.dataset import RealVideoFramesDataset, SyntheticMovingShapesDataset
from turbulence_restoration.experiments.metrics import (
    psnr,
    ssim,
    temporal_acceleration_error,
    temporal_delta_error,
)
from turbulence_restoration.models import RecurrentGeoLuckyRestorer, TimeAwareGeoLuckyRestorer
from turbulence_restoration.simulator.gpu_turbulence import GPUTurbulenceSimulator, TurbulenceConfig
from turbulence_restoration.training.losses import (
    RestorationLoss,
    alignment_loss,
    flow_smoothness_loss,
    fusion_weight_smoothness_loss,
    oracle_weight_loss,
    residual_acceleration_loss,
    temporal_acceleration_loss,
    temporal_velocity_loss,
)
from turbulence_restoration.training.validation_visualization import ValidationVisualizer

TEMPORAL_STAGES = {"temporal", "temporal_finetune", "stage5_temporal"}
RECURRENT_STAGES = {"recurrent", "recurrent_head", "recurrent_finetune", "stage6_recurrent"}
SEQUENCE_STAGES = TEMPORAL_STAGES | RECURRENT_STAGES


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


def _probability(value: Any, name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}")
    return probability


def build_train_simulation_mix(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    mix_cfg = cfg.get("simulator", {}).get("train_mix", {})
    if not mix_cfg:
        return {}

    clean_probability = _probability(
        mix_cfg.get("clean_probability", mix_cfg.get("stable_probability", 0.0)),
        "simulator.train_mix.clean_probability",
    )
    low_probability = _probability(
        mix_cfg.get("low_probability", 0.0),
        "simulator.train_mix.low_probability",
    )
    if clean_probability + low_probability > 1.0:
        raise ValueError("simulator.train_mix clean_probability + low_probability must be <= 1")

    granularity = str(mix_cfg.get("granularity", "frame"))
    if granularity not in {"frame", "clip"}:
        raise ValueError("simulator.train_mix.granularity must be 'frame' or 'clip'")

    low_simulator = None
    if low_probability > 0.0:
        low_severity = str(mix_cfg.get("low_severity", "very_weak"))
        low_simulator = GPUTurbulenceSimulator(TurbulenceConfig.from_severity(low_severity)).to(device)

    return {
        "clean_probability": clean_probability,
        "low_probability": low_probability,
        "granularity": granularity,
        "low_simulator": low_simulator,
    }


def simulate_frames(clean_frames: torch.Tensor, timestamps: torch.Tensor, simulator, mix: Dict[str, Any] | None = None):
    distorted = simulator(clean_frames, timestamps, return_meta=False).float()
    if not mix:
        return distorted

    clean_probability = float(mix.get("clean_probability", 0.0))
    low_probability = float(mix.get("low_probability", 0.0))
    if clean_probability <= 0.0 and low_probability <= 0.0:
        return distorted

    if clean_frames.dim() != 5:
        raise ValueError(f"simulate_frames expects [B,K,C,H,W], got {tuple(clean_frames.shape)}")

    B, K = clean_frames.shape[:2]
    if mix.get("granularity", "frame") == "clip":
        decision_shape = (B, 1, 1, 1, 1)
    else:
        decision_shape = (B, K, 1, 1, 1)
    selector = torch.rand(decision_shape, device=clean_frames.device, dtype=clean_frames.dtype)

    clean_mask = selector < clean_probability
    low_mask = (selector >= clean_probability) & (selector < clean_probability + low_probability)
    if low_probability > 0.0 and bool(low_mask.any()):
        low_simulator = mix.get("low_simulator")
        if low_simulator is None:
            raise ValueError("simulator.train_mix.low_probability requires a low_simulator")
        low_distorted = low_simulator(clean_frames, timestamps, return_meta=False).float()
        distorted = torch.where(low_mask, low_distorted, distorted)

    if clean_probability > 0.0 and bool(clean_mask.any()):
        distorted = torch.where(clean_mask, clean_frames.float(), distorted)
    return distorted


def simulate_batch(batch: Dict[str, Any], simulator, mix: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if "frames_seq" in batch:
        return batch
    if "clean_clip" in batch:
        clean_clip = batch["clean_clip"]
        timestamps = batch["clip_timestamps"]
        distorted_clip = simulate_frames(clean_clip, timestamps, simulator, mix)
        positions = batch["window_positions"].to(device=distorted_clip.device, dtype=torch.long)
        B, U, C, H, W = distorted_clip.shape
        S, K = positions.shape[1], positions.shape[2]
        expanded = distorted_clip[:, None].expand(B, S, U, C, H, W)
        gather_idx = positions[..., None, None, None].expand(B, S, K, C, H, W)
        batch["frames_seq"] = torch.gather(expanded, dim=2, index=gather_idx)
        return batch
    if "frames" in batch:
        return batch
    clean_frames = batch["clean_frames"] if "clean_frames" in batch else batch["frames"]
    timestamps = batch["timestamps"]
    batch["frames"] = simulate_frames(clean_frames, timestamps, simulator, mix)
    return batch


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


def _set_trainable(module, trainable: bool):
    if module is None:
        return
    if trainable:
        unfreeze(module)
    else:
        freeze(module)


def apply_freeze_config(model, freeze_cfg: Dict[str, Any]):
    groups = {
        "encoder": [getattr(model, "encoder", None)],
        "aligner": [getattr(model, "aligner", None)],
        "fusion": [getattr(model, "fusion", None)],
        "decoder": [getattr(model, "decoder", None)],
        "recurrent": [getattr(model, "gru2", None), getattr(model, "merge2", None)],
    }
    for name, should_freeze in freeze_cfg.items():
        if name not in groups:
            raise ValueError(f"Unknown freeze group: {name}. Available: {list(groups)}")
        for module in groups[name]:
            _set_trainable(module, not bool(should_freeze))


def configure_stage(model, stage: str, freeze_cfg: Dict[str, Any] | None = None):
    unfreeze(model)
    if stage == "fusion":
        freeze(model.encoder)
        freeze(model.aligner)
        unfreeze(model.fusion)
        unfreeze(model.decoder)
    elif stage in RECURRENT_STAGES:
        freeze(model.encoder)
        freeze(model.aligner)
        unfreeze(model.fusion)
        unfreeze(model.decoder)
        if hasattr(model, "gru2"):
            unfreeze(model.gru2)
        if hasattr(model, "merge2"):
            unfreeze(model.merge2)
    elif stage in {"single", "align", "e2e"} | TEMPORAL_STAGES:
        pass
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if freeze_cfg:
        apply_freeze_config(model, freeze_cfg)


def build_dataset(cfg, simulator, train: bool = True):
    data_cfg = cfg.get("data", {})
    sequence_cfg = cfg.get("sequence", {})
    output_frames = int(sequence_cfg.get("output_frames", data_cfg.get("output_frames", 1)))
    center_stride = int(sequence_cfg.get("center_stride", data_cfg.get("center_stride", 1)))
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
            cache_lru_size=int(data_cfg.get("cache_lru_size", 1)),
            output_frames=output_frames,
            center_stride=center_stride,
        )

    return SyntheticMovingShapesDataset(
        num_samples=int(data_cfg.get("num_samples", 256)),
        num_clean_frames=int(data_cfg.get("num_clean_frames", 40)),
        image_size=int(data_cfg.get("image_size", 128)),
        fps=float(data_cfg.get("fps", 60.0)),
        simulator=dataset_simulator,
        temporal_policy=data_cfg.get("temporal_policy", "k9_200ms"),
        include_frames=include_frames,
        output_frames=output_frames,
        center_stride=center_stride,
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


def tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


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


def build_model_from_config(cfg: Dict[str, Any], device: torch.device):
    model_cfg = cfg.get("model", {})
    stage = str(cfg.get("stage", "e2e"))
    model_type = str(model_cfg.get("type", "recurrent" if stage in RECURRENT_STAGES else "time_aware"))
    c0 = int(model_cfg.get("c0", 32))
    c1 = int(model_cfg.get("c1", 64))
    c2 = int(model_cfg.get("c2", 128))
    use_time = bool(model_cfg.get("use_time", True))

    base_model = TimeAwareGeoLuckyRestorer(c0=c0, c1=c1, c2=c2, use_time=use_time)
    if model_type in {"time_aware", "base", "geo_lucky"}:
        return base_model.to(device)
    if model_type in {"recurrent", "recurrent_geo_lucky"}:
        return RecurrentGeoLuckyRestorer(
            base_model,
            bottleneck_channels=int(model_cfg.get("bottleneck_channels", c2)),
            hidden_damping=float(model_cfg.get("hidden_damping", 1.0)),
        ).to(device)
    raise ValueError(f"Unknown model.type: {model_type}")


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
    loss_weights=None,
    writer=None,
    global_step: int = 0,
    step_log_interval: int = 10,
    simulation_mix=None,
):
    model.train()
    logs = []
    weights = weights or {}
    loss_weights = loss_weights or {}
    rec_weight = float(loss_weights.get("rec", 1.0))
    align_weight = float(weights.get("align", 1.0 if stage == "align" else 0.0))
    smooth_weight = float(weights.get("smooth", 0.01))
    oracle_weight = float(weights.get("oracle", 0.05))
    temporal_velocity_weight = float(loss_weights.get("temporal_velocity", 0.05 if stage in SEQUENCE_STAGES else 0.0))
    temporal_acceleration_weight = float(
        loss_weights.get("temporal_acceleration", 0.10 if stage in SEQUENCE_STAGES else 0.0)
    )
    residual_acceleration_weight = float(
        loss_weights.get("residual_acceleration", 0.05 if stage in SEQUENCE_STAGES else 0.0)
    )
    weight_smoothness_weight = float(
        loss_weights.get("weight_smoothness", 0.005 if stage in SEQUENCE_STAGES else 0.0)
    )
    temporal_lowpass_scale = int(loss_weights.get("temporal_lowpass_scale", 2))
    pbar = tqdm(loader, desc=f"train/{stage}")
    for batch in pbar:
        batch = to_device_batch(batch, device)
        batch = simulate_batch(batch, simulator, simulation_mix)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp):
            if "frames_seq" in batch:
                frames_seq = batch["frames_seq"]
                target_seq = batch["target_seq"]
                dt_seq = batch["dt_seq"]
                valid_seq = batch["valid_seq"]
                B, S, K, C, H, W = frames_seq.shape

                target = target_seq.reshape(B * S, C, H, W)
                if getattr(model, "supports_sequence", False):
                    pred_seq, aux = model(frames_seq, dt_seq, valid_seq, return_aux=True)
                    pred = pred_seq.reshape(B * S, C, H, W)
                else:
                    frames = frames_seq.reshape(B * S, K, C, H, W)
                    dt = dt_seq.reshape(B * S, K)
                    valid = valid_seq.reshape(B * S, K)
                    pred, aux = model(frames, dt, valid, return_aux=True)
                    pred_seq = pred.reshape(B, S, C, H, W)
                rec_loss, rec_logs = loss_fn(pred, target)
                loss = rec_weight * rec_loss

                input_center_seq = frames_seq[:, :, K // 2]
                loss_vel = temporal_velocity_loss(pred_seq, target_seq, temporal_lowpass_scale)
                loss_acc = temporal_acceleration_loss(pred_seq, target_seq, temporal_lowpass_scale)
                loss_res = residual_acceleration_loss(pred_seq, input_center_seq, temporal_lowpass_scale)
                weights_seq = aux["weights0"].reshape(B, S, K, 1, H, W)
                loss_weight_smooth = fusion_weight_smoothness_loss(weights_seq)
                loss = loss + temporal_velocity_weight * loss_vel
                loss = loss + temporal_acceleration_weight * loss_acc
                loss = loss + residual_acceleration_weight * loss_res
                loss = loss + weight_smoothness_weight * loss_weight_smooth

                la = torch.zeros((), device=device)
                ls = torch.zeros((), device=device)
                lw = torch.zeros((), device=device)
                tde = temporal_delta_error(pred_seq.detach(), target_seq).detach()
                tae = temporal_acceleration_error(pred_seq.detach(), target_seq).detach()
            else:
                frames, target, dt, valid = batch["frames"], batch["target"], batch["dt"], batch["valid"]
                pred, aux = model(frames, dt, valid, return_aux=True)
                rec_loss, rec_logs = loss_fn(pred, target)
                loss = rec_weight * rec_loss

                if stage == "align" or float(weights.get("align", 0.0)) > 0:
                    la = alignment_loss(frames, target, aux["flow0"])
                    flow_flat = aux["flow0"].reshape(-1, 2, aux["flow0"].shape[-2], aux["flow0"].shape[-1])
                    ls = flow_smoothness_loss(flow_flat)
                    loss = loss + align_weight * la
                    loss = loss + smooth_weight * ls
                else:
                    la = torch.zeros((), device=device)
                    ls = torch.zeros((), device=device)

                if stage == "fusion" or float(weights.get("oracle", 0.0)) > 0:
                    lw = oracle_weight_loss(frames, target, aux["flow0"], aux["weights0"])
                    loss = loss + oracle_weight * lw
                else:
                    lw = torch.zeros((), device=device)

                loss_vel = torch.zeros((), device=device)
                loss_acc = torch.zeros((), device=device)
                loss_res = torch.zeros((), device=device)
                loss_weight_smooth = torch.zeros((), device=device)
                tde = torch.zeros((), device=device)
                tae = torch.zeros((), device=device)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        item = {
            "loss": tensor_float(loss),
            "pix": tensor_float(rec_logs["pix"]),
            "edge": tensor_float(rec_logs["edge"]),
            "align": tensor_float(la),
            "smooth": tensor_float(ls),
            "oracle": tensor_float(lw),
            "temporal_velocity": tensor_float(loss_vel),
            "temporal_acceleration": tensor_float(loss_acc),
            "res_acc": tensor_float(loss_res),
            "weight_smooth": tensor_float(loss_weight_smooth),
            "tde": tensor_float(tde),
            "tae": tensor_float(tae),
            "psnr": float(psnr(pred.detach(), target).mean().detach().cpu()),
            "grad_norm": tensor_float(grad_norm),
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
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "error_mean": 0.0, "error_p95": 0.0, "error_max": 0.0}
    count = 0
    temporal_totals = {"tde": 0.0, "tae": 0.0}
    temporal_count = 0
    for batch in tqdm(loader, desc="val"):
        batch = to_device_batch(batch, device)
        batch = simulate_batch(batch, simulator)
        if "frames_seq" in batch:
            frames_seq = batch["frames_seq"]
            target_seq = batch["target_seq"]
            dt_seq = batch["dt_seq"]
            valid_seq = batch["valid_seq"]
            B, S, K, C, H, W = frames_seq.shape
            target = target_seq.reshape(B * S, C, H, W)

            if getattr(model, "supports_sequence", False):
                if visualizer is None:
                    pred_seq = model(frames_seq, dt_seq, valid_seq)
                    aux = None
                else:
                    pred_seq, aux = model(frames_seq, dt_seq, valid_seq, return_aux=True)
                pred = pred_seq.reshape(B * S, C, H, W)
            else:
                frames = frames_seq.reshape(B * S, K, C, H, W)
                dt = dt_seq.reshape(B * S, K)
                valid = valid_seq.reshape(B * S, K)
                if visualizer is None:
                    pred = model(frames, dt, valid)
                    aux = None
                else:
                    pred, aux = model(frames, dt, valid, return_aux=True)
                pred_seq = pred.reshape(B, S, C, H, W)

            loss, _ = loss_fn(pred, target)
            batch_psnr = psnr(pred, target).detach().cpu()
            batch_ssim = ssim(pred, target).detach().cpu()
            abs_error = (pred - target).detach().abs().flatten(1)
            batch_error_mean = abs_error.mean(dim=1).cpu()
            batch_error_p95 = abs_error.quantile(0.95, dim=1).cpu()
            batch_error_max = abs_error.max(dim=1).values.cpu()
            batch_size = pred.shape[0]

            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["psnr"] += float(batch_psnr.sum().item())
            totals["ssim"] += float(batch_ssim.sum().item())
            totals["error_mean"] += float(batch_error_mean.sum().item())
            totals["error_p95"] += float(batch_error_p95.sum().item())
            totals["error_max"] += float(batch_error_max.sum().item())
            count += batch_size

            temporal_totals["tde"] += float(temporal_delta_error(pred_seq, target_seq).detach().cpu()) * B
            temporal_totals["tae"] += float(temporal_acceleration_error(pred_seq, target_seq).detach().cpu()) * B
            temporal_count += B

            if visualizer is not None and not visualizer.is_full():
                center_seq_idx = S // 2
                center_pred = pred_seq[:, center_seq_idx]
                center_target = target_seq[:, center_seq_idx]
                center_psnr = psnr(center_pred, center_target).detach().cpu()
                center_ssim = ssim(center_pred, center_target).detach().cpu()
                center_batch = {
                    "frames": frames_seq[:, center_seq_idx],
                    "target": center_target,
                    "dt": dt_seq[:, center_seq_idx],
                    "valid": valid_seq[:, center_seq_idx],
                }
                center_aux = None
                if aux is not None:
                    center_aux = {}
                    for key, value in aux.items():
                        if torch.is_tensor(value) and value.shape[0] == B * S:
                            center_aux[key] = value.reshape(B, S, *value.shape[1:])[:, center_seq_idx]
                        else:
                            center_aux[key] = value
                visualizer.add_batch(center_batch, center_pred, center_psnr, center_ssim, aux=center_aux)
        else:
            if visualizer is None:
                pred = model(batch["frames"], batch["dt"], batch["valid"])
                aux = None
            else:
                pred, aux = model(batch["frames"], batch["dt"], batch["valid"], return_aux=True)
            loss, _ = loss_fn(pred, batch["target"])
            batch_psnr = psnr(pred, batch["target"]).detach().cpu()
            batch_ssim = ssim(pred, batch["target"]).detach().cpu()
            abs_error = (pred - batch["target"]).detach().abs().flatten(1)
            batch_error_mean = abs_error.mean(dim=1).cpu()
            batch_error_p95 = abs_error.quantile(0.95, dim=1).cpu()
            batch_error_max = abs_error.max(dim=1).values.cpu()
            batch_size = pred.shape[0]

            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["psnr"] += float(batch_psnr.sum().item())
            totals["ssim"] += float(batch_ssim.sum().item())
            totals["error_mean"] += float(batch_error_mean.sum().item())
            totals["error_p95"] += float(batch_error_p95.sum().item())
            totals["error_max"] += float(batch_error_max.sum().item())
            count += batch_size

            if visualizer is not None and not visualizer.is_full():
                visualizer.add_batch(batch, pred, batch_psnr, batch_ssim, aux=aux)

    if count == 0:
        raise RuntimeError("Validation loader produced no samples")

    out = {key: value / count for key, value in totals.items()}
    if temporal_count > 0:
        out.update({key: value / temporal_count for key, value in temporal_totals.items()})
    return out


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
    train_simulation_mix = build_train_simulation_mix(cfg, device)

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

    model = build_model_from_config(cfg, device)

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    stage = cfg.get("stage", "e2e")
    configure_stage(model, stage, freeze_cfg=cfg.get("freeze"))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters after stage/freeze configuration")

    optimizer = torch.optim.AdamW(
        trainable_params,
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
    best_ssim = -1e9
    try:
        for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
            epoch_start = time.perf_counter()
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
                loss_weights=cfg.get("loss", {}),
                writer=writer,
                global_step=global_step,
                step_log_interval=int(tb_cfg.get("train_step_interval", 10)),
                simulation_mix=train_simulation_mix,
            )

            collect_val_visuals = bool(val_tb_cfg.get("enabled", True)) and epoch % val_viz_every == 0
            visualizer = None
            if collect_val_visuals:
                visualizer = ValidationVisualizer(
                    num_samples=int(val_tb_cfg.get("num_samples", 8)),
                    grid_cols=int(val_tb_cfg.get("grid_cols", 0)),
                )

            val_log = validate(model, val_loader, loss_fn, simulator, device, visualizer=visualizer)
            train_log["epoch_time"] = time.perf_counter() - epoch_start
            is_best = val_log["psnr"] > best_psnr
            best_psnr = max(best_psnr, val_log["psnr"])
            best_ssim = max(best_ssim, val_log["ssim"])
            val_log["best_psnr"] = best_psnr
            val_log["best_ssim"] = best_ssim
            print({"epoch": epoch, "train": train_log, "val": val_log})

            maybe_add_scalars(writer, "train", train_log, epoch)
            maybe_add_scalars(writer, "val", val_log, epoch)
            if writer is not None:
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)

            if visualizer is not None and writer is not None and visualizer.samples:
                writer.add_image("val/summary", visualizer.summary_image(), epoch, dataformats="HWC")
                for idx, image in enumerate(visualizer.sequence_images):
                    writer.add_image(f"val/sample_{idx:02d}_sequence", image, epoch, dataformats="HWC")
                for idx, image in enumerate(visualizer.flow_images):
                    writer.add_image(f"val/sample_{idx:02d}_flow_magnitude", image, epoch, dataformats="HWC")
                for idx, image in enumerate(visualizer.weight_images):
                    writer.add_image(f"val/sample_{idx:02d}_fusion_weights", image, epoch, dataformats="HWC")

            ckpt = {"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "val": val_log}
            torch.save(ckpt, out_dir / "last.pt")
            if is_best:
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
