
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

NumberRange = Union[float, Tuple[float, float]]


@dataclass
class TurbulenceConfig:
    """
    GPU-friendly turbulence simulator.

    Units:
      spatial_scales: pixels
      temporal_scale: seconds
      max_displacement: pixels
      blur_sigma: pixels
    """

    octaves: int = 4
    spatial_scales: Tuple[float, ...] = (96.0, 48.0, 24.0, 12.0)
    octave_weights: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    temporal_scale: float = 0.12
    max_displacement: NumberRange = (8.0, 18.0)
    blur_sigma: NumberRange = (0.0, 4.0)
    scintillation_strength: NumberRange = (0.0, 0.05)
    scintillation_smoothing: float = 9.0
    flow_smoothing: float = 0.0
    clamp_output: bool = True
    seed_u: int = 17
    seed_v: int = 29
    seed_s: int = 43

    @staticmethod
    def from_severity(severity: str) -> "TurbulenceConfig":
        presets = {
            "weak": TurbulenceConfig(
                max_displacement=(2.0, 8.0),
                blur_sigma=(0.0, 2.0),
                scintillation_strength=(0.0, 0.025),
                temporal_scale=0.10,
            ),
            "medium": TurbulenceConfig(
                max_displacement=(8.0, 18.0),
                blur_sigma=(0.5, 4.0),
                scintillation_strength=(0.01, 0.05),
                temporal_scale=0.12,
            ),
            "strong": TurbulenceConfig(
                max_displacement=(18.0, 40.0),
                blur_sigma=(2.0, 10.0),
                scintillation_strength=(0.03, 0.10),
                temporal_scale=0.16,
            ),
            "stress": TurbulenceConfig(
                max_displacement=(30.0, 40.0),
                blur_sigma=(8.0, 40.0),
                scintillation_strength=(0.06, 0.14),
                temporal_scale=0.20,
            ),
        }
        if severity not in presets:
            raise ValueError(f"Unknown severity: {severity}. Available: {list(presets)}")
        return presets[severity]


def _as_range(value: NumberRange) -> Tuple[float, float]:
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def _rand_range(
    value: NumberRange,
    shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    lo, hi = _as_range(value)
    if lo == hi:
        return torch.full(tuple(shape), lo, device=device, dtype=dtype)
    return torch.empty(tuple(shape), device=device, dtype=dtype).uniform_(lo, hi, generator=generator)


def fade(t: torch.Tensor) -> torch.Tensor:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + t * (b - a)


def _hash3(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Int64 hash for GPU tensors.

    Constants are moderate to avoid Python->int64 overflow in seed term.
    Int64 multiplication can wrap; this is acceptable for hashing.
    """
    h = ix.to(torch.int64) * 374761393
    h = h + iy.to(torch.int64) * 668265263
    h = h + iz.to(torch.int64) * 1442695041
    h = h + int(seed) * 1013904223
    h = h ^ (h >> 13)
    h = h * 1274126177
    h = h ^ (h >> 16)
    return torch.remainder(h, 2147483647)


def _grad_dot(hash_value: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    h = torch.remainder(hash_value, 12)
    gx = torch.zeros_like(dx)
    gy = torch.zeros_like(dy)
    gz = torch.zeros_like(dz)

    gx = torch.where((h == 0) | (h == 2) | (h == 4) | (h == 6), torch.ones_like(gx), gx)
    gx = torch.where((h == 1) | (h == 3) | (h == 5) | (h == 7), -torch.ones_like(gx), gx)

    gy = torch.where((h == 0) | (h == 1) | (h == 8) | (h == 10), torch.ones_like(gy), gy)
    gy = torch.where((h == 2) | (h == 3) | (h == 9) | (h == 11), -torch.ones_like(gy), gy)

    gz = torch.where((h == 4) | (h == 5) | (h == 8) | (h == 9), torch.ones_like(gz), gz)
    gz = torch.where((h == 6) | (h == 7) | (h == 10) | (h == 11), -torch.ones_like(gz), gz)

    return gx * dx + gy * dy + gz * dz


def perlin3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    Vectorized 3D Perlin noise on CPU/GPU.

    x, y, z are broadcastable tensors.
    """
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    z0 = torch.floor(z).to(torch.int64)

    xf = x - x0.to(x.dtype)
    yf = y - y0.to(y.dtype)
    zf = z - z0.to(z.dtype)

    u = fade(xf)
    v = fade(yf)
    w = fade(zf)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    n000 = _grad_dot(_hash3(x0, y0, z0, seed), xf, yf, zf)
    n100 = _grad_dot(_hash3(x1, y0, z0, seed), xf - 1.0, yf, zf)
    n010 = _grad_dot(_hash3(x0, y1, z0, seed), xf, yf - 1.0, zf)
    n110 = _grad_dot(_hash3(x1, y1, z0, seed), xf - 1.0, yf - 1.0, zf)

    n001 = _grad_dot(_hash3(x0, y0, z1, seed), xf, yf, zf - 1.0)
    n101 = _grad_dot(_hash3(x1, y0, z1, seed), xf - 1.0, yf, zf - 1.0)
    n011 = _grad_dot(_hash3(x0, y1, z1, seed), xf, yf - 1.0, zf - 1.0)
    n111 = _grad_dot(_hash3(x1, y1, z1, seed), xf - 1.0, yf - 1.0, zf - 1.0)

    x00 = lerp(n000, n100, u)
    x10 = lerp(n010, n110, u)
    x01 = lerp(n001, n101, u)
    x11 = lerp(n011, n111, u)

    y0v = lerp(x00, x10, v)
    y1v = lerp(x01, x11, v)
    return lerp(y0v, y1v, w) * 0.87



def value_noise3d(
    B: int,
    K: int,
    H: int,
    W: int,
    timestamps: torch.Tensor,
    spatial_scale: float,
    temporal_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Fast GPU value-noise backend.

    It is Perlin-like in usage: smooth trilinear interpolation of a coarse
    random 3D lattice over x, y, t. It is much faster than explicit hash
    Perlin on CPU and remains fully GPU-accelerated.
    """
    Hc = max(4, int(H / float(spatial_scale)) + 4)
    Wc = max(4, int(W / float(spatial_scale)) + 4)

    t_min = timestamps.amin(dim=1, keepdim=True)
    t_span = (timestamps.amax(dim=1) - timestamps.amin(dim=1)).max()
    D = max(4, int(torch.ceil(t_span / float(temporal_scale)).detach().cpu().item()) + 4)

    if generator is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    else:
        gen = generator

    lattice = torch.randn(B, 1, D, Hc, Wc, device=device, dtype=dtype, generator=gen)

    xx, yy = _meshgrid_xy(H, W, device, dtype)
    x_c = xx / float(spatial_scale) + 1.0
    y_c = yy / float(spatial_scale) + 1.0

    x_norm = 2.0 * x_c / max(Wc - 1, 1) - 1.0
    y_norm = 2.0 * y_c / max(Hc - 1, 1) - 1.0

    z_c = (timestamps - t_min) / float(temporal_scale) + 1.0
    z_norm = 2.0 * z_c / max(D - 1, 1) - 1.0

    grid_x = x_norm[None, None].expand(B, K, H, W)
    grid_y = y_norm[None, None].expand(B, K, H, W)
    grid_z = z_norm[:, :, None, None].expand(B, K, H, W)
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    out = F.grid_sample(
        lattice,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return out[:, 0]

def _meshgrid_xy(H: int, W: int, device: torch.device, dtype: torch.dtype):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx, yy


def warp_image(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """
    x:    [B,K,C,H,W] or [N,C,H,W]
    flow: [B,K,2,H,W] or [N,2,H,W]
    Output samples x at output coordinates + flow.
    """
    original_dim = x.dim()
    if x.dim() == 5:
        B, K, C, H, W = x.shape
        x_flat = x.reshape(B * K, C, H, W)
        flow_flat = flow.reshape(B * K, 2, H, W)
    elif x.dim() == 4:
        x_flat = x
        flow_flat = flow
        _, C, H, W = x.shape
        B = K = None
    else:
        raise ValueError(f"Expected x with 4 or 5 dims, got {x.shape}")

    N = x_flat.shape[0]
    device = x_flat.device
    dtype = x_flat.dtype
    xx, yy = _meshgrid_xy(H, W, device, dtype)
    xx = xx[None].expand(N, H, W)
    yy = yy[None].expand(N, H, W)

    grid_x = xx + flow_flat[:, 0]
    grid_y = yy + flow_flat[:, 1]
    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    warped = F.grid_sample(x_flat, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    if original_dim == 5:
        return warped.reshape(B, K, C, H, W)
    return warped


def gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype, truncate: float = 3.0) -> torch.Tensor:
    if sigma <= 1e-6:
        return torch.ones(1, device=device, dtype=dtype)
    radius = int(max(1, round(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def gaussian_blur2d_per_frame(x: torch.Tensor, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Depthwise separable Gaussian blur.
    x: [B,K,C,H,W] or [N,C,H,W]
    sigma: scalar or [B,K] or [N]
    """
    original_dim = x.dim()
    if original_dim == 5:
        B, K, C, H, W = x.shape
        x_flat = x.reshape(B * K, C, H, W)
        sigma_flat = torch.full((B * K,), float(sigma), device=x.device, dtype=x.dtype) if not torch.is_tensor(sigma) else sigma.reshape(B * K).to(x)
    elif original_dim == 4:
        N, C, H, W = x.shape
        x_flat = x
        sigma_flat = torch.full((N,), float(sigma), device=x.device, dtype=x.dtype) if not torch.is_tensor(sigma) else sigma.reshape(N).to(x)
        B = K = None
    else:
        raise ValueError(f"Expected x with 4 or 5 dims, got {x.shape}")

    out = []
    for i in range(x_flat.shape[0]):
        sig = float(sigma_flat[i].detach().clamp_min(0.0).item())
        if sig <= 1e-4:
            out.append(x_flat[i : i + 1])
            continue
        # Reflect padding requires pad < dimension. Clamp radius for tiny crops.
        radius = int(max(1, round(3.0 * sig)))
        radius = min(radius, max(1, W - 1), max(1, H - 1))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-(coords ** 2) / (2.0 * sig * sig))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        kx = kernel.reshape(1, 1, 1, -1).repeat(C, 1, 1, 1)
        ky = kernel.reshape(1, 1, -1, 1).repeat(C, 1, 1, 1)
        xi = F.pad(x_flat[i : i + 1], (radius, radius, 0, 0), mode="reflect")
        xi = F.conv2d(xi, kx, groups=C)
        xi = F.pad(xi, (0, 0, radius, radius), mode="reflect")
        xi = F.conv2d(xi, ky, groups=C)
        out.append(xi)
    y = torch.cat(out, dim=0)
    if original_dim == 5:
        return y.reshape(B, K, C, H, W)
    return y


class GPUTurbulenceSimulator(nn.Module):
    """
    Synthetic turbulence simulator with GPU acceleration.

    clean_frames: [K,3,H,W] or [B,K,3,H,W], float in [0,1]
    timestamps:   [K] or [B,K], seconds
    """

    def __init__(self, config: TurbulenceConfig = TurbulenceConfig()):
        super().__init__()
        self.config = config

    def forward(
        self,
        clean_frames: torch.Tensor,
        timestamps: torch.Tensor,
        return_meta: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        squeeze_batch = False
        if clean_frames.dim() == 4:
            clean_frames = clean_frames.unsqueeze(0)
            squeeze_batch = True
        if clean_frames.dim() != 5:
            raise ValueError(f"clean_frames must be [K,C,H,W] or [B,K,C,H,W], got {clean_frames.shape}")

        B, K, C, H, W = clean_frames.shape
        device = clean_frames.device
        dtype = clean_frames.dtype

        timestamps = timestamps.to(device=device, dtype=dtype)
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0).expand(B, K)
        elif timestamps.shape != (B, K):
            raise ValueError(f"timestamps must be [K] or [B,K], got {timestamps.shape}")

        flow = self.generate_flow(B, K, H, W, timestamps, device, dtype, generator)
        distorted = warp_image(clean_frames, flow)

        blur_sigma = _rand_range(self.config.blur_sigma, (B, K), device, dtype, generator)
        distorted = gaussian_blur2d_per_frame(distorted, blur_sigma)

        alpha = self.generate_scintillation(B, K, H, W, device, dtype, generator)
        distorted = distorted * alpha

        if self.config.clamp_output:
            distorted = distorted.clamp(0.0, 1.0)

        if squeeze_batch:
            distorted_out, flow_out, alpha_out, blur_out = distorted[0], flow[0], alpha[0], blur_sigma[0]
        else:
            distorted_out, flow_out, alpha_out, blur_out = distorted, flow, alpha, blur_sigma

        if not return_meta:
            return distorted_out
        return distorted_out, {"flow": flow_out, "alpha": alpha_out, "blur_sigma": blur_out}

    def generate_flow(self, B, K, H, W, timestamps, device, dtype, generator=None) -> torch.Tensor:
        cfg = self.config
        u = torch.zeros(B, K, H, W, device=device, dtype=dtype)
        v = torch.zeros(B, K, H, W, device=device, dtype=dtype)

        weights = cfg.octave_weights[: cfg.octaves]
        scales = cfg.spatial_scales[: cfg.octaves]
        if len(weights) < cfg.octaves or len(scales) < cfg.octaves:
            raise ValueError("octave_weights and spatial_scales must have at least `octaves` values")

        for octave_idx, (scale, weight) in enumerate(zip(scales, weights)):
            # Fast GPU value-noise backend: temporally coherent smooth fields.
            u = u + float(weight) * value_noise3d(
                B, K, H, W, timestamps, float(scale), float(cfg.temporal_scale),
                seed=cfg.seed_u + 101 * octave_idx, device=device, dtype=dtype, generator=generator
            )
            v = v + float(weight) * value_noise3d(
                B, K, H, W, timestamps, float(scale), float(cfg.temporal_scale),
                seed=cfg.seed_v + 103 * octave_idx, device=device, dtype=dtype, generator=generator
            )

        mag = torch.sqrt(u * u + v * v)
        max_mag = mag.flatten(2).amax(dim=2).clamp_min(1e-6)
        u = u / max_mag[:, :, None, None]
        v = v / max_mag[:, :, None, None]

        amp = _rand_range(cfg.max_displacement, (B, K), device, dtype, generator)
        u = u * amp[:, :, None, None]
        v = v * amp[:, :, None, None]

        flow = torch.stack([u, v], dim=2)
        if cfg.flow_smoothing > 1e-4:
            flow_as_img = flow.reshape(B * K, 2, H, W)
            flow_as_img = gaussian_blur2d_per_frame(flow_as_img, cfg.flow_smoothing)
            flow = flow_as_img.reshape(B, K, 2, H, W)
        return flow

    def generate_scintillation(self, B, K, H, W, device, dtype, generator=None) -> torch.Tensor:
        cfg = self.config
        beta = _rand_range(cfg.scintillation_strength, (B, K, 1, 1, 1), device, dtype, generator)
        if torch.all(beta <= 1e-8):
            return torch.ones(B, K, 1, H, W, device=device, dtype=dtype)

        noise = torch.randn(B, K, 1, H, W, device=device, dtype=dtype, generator=generator)
        if cfg.scintillation_smoothing > 1e-4:
            noise = gaussian_blur2d_per_frame(noise, cfg.scintillation_smoothing)

        flat = noise.flatten(3)
        mean = flat.mean(dim=3, keepdim=True)[:, :, :, :, None]
        std = flat.std(dim=3, keepdim=True, unbiased=False).clamp_min(1e-6)[:, :, :, :, None]
        noise = (noise - mean) / std
        alpha = 1.0 + beta * noise
        return alpha.clamp(0.5, 1.5)
