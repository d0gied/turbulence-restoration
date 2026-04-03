
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

TEMPORAL_POLICIES: Dict[str, np.ndarray] = {
    "k9_200ms": np.array([-0.1000, -0.0667, -0.0333, -0.0167, 0.0, 0.0167, 0.0333, 0.0667, 0.1000], dtype=np.float32),
    "k9_150ms": np.array([-0.0750, -0.0500, -0.0333, -0.0167, 0.0, 0.0167, 0.0333, 0.0500, 0.0750], dtype=np.float32),
    "k9_400ms": np.array([-0.2000, -0.1333, -0.0667, -0.0333, 0.0, 0.0333, 0.0667, 0.1333, 0.2000], dtype=np.float32),
    "k5_consecutive_60fps": np.array([-0.0333, -0.0167, 0.0, 0.0167, 0.0333], dtype=np.float32),
    "k9_consecutive_60fps": np.array([-0.0667, -0.0500, -0.0333, -0.0167, 0.0, 0.0167, 0.0333, 0.0500, 0.0667], dtype=np.float32),
    "k1_center": np.array([0.0], dtype=np.float32),
}


@dataclass
class TemporalSample:
    indices: np.ndarray
    dt: np.ndarray
    valid: np.ndarray


class TemporalSampler:
    """Select frames by physical time offsets, not only by index count."""

    def __init__(self, policy: str = "k9_200ms", drop_duplicates: bool = False):
        if policy not in TEMPORAL_POLICIES:
            raise ValueError(f"Unknown policy: {policy}. Available: {list(TEMPORAL_POLICIES)}")
        self.policy = policy
        self.offsets_sec = TEMPORAL_POLICIES[policy]
        self.drop_duplicates = drop_duplicates

    def sample(self, center_idx: int, fps: float, num_frames: int) -> TemporalSample:
        offsets_idx = np.round(self.offsets_sec * float(fps)).astype(np.int64)
        raw_indices = center_idx + offsets_idx
        indices = self._reflect_indices(raw_indices, num_frames)
        dt = self.offsets_sec.copy()
        valid = np.ones(len(indices), dtype=np.float32)

        if self.drop_duplicates:
            seen = set()
            new_idx, new_dt = [], []
            for i, d in zip(indices, dt):
                if int(i) not in seen:
                    new_idx.append(int(i))
                    new_dt.append(float(d))
                    seen.add(int(i))
            indices = np.array(new_idx, dtype=np.int64)
            dt = np.array(new_dt, dtype=np.float32)
            valid = np.ones(len(indices), dtype=np.float32)

        return TemporalSample(indices=indices, dt=dt, valid=valid)

    @staticmethod
    def _reflect_indices(indices: np.ndarray, n: int) -> np.ndarray:
        if n <= 1:
            return np.zeros_like(indices, dtype=np.int64)
        out = []
        for idx in indices:
            idx = int(idx)
            while idx < 0 or idx >= n:
                if idx < 0:
                    idx = -idx
                if idx >= n:
                    idx = 2 * n - idx - 2
            out.append(int(np.clip(idx, 0, n - 1)))
        return np.array(out, dtype=np.int64)
