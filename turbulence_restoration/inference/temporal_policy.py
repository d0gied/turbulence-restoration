
from __future__ import annotations

from typing import Dict

import numpy as np

from turbulence_restoration.data.temporal_sampler import TEMPORAL_POLICIES, TemporalSampler

INFERENCE_POLICIES = {
    "dynamic_150ms": TEMPORAL_POLICIES["k9_150ms"],
    "default_200ms": TEMPORAL_POLICIES["k9_200ms"],
    "static_400ms": TEMPORAL_POLICIES["k9_400ms"],
    "k5_consecutive": TEMPORAL_POLICIES["k5_consecutive_60fps"],
}


class InferenceTemporalPolicy:
    def __init__(self, policy: str = "default_200ms"):
        if policy not in INFERENCE_POLICIES:
            raise ValueError(f"Unknown inference policy: {policy}. Available: {list(INFERENCE_POLICIES)}")
        self.policy = policy
        self.offsets_sec = INFERENCE_POLICIES[policy]
        self._offsets_idx_cache: Dict[float, np.ndarray] = {}

    def get_offsets_idx(self, fps: float) -> np.ndarray:
        fps = float(fps)
        offsets_idx = self._offsets_idx_cache.get(fps)
        if offsets_idx is None:
            offsets_idx = np.round(self.offsets_sec * fps).astype(np.int64)
            self._offsets_idx_cache[fps] = offsets_idx
        return offsets_idx

    def get_frame_indices(self, center_idx: int, fps: float, num_frames: int):
        offsets_idx = self.get_offsets_idx(fps)
        raw = center_idx + offsets_idx
        return TemporalSampler._reflect_indices(raw, num_frames)

    def get_indices(self, center_idx: int, fps: float, num_frames: int):
        indices = self.get_frame_indices(center_idx, fps, num_frames)
        valid = np.ones(len(indices), dtype=np.float32)
        return indices, self.offsets_sec.copy(), valid
