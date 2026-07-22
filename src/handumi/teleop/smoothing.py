"""Joint-command smoothing for real robot teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class JointSmoothingConfig:
    """Runtime tuning for post-IK joint command smoothing."""

    cutoff_hz: float = 4.0
    max_velocity_rad_s: float = 1.5

    def validate(self) -> None:
        if self.cutoff_hz < 0.0:
            raise ValueError("cutoff_hz must be >= 0.")
        if self.max_velocity_rad_s < 0.0:
            raise ValueError("max_velocity_rad_s must be >= 0.")


class JointCommandSmoother:
    """Low-pass and velocity-limit joint targets without changing the IK solver."""

    def __init__(self, config: JointSmoothingConfig | None = None) -> None:
        self.config = config or JointSmoothingConfig()
        self.config.validate()
        self._q: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._q is not None

    def reset(self, q: np.ndarray | None = None) -> None:
        self._q = None if q is None else np.asarray(q, dtype=np.float32).copy()

    def smooth(self, target_q: np.ndarray, *, dt: float) -> np.ndarray:
        target = np.asarray(target_q, dtype=np.float32)
        if self._q is None:
            self._q = target.copy()
            return self._q.copy()
        if dt <= 0.0:
            return self._q.copy()

        q = self._low_pass(target, dt=dt)
        q = self._limit_velocity(q, dt=dt)
        self._q = q.astype(np.float32, copy=False)
        return self._q.copy()

    def _low_pass(self, target: np.ndarray, *, dt: float) -> np.ndarray:
        assert self._q is not None
        if self.config.cutoff_hz <= 0.0:
            return target.copy()
        tau = 1.0 / (2.0 * math.pi * self.config.cutoff_hz)
        alpha = dt / (tau + dt)
        return self._q + alpha * (target - self._q)

    def _limit_velocity(self, target: np.ndarray, *, dt: float) -> np.ndarray:
        assert self._q is not None
        if self.config.max_velocity_rad_s <= 0.0:
            return target
        max_delta = self.config.max_velocity_rad_s * dt
        delta = np.clip(target - self._q, -max_delta, max_delta)
        return self._q + delta
