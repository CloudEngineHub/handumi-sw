"""HandUMI teleop backend adapter for AgileX Piper arms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from handumi.real.base import TeleopRobotBackend
from handumi.real.piper.driver import (
    PiperCanEnvironment,
    load_piper_can_settings,
    piper_mdeg_to_q,
    q_to_piper_mdeg,
)
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.robots.registry import RobotRuntime


class PiperBackend:
    """Robot-neutral teleop wrapper around ``PiperCanEnvironment``."""

    name = "piper"

    def __init__(
        self,
        environment: PiperCanEnvironment,
        *,
        joint_names: tuple[str, ...],
        max_width_mm: float,
        active_sides: tuple[str, ...] = ("left", "right"),
    ) -> None:
        self.environment = environment
        self.joint_names = tuple(joint_names)
        self.max_width_mm = float(max_width_mm)
        self.active_sides = tuple(active_sides)
        self._last_q: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        *,
        runtime: RobotRuntime,
        rig_config: Path,
        active_sides: tuple[str, ...] = ("left", "right"),
    ) -> "PiperBackend":
        return cls(
            PiperCanEnvironment(
                load_piper_can_settings(rig_config, runtime.config.real)
            ),
            joint_names=runtime.joint_names,
            max_width_mm=runtime.config.gripper_max_width_m * 1000.0,
            active_sides=active_sides,
        )

    def setup(self, *, repair: bool = True) -> None:
        settings = self.environment.settings
        ensure_can_interfaces_ready(
            [settings.left_port, settings.right_port],
            bitrate=settings.bitrate,
            restart_ms=settings.restart_ms,
            repair=repair,
        )

    def connect(self) -> None:
        self.environment.connect()

    def disconnect(self) -> None:
        self.environment.close()

    def read(self, base_q: np.ndarray | None = None) -> np.ndarray:
        base = self._base_q(base_q)
        feedback = self.environment.feedback_mdeg()
        if not feedback:
            return base
        return piper_mdeg_to_q(
            left_mdeg=feedback.get("left", np.zeros(6, dtype=np.int64)),
            right_mdeg=feedback.get("right", np.zeros(6, dtype=np.int64)),
            actuated_names=self.joint_names,
            base_q=base,
        )

    def home(self, q: np.ndarray) -> None:
        self._last_q = np.asarray(q, dtype=np.float32).copy()
        self.environment.home(q_to_piper_mdeg(q, self.joint_names))

    def move_home(self, q: np.ndarray) -> None:
        self._last_q = np.asarray(q, dtype=np.float32).copy()
        self.environment.move_home(q_to_piper_mdeg(q, self.joint_names))

    def write(
        self,
        q: np.ndarray,
        gripper_openings: dict[str, float],
    ) -> None:
        self._last_q = np.asarray(q, dtype=np.float32).copy()
        self.environment.set_q(q, self.joint_names)
        self.environment.set_gripper_widths_mm(
            {
                side: float(np.clip(value, 0.0, 1.0)) * self.max_width_mm
                for side, value in gripper_openings.items()
            }
        )

    def hold(self, base_q: np.ndarray) -> np.ndarray:
        held = self.environment.hold_current_commands_mdeg()
        q = piper_mdeg_to_q(
            left_mdeg=held["left"],
            right_mdeg=held["right"],
            actuated_names=self.joint_names,
            base_q=base_q,
        )
        self._last_q = q.copy()
        return q

    def check_health(self) -> None:
        self.environment.raise_if_failed()

    def _base_q(self, base_q: np.ndarray | None) -> np.ndarray:
        if base_q is not None:
            return np.asarray(base_q, dtype=np.float32).copy()
        if self._last_q is not None:
            return self._last_q.copy()
        return np.zeros(len(self.joint_names), dtype=np.float32)


def build_backend(
    *,
    runtime: RobotRuntime,
    rig_config: Path,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> TeleopRobotBackend:
    return PiperBackend.from_config(
        runtime=runtime,
        rig_config=rig_config,
        active_sides=active_sides,
    )


__all__ = ["PiperBackend", "build_backend"]
