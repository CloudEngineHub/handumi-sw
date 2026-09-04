"""Arms that stay idle through an episode.

A single-arm task recorded with a bimanual HandUMI rig still carries the idle
hand's pose in every frame, resting wherever the demonstrator left it. On
tblock that is 33 cm in front of the robot base and 7 cm above the table with
the tool pointing down, which a Piper reaches only with a straight elbow and
the wrist on its limit. The pose is what the cameras saw, so the conversion
keeps it; what this module provides is the measurement, so screening can
report how each arm was used and operators can be trained to rest the idle
hand where the robot is comfortable.

An arm is idle when its tool travelled less than ``max_travel_m`` over the
episode and its gripper never changed by more than ``max_gripper_change_m``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from handumi.dataset.raw import (
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)

SIDES = ("left", "right")
# On tblock the two populations do not overlap: idle hands travel under
# 0.16 m over an episode (bounding-box diagonal) and working hands over 0.32 m.
DEFAULT_MAX_TRAVEL_M = 0.20
# A grasp cycle opens and closes the tool by several centimeters; an idle hand
# squeezing the handle without moving stays well under this.
DEFAULT_MAX_GRIPPER_CHANGE_M = 0.04


@dataclass(frozen=True)
class IdleArmThresholds:
    max_travel_m: float = DEFAULT_MAX_TRAVEL_M
    max_gripper_change_m: float = DEFAULT_MAX_GRIPPER_CHANGE_M

    def __post_init__(self) -> None:
        if self.max_travel_m <= 0.0 or self.max_gripper_change_m < 0.0:
            raise ValueError(
                "max_travel_m must be > 0 and max_gripper_change_m >= 0."
            )


@dataclass(frozen=True)
class ArmActivity:
    """How much one arm did over an episode."""

    side: str
    travel_m: float
    gripper_change_m: float
    idle: bool

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def arm_activity(
    positions: np.ndarray,
    gripper_widths_m: np.ndarray | None,
    *,
    side: str,
    thresholds: IdleArmThresholds = IdleArmThresholds(),
) -> ArmActivity:
    """Grade one arm from its tool positions ``(T, 3)`` and gripper widths ``(T,)``.

    Travel is the diagonal of the bounding box of the path, which ignores
    tracking jitter and hand tremor but grows with any real displacement.
    Non-finite gripper samples (a capture without grippers) count as no change.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (T, 3), got {pos.shape}.")
    finite = np.isfinite(pos).all(axis=1)
    travel = 0.0
    if finite.any():
        span = pos[finite].max(axis=0) - pos[finite].min(axis=0)
        travel = float(np.linalg.norm(span))
    change = 0.0
    if gripper_widths_m is not None:
        widths = np.asarray(gripper_widths_m, dtype=np.float64).reshape(-1)
        widths = widths[np.isfinite(widths)]
        if widths.size:
            change = float(widths.max() - widths.min())
    idle = (
        travel < thresholds.max_travel_m
        and change < thresholds.max_gripper_change_m
    )
    return ArmActivity(side=side, travel_m=travel, gripper_change_m=change, idle=idle)


def raw_state_arm_activity(
    states: np.ndarray,
    *,
    thresholds: IdleArmThresholds = IdleArmThresholds(),
) -> dict[str, ArmActivity]:
    """Grade both arms from raw HandUMI states ``(T, 16)``."""
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < RIGHT_GRIPPER_INDEX + 1:
        raise ValueError(f"states must have shape (T, 16), got {arr.shape}.")
    return {
        "left": arm_activity(
            arr[:, LEFT_POSE_SLICE][:, :3],
            arr[:, LEFT_GRIPPER_INDEX],
            side="left",
            thresholds=thresholds,
        ),
        "right": arm_activity(
            arr[:, RIGHT_POSE_SLICE][:, :3],
            arr[:, RIGHT_GRIPPER_INDEX],
            side="right",
            thresholds=thresholds,
        ),
    }


def idle_sides(activity: dict[str, ArmActivity]) -> tuple[str, ...]:
    """The arms that did nothing in the episode, in ``left``/``right`` order."""
    return tuple(side for side in SIDES if side in activity and activity[side].idle)


__all__ = [
    "DEFAULT_MAX_GRIPPER_CHANGE_M",
    "DEFAULT_MAX_TRAVEL_M",
    "ArmActivity",
    "IdleArmThresholds",
    "arm_activity",
    "idle_sides",
    "raw_state_arm_activity",
]
