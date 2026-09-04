"""Idle-arm detection and parking for single-arm episodes."""

from __future__ import annotations

import numpy as np
import pytest

from handumi.dataset.idle_arm import (
    ArmActivity,
    IdleArmThresholds,
    arm_activity,
    idle_sides,
    raw_state_arm_activity,
)


def _path(travel: float, frames: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames)
    return np.column_stack([0.3 + travel * t, 0.1 * np.sin(t), 0.05 + 0.0 * t])


def test_travel_is_the_bounding_box_diagonal_not_the_path_length() -> None:
    # A hand trembling back and forth covers a long path in a tiny box.
    jitter = 0.005 * np.sin(np.linspace(0.0, 200.0, 600))
    positions = np.column_stack([0.3 + jitter, 0.2 + jitter, 0.05 + jitter])
    activity = arm_activity(positions, None, side="left")
    assert activity.travel_m == pytest.approx(np.sqrt(3) * 0.01, abs=1e-6)
    assert activity.idle


def test_thresholds_split_idle_from_working_arms() -> None:
    thresholds = IdleArmThresholds(max_travel_m=0.2, max_gripper_change_m=0.04)
    idle = arm_activity(_path(0.05), np.full(60, 0.03), side="left", thresholds=thresholds)
    working = arm_activity(_path(0.45), np.full(60, 0.03), side="left", thresholds=thresholds)
    squeezing = arm_activity(
        _path(0.05), np.linspace(0.0, 0.06, 60), side="left", thresholds=thresholds
    )
    assert idle.idle and not working.idle
    assert not squeezing.idle  # a full grasp cycle is task activity


def test_missing_grippers_count_as_no_change() -> None:
    activity = arm_activity(_path(0.05), np.full(60, np.nan), side="right")
    assert activity.gripper_change_m == 0.0 and activity.idle


def test_thresholds_reject_nonsense() -> None:
    with pytest.raises(ValueError):
        IdleArmThresholds(max_travel_m=0.0)


def test_raw_states_grade_both_arms() -> None:
    states = np.zeros((60, 16), dtype=np.float32)
    states[:, 0:3] = _path(0.05)
    states[:, 7:10] = _path(0.5)
    states[:, 14] = 0.03
    states[:, 15] = np.linspace(0.0, 0.06, 60)
    activity = raw_state_arm_activity(states)
    assert activity["left"].idle and not activity["right"].idle
    assert idle_sides(activity) == ("left",)


def test_idle_sides_lists_every_idle_arm_in_order() -> None:
    both_idle = {
        side: ArmActivity(side=side, travel_m=0.01, gripper_change_m=0.0, idle=True)
        for side in ("left", "right")
    }
    both_working = {
        side: ArmActivity(side=side, travel_m=0.5, gripper_change_m=0.05, idle=False)
        for side in ("left", "right")
    }
    assert idle_sides(both_idle) == ("left", "right")
    assert idle_sides(both_working) == ()
