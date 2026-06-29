"""Helpers for converting HandUMI raw state vectors into pose targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handumi.dataset.raw import (
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
    validate_raw_state_shape,
)


@dataclass(frozen=True)
class HandumiRawState:
    left_position: np.ndarray
    left_rotation: np.ndarray
    right_position: np.ndarray
    right_rotation: np.ndarray
    left_gripper_width: float
    right_gripper_width: float


def quaternion_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an ``[qx, qy, qz, qw]`` quaternion to a 3x3 rotation matrix."""
    qx, qy, qz, qw = np.asarray(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm([qx, qy, qz, qw]))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def split_raw_state(state: np.ndarray) -> HandumiRawState:
    """Split one 16D HandUMI raw state into left/right pose and gripper values."""
    arr = np.asarray(state, dtype=np.float32)
    validate_raw_state_shape(arr)

    left_pose = arr[LEFT_POSE_SLICE]
    right_pose = arr[RIGHT_POSE_SLICE]
    return HandumiRawState(
        left_position=left_pose[:3].copy(),
        left_rotation=quaternion_xyzw_to_matrix(left_pose[3:7]),
        right_position=right_pose[:3].copy(),
        right_rotation=quaternion_xyzw_to_matrix(right_pose[3:7]),
        left_gripper_width=float(arr[LEFT_GRIPPER_INDEX]),
        right_gripper_width=float(arr[RIGHT_GRIPPER_INDEX]),
    )


def raw_state_target_poses(
    state: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return ``(left_pose, right_pose)`` tuples compatible with IK solvers."""
    raw = split_raw_state(state)
    return (
        (raw.left_position, raw.left_rotation),
        (raw.right_position, raw.right_rotation),
    )

