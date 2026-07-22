"""Shared live-teleop utilities used by sim, real, and recording frontends."""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from typing import Any

import numpy as np

from handumi.dataset.raw import pose_to_state_vector
from handumi.feetech import zero_gripper_widths
from handumi.retargeting.handumi_to_robot import VR_TO_ROBOT
from handumi.tracking.transforms import Pose

SIDE_CHOICES = ("left", "right", "both")


class KeyboardSpaceListener:
    """Non-blocking Space listener for terminal-triggered teleop starts."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self._space = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-teleop-space",
            daemon=True,
        )
        self._thread.start()

    def consume_space(self) -> bool:
        if not self._space.is_set():
            return False
        self._space.clear()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                if char == " ":
                    self._space.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def enabled_sides(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("left", "right")
    return (side,)


def start_sides(
    anchors: dict[str, dict[str, np.ndarray] | None],
    enabled: tuple[str, ...],
) -> tuple[str, ...]:
    """Return enabled arms that are idle and can be started from Space."""
    return tuple(side for side in enabled if anchors[side] is None)


def tracking_world_map(device: str) -> np.ndarray:
    """Map provider TCP world axes into robot-world axes."""
    return VR_TO_ROBOT if device == "pico" else np.eye(3, dtype=np.float32)


def tracking_ready_for_sides(
    source_poses: dict[str, np.ndarray],
    side_tracked: dict[str, bool],
    enabled: tuple[str, ...],
) -> bool:
    """Require a real finite controller pose for every arm being auto-started."""
    return all(
        side_tracked[side]
        and np.isfinite(source_poses[side]).all()
        and float(np.linalg.norm(source_poses[side][:3])) > 1e-6
        for side in enabled
    )


def enabled_tracking_ok(
    side_tracked: dict[str, bool],
    enabled: tuple[str, ...],
) -> bool:
    return all(side_tracked[side] for side in enabled)


def latest_widths(grippers: Any):
    return (
        zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
    )


def sample_state(sample, widths=None) -> np.ndarray:
    """16D raw state from a live sample's calibrated TCP poses + gripper widths."""
    left = Pose(sample.left_tcp_pose[:3], sample.left_tcp_pose[3:7])
    right = Pose(sample.right_tcp_pose[:3], sample.right_tcp_pose[3:7])
    left_w = 0.0 if widths is None else widths.left
    right_w = 0.0 if widths is None else widths.right
    return pose_to_state_vector(left, right, left_w, right_w)
