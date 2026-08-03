"""Shared live-teleop utilities used by sim, real, and recording frontends."""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable

import numpy as np

from handumi.dataset.raw import pose_to_state_vector
from handumi.feetech import zero_gripper_widths
from handumi.retargeting.handumi_to_robot import VR_TO_ROBOT
from handumi.tracking.transforms import Pose

SIDE_CHOICES = ("left", "right", "both")
# PICO's live tracking stream is 30 Hz. Driving IK faster only retransmits the
# same pose and, because IK limits are expressed per frame, can request joint
# motion faster than a real backend is allowed to stream it.
DEFAULT_TELEOP_FPS = 30
DEFAULT_GRIPPER_SAMPLE_HZ = 200.0


class BestEffortPeriodicWorker:
    """Run disposable peripheral work without blocking the control loop."""

    def __init__(
        self,
        callback: Callable[[], Any],
        *,
        rate_hz: float,
        thread_name: str,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        self.callback = callback
        self.period_s = 1.0 / float(rate_hz)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self.error: BaseException | None = None

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def close(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout_s))

    def _run(self) -> None:
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                self.callback()
                now = time.monotonic()
                next_tick += self.period_s
                if next_tick <= now:
                    # Preview work is disposable; never replay missed frames.
                    next_tick = now + self.period_s
                self._stop.wait(max(0.0, next_tick - now))
        except BaseException as exc:
            self.error = exc
            self._stop.set()


def _normalized_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (value / norm).astype(np.float32)


def _slerp_xyzw(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate quaternions on their shortest arc."""
    first = _normalized_quaternion_xyzw(start)
    second = _normalized_quaternion_xyzw(end)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalized_quaternion_xyzw(first + fraction * (second - first))
    angle = float(np.arccos(dot))
    sine = float(np.sin(angle))
    return _normalized_quaternion_xyzw(
        (np.sin((1.0 - fraction) * angle) / sine) * first
        + (np.sin(fraction * angle) / sine) * second
    )


class TeleopMotionSmoother:
    """Shared causal pose and joint-command low-pass for live teleoperation.

    It filters calibrated controller TCP poses before retargeting/IK, then
    filters the IK command that is rendered, sent to hardware, or recorded.
    The filter advances from timestamps, so every frontend has the same
    behaviour at 30, 60, or 100 Hz.  ``time_constant_s=0`` disables it for
    diagnosis without changing any other part of the pipeline.
    """

    def __init__(
        self,
        time_constant_s: float = 0.0,
        *,
        position_deadband_m: float = 0.0,
        orientation_deadband_rad: float = 0.0,
    ) -> None:
        if time_constant_s < 0.0:
            raise ValueError("time_constant_s must be >= 0.")
        if position_deadband_m < 0.0:
            raise ValueError("position_deadband_m must be >= 0.")
        if orientation_deadband_rad < 0.0:
            raise ValueError("orientation_deadband_rad must be >= 0.")
        self.time_constant_s = float(time_constant_s)
        self.position_deadband_m = float(position_deadband_m)
        self.orientation_deadband_rad = float(orientation_deadband_rad)
        self._source_poses: dict[str, np.ndarray] = {}
        self._last_source_time_ns: int | None = None
        self._joint_q: np.ndarray | None = None
        self._last_joint_time_s: float | None = None

    def reset(self, q: np.ndarray | None = None) -> None:
        self._source_poses.clear()
        self._last_source_time_ns = None
        self._joint_q = None if q is None else np.asarray(q, dtype=np.float32).copy()
        self._last_joint_time_s = None

    def anchor_sources(
        self, source_poses: dict[str, np.ndarray], sides: tuple[str, ...]
    ) -> None:
        """Make a newly anchored controller pose exact (never lagged)."""
        for side in sides:
            if side in source_poses:
                self._source_poses[side] = np.asarray(
                    source_poses[side], dtype=np.float32
                ).copy()

    def smooth_source_poses(
        self,
        source_poses: dict[str, np.ndarray],
        side_tracked: dict[str, bool],
        sample_time_ns: int,
    ) -> dict[str, np.ndarray]:
        """Filter only fresh tracker frames and preserve untracked poses."""
        timestamp = int(sample_time_ns)
        is_fresh = self._last_source_time_ns is None or timestamp > self._last_source_time_ns
        if is_fresh:
            if self._last_source_time_ns is None:
                alpha = 1.0
            else:
                dt_s = min((timestamp - self._last_source_time_ns) * 1e-9, 0.25)
                alpha = self._alpha(dt_s)
            for side, current_value in source_poses.items():
                if not side_tracked.get(side, False):
                    continue
                current = np.asarray(current_value, dtype=np.float32)
                previous = self._source_poses.get(side)
                if previous is None:
                    filtered = current.copy()
                else:
                    position = current[:3]
                    if (
                        np.linalg.norm(position - previous[:3])
                        <= self.position_deadband_m
                    ):
                        position = previous[:3]
                    orientation = current[3:7]
                    dot = abs(
                        float(
                            np.dot(
                                _normalized_quaternion_xyzw(previous[3:7]),
                                _normalized_quaternion_xyzw(orientation),
                            )
                        )
                    )
                    angle = 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))
                    if angle <= self.orientation_deadband_rad:
                        orientation = previous[3:7]
                    if alpha >= 1.0:
                        filtered = current.copy()
                        filtered[:3] = position
                        filtered[3:7] = orientation
                    else:
                        filtered = previous.copy()
                        filtered[:3] += alpha * (position - filtered[:3])
                        filtered[3:7] = _slerp_xyzw(
                            previous[3:7], orientation, alpha
                        )
                self._source_poses[side] = filtered
            self._last_source_time_ns = timestamp

        return {
            side: self._source_poses.get(side, np.asarray(pose, dtype=np.float32)).copy()
            for side, pose in source_poses.items()
        }

    def smooth_joint_command(self, q: np.ndarray, now_s: float) -> np.ndarray:
        """Return the time-normalized filtered joint command."""
        current = np.asarray(q, dtype=np.float32)
        if self._joint_q is None:
            self._joint_q = current.copy()
        else:
            dt_s = (
                0.0
                if self._last_joint_time_s is None
                else min(max(float(now_s) - self._last_joint_time_s, 0.0), 0.25)
            )
            alpha = self._alpha(dt_s)
            self._joint_q = self._joint_q + alpha * (current - self._joint_q)
        self._last_joint_time_s = float(now_s)
        return self._joint_q.copy()

    def restore_joint_command(self, q: np.ndarray) -> None:
        """Forget a computed joint result that was superseded before publish."""
        self._joint_q = np.asarray(q, dtype=np.float32).copy()
        self._last_joint_time_s = None

    def _alpha(self, dt_s: float) -> float:
        if self.time_constant_s == 0.0:
            return 1.0
        return float(1.0 - np.exp(-max(dt_s, 0.0) / self.time_constant_s))


class TeleopLoopTimer:
    """Fixed-rate teleop loop timer with real elapsed command dt."""

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        self.interval = 1.0 / float(fps)
        self._last_start: float | None = None

    def tick(self) -> tuple[float, float]:
        now = time.perf_counter()
        if self._last_start is None:
            dt = self.interval
        else:
            dt = max(now - self._last_start, 1e-6)
        self._last_start = now
        return now, min(dt, 2.0 * self.interval)

    def sleep(self, loop_start: float) -> float:
        elapsed = time.perf_counter() - loop_start
        if (delay := self.interval - elapsed) > 0:
            time.sleep(delay)
        return elapsed


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


def tracking_sample_time_ns(sample: Any) -> int:
    """Stable tracker-frame time for smoothing, preferring device generation time."""
    for value in (
        getattr(sample, "device_time_ns", 0),
        getattr(sample, "aligned_time_ns", 0),
        getattr(sample, "pc_monotonic_ns", 0),
    ):
        if int(value) > 0:
            return int(value)
    return time.monotonic_ns()


def latest_widths(grippers: Any):
    if grippers is None:
        return zero_gripper_widths()
    latest = getattr(grippers, "latest", None)
    if callable(latest):
        sample = latest()
        return zero_gripper_widths() if sample is None else getattr(sample, "widths", zero_gripper_widths())
    return grippers.read_normalized_widths()


def sample_state(sample, widths=None) -> np.ndarray:
    """16D raw state from a live sample's calibrated TCP poses + gripper widths."""
    left = Pose(sample.left_tcp_pose[:3], sample.left_tcp_pose[3:7])
    right = Pose(sample.right_tcp_pose[:3], sample.right_tcp_pose[3:7])
    left_w = 0.0 if widths is None else widths.left
    right_w = 0.0 if widths is None else widths.right
    return pose_to_state_vector(left, right, left_w, right_w)
