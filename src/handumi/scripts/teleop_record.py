"""Record joint-level real-robot teleoperation demonstrations.

This is the recording sibling of ``handumi teleop-real``. The operator drives
the real robot with HandUMI tracking and Feetech gripper widths, while each
LeRobot row stores canonical robot joints directly:

* ``observation.state`` is the robot feedback read from the real backend.
* ``action`` is the next joint command produced by the teleop controller.

Before recording, controller->TCP calibration and Feetech calibration must be
available. Episode control is optimized for continuous real-robot collection:

* opening either enabled gripper: start the waiting episode
* double-squeeze right: save the current episode and start the next one
* double-squeeze left: discard the current episode
* double-squeeze both grippers: discard the active episode and finish the session
* ``Esc`` / ``Ctrl+C``: discard the active episode and stop

Gripper commands are streamed continuously, including while recording is
waiting to start. Opening a HandUMI gripper activates and anchors that arm;
holding the corresponding robot gripper fully closed for three seconds parks it
at home. Episode gestures control recording only and never activate an arm.

PICO tracking uses ADB.

Examples
--------
::

    handumi teleop-record --device pico --robot piper --output-dir outputs/piper-demo
    handumi teleop-record --device pico --robot openarmv1 --output-dir outputs/openarm-demo
    handumi teleop-record --device pico --robot piper --side right --output-dir outputs/right-arm-demo
    handumi teleop-record --device pico --robot piper \
        --output-dir outputs/piper-demo --resume

Common options:

* ``--device``        pico|meta tracking device.
* ``--robot``         Registered real backend, for example piper or openarmv1.
* ``--side``          left|right|both enabled arms.
* ``--fps``           Recording/control frequency in Hz.
* ``--num-episodes``  Number of episodes to record; 0 means until stopped.
* ``--task``          Task description stored in the dataset.
* ``--output-dir``    Destination directory, for example ``outputs/piper-demo``.
* ``--resume``        Append episodes from an existing dataset in that directory.
"""

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Real-time IK favors CPU tail latency; an explicit environment override wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from handumi.cameras import (
    build_camera_specs,
    camera_output_size,
    connect_cameras,
    disconnect_cameras,
    read_camera_samples,
    resolve_camera_ids,
)
from handumi.dataset.canonical import canonical_joint_layout, canonicalize_command
from handumi.dataset.capture import (
    CAMERA_STALE_TIMEOUT_S,
    GRIPPER_STALE_TIMEOUT_S,
    MAX_SYNC_SKEW_S,
    SENSOR_LOSS_TIMEOUT_S,
    SYNC_LAG_S,
    TRACKING_LOSS_TIMEOUT_S,
)
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    camera_health_features,
    capture_timing_features,
    feetech_features,
)
from handumi.feetech import FeetechGripperPair, FeetechGripperSampler, GripperWidths
from handumi.real.registry import make_real_backend
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.scripts.record import (
    StreamingEncodingError,
    _SOFTWARE_VIDEO_CODEC,
    _EscapeStopListener,
    _install_strict_streaming_encoder,
    _prepare_streaming_episode,
    _robot_metadata,
    _select_video_encoder,
    _validate_finalized_lerobot_dataset,
    _write_dataset_readme,
    build_tracker,
    connect_feetech,
)
from handumi.synchronization import (
    SustainedHealthGate,
    capture_timing_frame,
    synchronized_gripper_frame,
)
from handumi.teleop.common import (
    AdaptiveJointFilter,
    BestEffortPeriodicWorker,
    KeyboardSpaceListener,
    TeleopLoopTimer,
)
from handumi.teleop.common import (
    enabled_sides as _enabled_sides,
)
from handumi.teleop.common import (
    enabled_tracking_ok as _enabled_tracking_ok,
)
from handumi.teleop.common import (
    latest_widths as _latest_widths,
)
from handumi.teleop.common import (
    tracking_world_map as _tracking_world_map,
)
from handumi.teleop.core import TeleopController
from handumi.teleop.hardware import (
    load_required_controller_tcp_calibration as _load_required_calibration,
)
from handumi.teleop.hardware import (
    validate_feetech_ready as _validate_feetech_ready,
)
from handumi.teleop.motion import (
    TeleopMotionConfig,
    validate_teleop_motion_args,
)
from handumi.teleop.physical import (
    DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S,
    add_physical_teleop_arguments,
    validate_physical_teleop_args,
)
from handumi.teleop.session import TeleopSession
from handumi.teleop.standby import (
    GRIPPER_PARK_HOLD_S,
    GRIPPER_REOPENED,
    GripperHomeStandby,
)
from handumi.teleop.tracking import LatestTrackingSampler, TrackingRecoveryPolicy
from handumi.teleop.trajectory import TeleopCommandStream
from handumi.tracking.base import TrackingProvider
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say
from handumi.visualize import LiveCameraViews, RerunCameraViewer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
record_log = logging.getLogger("handumi.record_teleop")


class DatasetWriteError(RuntimeError):
    """A background LeRobot write failed or could not keep up."""


class _DatasetRequest:
    def __init__(self, operation: str, payload: Any = None) -> None:
        self.operation = operation
        self.payload = payload
        self.done = threading.Event()
        self.error: BaseException | None = None


class AsyncLeRobotWriter:
    """Serialize LeRobot operations away from the real-time control loop."""

    def __init__(
        self,
        dataset: Any,
        *,
        max_pending_frames: int,
        use_videos: bool,
    ) -> None:
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be > 0")
        self.dataset = dataset
        self.use_videos = bool(use_videos)
        self._queue: queue.Queue[_DatasetRequest] = queue.Queue(
            maxsize=max_pending_frames
        )
        self._episode_error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-lerobot-writer",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    @property
    def pending_frames(self) -> int:
        return self._queue.qsize()

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.raise_if_failed()
        try:
            self._queue.put_nowait(_DatasetRequest("frame", frame))
        except queue.Full as exc:
            raise DatasetWriteError(
                "LeRobot writer queue is full; recording cannot remain frame-aligned"
            ) from exc

    def save_episode(self, expected_frames: int) -> None:
        self._request("save", int(expected_frames))

    def clear_episode(self) -> None:
        self._request("clear")

    def finalize(self) -> None:
        try:
            self._request("finalize")
        finally:
            self._stop_thread()

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._episode_error
        if error is not None:
            raise DatasetWriteError(
                "Background LeRobot writer failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.clear_episode()
        except Exception:
            pass
        self._stop_thread()

    def _request(self, operation: str, payload: Any = None) -> None:
        if self._closed:
            raise DatasetWriteError("LeRobot writer is closed")
        request = _DatasetRequest(operation, payload)
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise DatasetWriteError(
                f"LeRobot writer operation {operation!r} failed"
            ) from request.error

    def _stop_thread(self) -> None:
        if self._closed:
            return
        self._closed = True
        request = _DatasetRequest("stop")
        self._queue.put(request)
        request.done.wait()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            try:
                if request.operation == "stop":
                    return
                if request.operation == "frame":
                    with self._error_lock:
                        episode_error = self._episode_error
                    if episode_error is None:
                        try:
                            self.dataset.add_frame(request.payload)
                        except BaseException as exc:
                            with self._error_lock:
                                self._episode_error = exc
                    continue
                if request.operation == "clear":
                    try:
                        self.dataset.clear_episode_buffer()
                    finally:
                        with self._error_lock:
                            self._episode_error = None
                    continue
                if request.operation == "save":
                    self.raise_if_failed()
                    if self.use_videos:
                        _prepare_streaming_episode(self.dataset, request.payload)
                    self.dataset.save_episode()
                    continue
                if request.operation == "finalize":
                    self.raise_if_failed()
                    self.dataset.finalize()
                    continue
                raise RuntimeError(f"Unknown dataset operation: {request.operation}")
            except BaseException as exc:
                request.error = exc
            finally:
                request.done.set()
                self._queue.task_done()


def _log_episode_interface(
    episode: int,
    total: str,
    *,
    waiting: bool,
    space_start: bool,
    park_hold_s: float,
) -> None:
    state = "WAITING FOR OPEN GRIPPER" if waiting else "RECORDING"
    start = "open either enabled HandUMI gripper"
    if space_start:
        start += " or press SPACE"
    record_log.info(
        "\n"
        "┌─ HandUMI teleop recording "
        "─────────────────────────────\n"
        "│ Episode %d/%s  •  %s\n"
        "│ Start: %s\n"
        "│ Save + next: double-squeeze RIGHT\n"
        "│ Discard: double-squeeze LEFT\n"
        "│ Finish session: double-squeeze BOTH  •  Stop: Esc / Ctrl+C\n"
        "│ Arms: open HandUMI to wake  •  robot gripper closed %.1f s to park\n"
        "└────────────────────────────"
        "─────────────────────────────",
        episode,
        total,
        state,
        start,
        park_hold_s,
    )


class _BilateralClapArbiter:
    """Resolve near-simultaneous side events into one bilateral gesture."""

    def __init__(self, *, bilateral_window_s: float = 0.2) -> None:
        self._bilateral_window_s = bilateral_window_s
        self._pending_side: str | None = None
        self._pending_since_s = 0.0

    def reset(self) -> None:
        self._pending_side = None
        self._pending_since_s = 0.0

    def update(
        self,
        detector: DoubleClapDetector,
        widths: GripperWidths,
        now_s: float,
    ) -> str | None:
        """Return ``left``, ``right`` or ``both`` after chord arbitration."""
        triggered = detector.update_sides(widths.left_mm, widths.right_mm, now_s)
        if len(triggered) == 2:
            self.reset()
            return "both"

        new_side = triggered[0] if triggered else None
        if self._pending_side is not None:
            pending_side = self._pending_side
            deadline_s = self._pending_since_s + self._bilateral_window_s
            if (
                new_side is not None
                and new_side != pending_side
                and now_s <= deadline_s
            ):
                self.reset()
                return "both"
            if now_s >= deadline_s:
                self.reset()
                if new_side is not None:
                    self._pending_side = new_side
                    self._pending_since_s = now_s
                return pending_side

        if new_side is not None:
            self._pending_side = new_side
            self._pending_since_s = now_s
        return None


def _episode_gesture_action(
    gesture: str | None, *, recording: bool
) -> str | None:
    """Map a resolved gripper gesture to the episode state transition."""
    if gesture == "both":
        return "finish"
    if gesture == "right" and recording:
        return "save"
    if gesture == "left" and recording:
        return "discard"
    return None


def _newly_opened_sides(
    previous: dict[str, bool] | None,
    current: dict[str, bool],
    enabled_sides: tuple[str, ...],
) -> tuple[str, ...]:
    """Return closed-to-open edges, ignoring the first observed sample."""
    if previous is None:
        return ()
    return tuple(
        side for side in enabled_sides if current[side] and not previous[side]
    )


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    joint_names: list[str] | tuple[str, ...],
    camera_specs: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    img_dtype = "video" if use_videos else "image"
    features: dict[str, Any] = {}
    specs_by_name = {str(spec["name"]): spec for spec in (camera_specs or [])}
    for cam in cam_names:
        width, height = camera_output_size(
            specs_by_name.get(cam, {}),
            default_width=cam_width,
            default_height=cam_height,
        )
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    state_action = joint_state_feature(joint_names)
    features["observation.state"] = state_action
    features["action"] = dict(state_action)
    features.update(feetech_features())
    features.update(capture_timing_features())
    features.update(camera_health_features(cam_names))
    features["calibration_id"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["calibration_id"],
    }
    features["source_kind"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["source_kind"],
    }
    return features


def joint_state_feature(joint_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    names = list(joint_names)
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }


def build_joint_frame(
    *,
    observation_q: np.ndarray,
    action_q: np.ndarray,
    widths: GripperWidths,
) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.asarray(observation_q, dtype=np.float32).copy(),
        "action": np.asarray(action_q, dtype=np.float32).copy(),
        **_gripper_width_frame(widths),
    }


def _gripper_width_frame(widths: GripperWidths) -> dict[str, np.ndarray]:
    """Build the calibrated Feetech values required by every dataset row."""
    return {
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array(
            [widths.right_ticks], dtype=np.int64
        ),
        "observation.feetech.left_width_mm": np.array(
            [widths.left_mm], dtype=np.float32
        ),
        "observation.feetech.right_width_mm": np.array(
            [widths.right_mm], dtype=np.float32
        ),
        "observation.feetech.left_normalized": np.array(
            [widths.left_normalized], dtype=np.float32
        ),
        "observation.feetech.right_normalized": np.array(
            [widths.right_normalized], dtype=np.float32
        ),
    }


def _gripper_openings(widths: GripperWidths) -> dict[str, float]:
    return {
        "left": float(widths.left_normalized),
        "right": float(widths.right_normalized),
    }


def _parse_record_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    p = argparse.ArgumentParser(
        description="Record real-robot HandUMI teleoperation demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--help-advanced", action="store_true", help="Show expert hardware options."
    )
    add_physical_teleop_arguments(p)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--task", type=str, default="HandUMI real teleop recording")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local dataset directory, for example outputs/piper-demo.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to the finalized dataset in --output-dir.",
    )
    if not show_advanced:
        normal = {
            "help",
            "help_advanced",
            "device",
            "robot",
            "side",
            "space_start",
            "no_sounds",
            "num_episodes",
            "task",
            "output_dir",
            "resume",
            "cameras",
            "skip_cameras",
            "no_rerun",
        }
        for action in p._actions:
            if action.dest not in normal:
                action.help = argparse.SUPPRESS
    else:
        p.print_help()
        raise SystemExit(0)
    args = p.parse_args(raw_argv)
    _apply_recording_defaults(args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_record_args(argv)


def _apply_recording_defaults(args: argparse.Namespace) -> None:
    args.sync_lag_s = SYNC_LAG_S
    args.max_sync_skew_s = MAX_SYNC_SKEW_S
    args.gripper_stale_timeout_s = GRIPPER_STALE_TIMEOUT_S
    args.sensor_loss_timeout_s = SENSOR_LOSS_TIMEOUT_S
    # Match teleop-real: keep serial reads off the control loop and sample at
    # least at 100 Hz (or at the input rate when configured above 100 Hz).
    args.feetech_sample_hz = max(100.0, float(args.fps))
    args.tracking_loss_timeout_s = TRACKING_LOSS_TIMEOUT_S
    args.camera_stale_timeout_s = CAMERA_STALE_TIMEOUT_S
    # PICO defaults to the same ADB transport used by the recording workflow,
    # while an explicit --pico-wifi selection remains available.
    if not args.pico_wifi:
        args.pico_adb = True


def _validate_record_args(args: argparse.Namespace) -> None:
    validate_teleop_motion_args(args)
    validate_physical_teleop_args(args)
    if args.num_episodes < 0:
        raise SystemExit("--num-episodes must be >= 0.")
    for name in (
        "sync_lag_s",
        "max_sync_skew_s",
        "gripper_stale_timeout_s",
        "camera_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
        "tracking_loss_timeout_s",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero.")


def _wait_for_tracking_sampler(
    tracking_sampler: LatestTrackingSampler,
    stop_event: threading.Event,
    *,
    enabled_sides: tuple[str, ...],
    tracking_stale_ms: float,
    poll_s: float = 0.05,
) -> bool:
    """Wait for a fresh sample without polling the tracking SDK inline."""
    last_report = float("-inf")
    while not stop_event.is_set():
        now = time.monotonic()
        snapshot = tracking_sampler.latest()
        if snapshot is not None:
            sample = snapshot.sample
            side_tracked = {
                "left": sample.left_tracked,
                "right": sample.right_tracked,
            }
            if snapshot.age_s(
                now
            ) <= tracking_stale_ms / 1000.0 and _enabled_tracking_ok(
                side_tracked, enabled_sides
            ):
                record_log.info("Enabled controllers tracked; recording gate open.")
                return True
        if now - last_report >= 2.0:
            record_log.warning("Waiting for fresh controller tracking ...")
            last_report = now
        stop_event.wait(poll_s)
    return False


def record_episode(
    *,
    tracker: TrackingProvider,
    tracking_sampler: LatestTrackingSampler,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    real_env,
    controller: TeleopController,
    runtime,
    home_q: np.ndarray,
    enabled_sides: tuple[str, ...],
    space_listener: KeyboardSpaceListener,
    clap_detector: DoubleClapDetector,
    clap_arbiter: _BilateralClapArbiter,
    fps: int,
    task: str,
    stop_event: threading.Event,
    play_sounds: bool,
    initial_start_sides: tuple[str, ...],
    sync_lag_s: float,
    max_sync_skew_s: float,
    gripper_stale_timeout_s: float,
    sensor_loss_timeout_s: float,
    tracking_loss_timeout_s: float,
    tracking_stale_ms: float,
    command_stream: TeleopCommandStream,
    park_hold_s: float = GRIPPER_PARK_HOLD_S,
    park_max_joint_speed_deg_s: float = DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S,
    joint_filter: AdaptiveJointFilter | None = None,
    home_standby: GripperHomeStandby | None = None,
    dataset_writer: AsyncLeRobotWriter | None = None,
    cameras: list[Any] | None = None,
    camera_names: list[str] | None = None,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_stale_timeout_s: float = CAMERA_STALE_TIMEOUT_S,
    episode_number: int = 1,
    episode_total: str = "?",
) -> tuple[np.ndarray, np.ndarray, int, str, np.ndarray]:
    loop_timer = TeleopLoopTimer(fps)
    n_frames = 0
    start_t: float | None = None
    episode_start_ns: int | None = None
    status = "recorded"
    del initial_start_sides
    tracking_recovery = TrackingRecoveryPolicy()
    health_gate = SustainedHealthGate(sensor_loss_timeout_s)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)
    sync_lag_ns = int(sync_lag_s * 1e9)
    q = controller.q.copy()
    if joint_filter is None:
        finger_indices = {
            finger.index
            for fingers in (runtime.finger_joints or {}).values()
            for finger in fingers
        }
        indices = tuple(
            dict.fromkeys(
                index
                for side in enabled_sides
                for index in runtime.arm_joint_indices(side)
                if index not in finger_indices
            )
        )
        joint_filter = TeleopMotionConfig(input_rate_hz=float(fps)).make_joint_filter(
            filtered_indices=indices
        )
    joint_filter.reset(q)
    teleop_session = TeleopSession(controller, joint_filter)
    if home_standby is None:
        home_standby = GripperHomeStandby(initial_standby=True)
    observations: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    last_processed_tracking_time_ns: int | None = None
    timing_next_log_s = time.perf_counter() + 5.0
    timing_ik_total_s = 0.0
    timing_ik_max_s = 0.0
    timing_ik_samples = 0
    timing_tracking_age_max_s = 0.0
    timing_discarded_ik = 0
    pending_dataset_frame: dict[str, Any] | None = None
    previous_episode_open: dict[str, bool] | None = None
    cameras = cameras or []
    camera_names = camera_names or []

    while True:
        loop_start, _ = loop_timer.tick()
        record_time_ns = time.monotonic_ns()

        if stop_event.is_set():
            status = "interrupted"
            observations.clear()
            commands.clear()
            break

        tracking_snapshot = tracking_sampler.latest()
        if tracking_snapshot is None:
            immediate_widths = _latest_widths(grippers)
            command_stream.update_openings(_gripper_openings(immediate_widths))
            space_listener.consume_space()
            clap_detector.update_sides(
                immediate_widths.left_mm, immediate_widths.right_mm, loop_start
            )
            clap_arbiter.reset()
            loop_timer.sleep(loop_start)
            continue
        sample = tracking_snapshot.sample
        tracking_age_s = tracking_snapshot.age_s(loop_start)
        timing_tracking_age_max_s = max(timing_tracking_age_max_s, tracking_age_s)
        tracking_stale = tracking_age_s > tracking_stale_ms / 1000.0
        side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
        tracking_ok = not tracking_stale and _enabled_tracking_ok(
            side_tracked, enabled_sides
        )

        if not controller.active and not tracking_ok:
            tracking_recovery.reset()
            immediate_widths = _latest_widths(grippers)
            command_stream.update_openings(_gripper_openings(immediate_widths))
            space_listener.consume_space()
            clap_detector.update_sides(
                immediate_widths.left_mm, immediate_widths.right_mm, loop_start
            )
            clap_arbiter.reset()
            loop_timer.sleep(loop_start)
            continue

        if not tracking_ok:
            immediate_widths = _latest_widths(grippers)
            if tracking_recovery.note_missing(
                loop_start,
                observed_since=(
                    tracking_snapshot.fresh_at_s if tracking_stale else None
                ),
            ):
                command_stream.stop()
                held = real_env.hold(q)
                controller.tracking_lost(held)
                home_standby.enter_standby(enabled_sides)
                joint_filter.reset(held)
                q = held
                command_stream.submit(
                    held,
                    _gripper_openings(immediate_widths),
                    time_s=loop_start,
                    active=True,
                    new_epoch=True,
                )
                record_log.warning(
                    "Tracking lost%s; robot command held and episode discarded.",
                    (
                        f" (sample age {tracking_age_s * 1000.0:.0f} ms)"
                        if tracking_stale
                        else ""
                    ),
                )
                log_say("tracking lost", play_sounds=play_sounds)
            if (
                tracking_recovery.lost
                and tracking_recovery.lost_for(loop_start) >= tracking_loss_timeout_s
            ):
                status = "tracking_lost"
                observations.clear()
                commands.clear()
                break
            recover = getattr(tracker, "recover", None)
            if callable(recover) and tracking_recovery.should_recover(loop_start):
                acquired, recovered = tracking_sampler.try_source_call(recover)
                if acquired and recovered:
                    record_log.info(
                        "Tracking recovered; open the HandUMI gripper to re-anchor."
                    )
                    log_say("tracking recovered", play_sounds=play_sounds)
            command_stream.update_openings(_gripper_openings(immediate_widths))
            loop_timer.sleep(loop_start)
            continue
        if tracking_recovery.lost:
            record_log.info("Tracking stream recovered; checking gripper state.")
        tracking_recovery.reset()

        immediate_widths = _latest_widths(grippers)
        start_sides: tuple[str, ...] = ()
        if space_listener.consume_space():
            # Space remains the arm-start fallback when Feetech input was
            # explicitly disabled; with grippers present, it is an explicit
            # recording-start override and does not activate either arm.
            if grippers is None:
                start_sides = controller.idle_sides()
            if start_t is None:
                start_t = loop_start
                episode_start_ns = record_time_ns
                home_standby.reset_close_timers()
                record_log.info("Recording episode started with Space.")
                log_say("recording episode", play_sounds=play_sounds)
        clap_gesture = clap_arbiter.update(
            clap_detector, immediate_widths, loop_start
        )
        gesture_action = _episode_gesture_action(
            clap_gesture, recording=start_t is not None
        )
        if clap_gesture is not None:
            record_log.info(
                "Resolved double clap: %s%s.",
                clap_gesture,
                " (episode not recording)" if start_t is None else "",
            )
        if gesture_action == "save":
            status = "recorded"
            break
        elif gesture_action == "discard":
            status = "discarded"
            observations.clear()
            commands.clear()
            break
        elif gesture_action == "finish":
            status = "session_finished"
            observations.clear()
            commands.clear()
            break

        inputs = teleop_session.inputs(sample, immediate_widths)
        current_episode_open = {
            side: bool(
                inputs.side_tracked[side]
                and inputs.openings[side] >= GRIPPER_REOPENED
            )
            for side in enabled_sides
        }
        if start_t is None:
            opened_sides = _newly_opened_sides(
                previous_episode_open,
                current_episode_open,
                enabled_sides,
            )
            if opened_sides:
                start_t = loop_start
                episode_start_ns = record_time_ns
                home_standby.reset_close_timers()
                record_log.info(
                    "● REC | episode %d/%s | started by opening %s gripper | "
                    "0 frames.",
                    episode_number,
                    episode_total,
                    "/".join(opened_sides),
                )
                log_say("recording episode", play_sounds=play_sounds)
        previous_episode_open = current_episode_open
        # The park timer observes the command actually handed to the robot,
        # not the raw HandUMI opening. The HandUMI opening is used only to wake
        # an already parked arm (and independently for episode gestures).
        played_command = command_stream.latest()
        if grippers is not None and played_command is not None:
            _, robot_openings = played_command
            park_sides, wake_sides = home_standby.update(
                robot_openings,
                loop_start,
                enabled_sides,
                wake_openings=inputs.openings,
            )
        else:
            park_sides, wake_sides = (), ()
        if park_sides:
            parked = controller.park(park_sides)
            if parked:
                park_indices = tuple(
                    index
                    for index in joint_filter.filtered_indices or ()
                    if any(index in controller.side_indices[side] for side in parked)
                )
                command_stream.limit_joint_rates(
                    park_indices,
                    np.deg2rad(park_max_joint_speed_deg_s),
                )
                record_log.info(
                    "Robot %s gripper held closed for %.1fs; arm returning home "
                    "and entering standby while recording continues.",
                    "/".join(parked),
                    park_hold_s,
                )
                for side in parked:
                    log_say(f"{side} arm standby", play_sounds=play_sounds)
        if wake_sides:
            wake_indices = tuple(
                index
                for index in joint_filter.filtered_indices or ()
                if any(index in controller.side_indices[side] for side in wake_sides)
            )
            command_stream.clear_joint_rate_limits(wake_indices)
            start_sides += tuple(
                side for side in wake_sides if side not in start_sides
            )
            record_log.info(
                "HandUMI %s gripper open; waking and re-anchoring arm.",
                "/".join(wake_sides),
            )
        fresh_tracking = (
            last_processed_tracking_time_ns is None
            or tracking_snapshot.source_time_ns != last_processed_tracking_time_ns
        )
        if not fresh_tracking and not start_sides and not park_sides:
            command_stream.update_openings(inputs.openings)
            real_env.check_health()
            loop_timer.sleep(loop_start)
            continue

        observation_q = real_env.read(base_q=q)
        controller_q_before_ik = controller.q.copy()
        filter_before_ik = teleop_session.snapshot_filter()
        ik_start_s = time.perf_counter()
        teleop_frame = teleop_session.advance(
            inputs,
            now_s=loop_start,
            start_sides=start_sides,
            exact_sides=park_sides,
        )
        ik_elapsed_s = time.perf_counter() - ik_start_s
        timing_ik_total_s += ik_elapsed_s
        timing_ik_max_s = max(timing_ik_max_s, ik_elapsed_s)
        timing_ik_samples += 1
        newest_after_ik = tracking_sampler.latest()
        if (
            newest_after_ik is not None
            and newest_after_ik.source_time_ns != tracking_snapshot.source_time_ns
            and not start_sides
            and not park_sides
        ):
            controller.q = controller_q_before_ik
            teleop_session.restore_filter(filter_before_ik)
            timing_discarded_ik += 1
            real_env.check_health()
            continue
        anchored = teleop_frame.anchored_sides
        if anchored:
            record_log.info(
                "Teleop arm anchored after opening %s.", "/".join(anchored)
            )

        action_q = teleop_frame.q
        command_stream.submit(
            action_q,
            teleop_frame.inputs.openings,
            time_s=tracking_snapshot.fresh_at_s,
            active=controller.active or bool(park_sides),
            new_epoch=bool(anchored),
        )
        # Grippers remain live even when every arm is parked. Arm activity
        # only controls IK motion, never gripper communication.
        command_stream.update_openings(teleop_frame.inputs.openings)
        last_processed_tracking_time_ns = tracking_snapshot.source_time_ns
        real_env.check_health()
        q = action_q

        timing_now_s = time.perf_counter()
        if command_stream.running and timing_now_s >= timing_next_log_s:
            output_stats = command_stream.stats()
            elapsed_s = max(0, int(timing_now_s - (start_t or timing_now_s)))
            queued_frames = (
                dataset_writer.pending_frames if dataset_writer is not None else 0
            )
            record_log.info(
                "%s | episode %d/%s | %02d:%02d | %d frames | "
                "writer_queue=%d | output=%.1f Hz, missed=%d, "
                "tracking_age_max=%.0f ms, IK_avg/max=%.1f/%.1f ms, "
                "backend_write_max=%.1f ms, stale_IK_discarded=%d.",
                "● REC" if start_t is not None else "○ READY",
                episode_number,
                episode_total,
                elapsed_s // 60,
                elapsed_s % 60,
                n_frames,
                queued_frames,
                output_stats.effective_rate_hz,
                output_stats.missed_deadlines,
                timing_tracking_age_max_s * 1000.0,
                (
                    timing_ik_total_s / timing_ik_samples * 1000.0
                    if timing_ik_samples
                    else 0.0
                ),
                timing_ik_max_s * 1000.0,
                output_stats.max_write_ms,
                timing_discarded_ik,
            )
            timing_next_log_s = timing_now_s + 5.0
            timing_ik_total_s = 0.0
            timing_ik_max_s = 0.0
            timing_ik_samples = 0
            timing_tracking_age_max_s = 0.0
            timing_discarded_ik = 0

        played_command = command_stream.latest()
        if played_command is None:
            played_action_q = action_q
            played_openings = teleop_frame.inputs.openings
        else:
            played_action_q, played_openings = played_command

        if start_t is None:
            loop_timer.sleep(loop_start)
            continue

        assert episode_start_ns is not None
        target_time_ns = max(episode_start_ns, record_time_ns - sync_lag_ns)
        camera_frame, camera_health = read_camera_samples(
            cameras,
            camera_names,
            target_time_ns=target_time_ns,
            record_time_ns=record_time_ns,
            width=camera_width,
            height=camera_height,
            stale_timeout_s=camera_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        gripper_frame = synchronized_gripper_frame(
            grippers,
            target_time_ns=target_time_ns,
            record_time_ns=record_time_ns,
            stale_timeout_s=gripper_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        tracking_time_ns = int(sample.aligned_time_ns or sample.pc_monotonic_ns)
        tracking_sync_ok = bool(
            tracking_time_ns > 0
            and abs(tracking_time_ns - target_time_ns) <= max_sync_skew_ns
        )
        sensor_health = {
            **camera_health,
            "feetech": gripper_frame.healthy_for_gate,
            "tracking": tracking_sync_ok,
        }
        _, timed_out_sensors = health_gate.update(sensor_health, record_time_ns)
        if timed_out_sensors:
            status = "sensor_unhealthy"
            record_log.error(
                "Sensor health timed out: %s.",
                ", ".join(sorted(timed_out_sensors)),
            )
            observations.clear()
            commands.clear()
            break

        canonical_observation = canonicalize_command(
            observation_q,
            runtime=runtime,
            openings={
                "left": gripper_frame.widths.left_normalized,
                "right": gripper_frame.widths.right_normalized,
            },
        )
        canonical_action = canonicalize_command(
            played_action_q,
            runtime=runtime,
            openings=played_openings,
        )
        if dataset_writer is not None and pending_dataset_frame is not None:
            try:
                dataset_writer.add_frame(
                    {
                        **pending_dataset_frame,
                        "action": canonical_action,
                        "task": task,
                    }
                )
            except (DatasetWriteError, StreamingEncodingError) as exc:
                status = "dataset_unhealthy"
                record_log.error(
                    "Dataset frame writer failed; discarding episode: %s", exc
                )
                observations.clear()
                commands.clear()
                break
        pending_dataset_frame = {
            **camera_frame,
            "observation.state": canonical_observation,
            **_gripper_width_frame(gripper_frame.widths),
            **gripper_frame.frame,
            **capture_timing_frame(target_time_ns, record_time_ns),
            "calibration_id": np.array([-1], dtype=np.int64),
            "source_kind": np.array([1], dtype=np.int64),
        }
        observations.append(canonical_observation)
        commands.append(canonical_action)
        n_frames += 1
        loop_timer.sleep(loop_start)

    if len(observations) < 2:
        return (
            np.empty((0, canonical_joint_layout(runtime).size), dtype=np.float32),
            np.empty((0, canonical_joint_layout(runtime).size), dtype=np.float32),
            n_frames,
            status,
            q,
        )
    states = np.asarray(observations[:-1], dtype=np.float32)
    actions = np.asarray(commands[1:], dtype=np.float32)
    return states, actions, len(states), status, q


def _run_record() -> None:
    args = _parse_record_args()
    _validate_record_args(args)
    args.output_dir = Path(args.output_dir)
    args.repo_id = f"local/{args.output_dir.name}"
    if args.resume:
        _validate_resume_dataset(args.output_dir)
    play_sounds = not args.no_sounds
    stop_event = threading.Event()

    record_log.info("Loading %s IK solver.", args.robot)
    runtime = load_embodiment(args.robot)
    import jax

    record_log.info("IK compute platform: %s.", jax.default_backend())
    try:
        home_pose_name, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    enabled_sides = _enabled_sides(args.side)
    controller = TeleopController(
        runtime,
        home_q=home_q,
        enabled_sides=enabled_sides,
        source_world_to_robot_world=_tracking_world_map(args.device),
        translation_scale=args.translation_scale,
    )
    controller.warmup()
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    real_env = make_real_backend(
        args.robot,
        runtime=runtime,
        rig_config=args.rig_config,
        active_sides=enabled_sides,
    )
    gripper_pair = None
    grippers = None
    tracker_started = False
    tracking_sampler: LatestTrackingSampler | None = None
    cameras: list[Any] = []
    camera_names = [] if args.skip_cameras else list(args.cameras)
    camera_specs: list[dict[str, Any]] = []
    camera_views: LiveCameraViews | None = None
    camera_worker: BestEffortPeriodicWorker | None = None
    dataset = None
    dataset_writer: AsyncLeRobotWriter | None = None
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    motion_config = TeleopMotionConfig.from_args(args)
    finger_indices = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    motion_joint_indices = tuple(
        dict.fromkeys(
            index
            for side in enabled_sides
            for index in runtime.arm_joint_indices(side)
            if index not in finger_indices
        )
    )
    joint_filter = motion_config.make_joint_filter(
        filtered_indices=motion_joint_indices
    )
    command_stream = motion_config.make_command_stream(real_env.write)

    def _on_signal(signum, frame):
        del signum, frame
        record_log.info("Signal received - discarding active episode and stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    escape_listener = _EscapeStopListener(stop_event)
    escape_listener.start()

    try:
        record_log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        tracking_sampler = LatestTrackingSampler(
            tracker.latest,
            sample_rate_hz=motion_config.input_rate_hz,
        )
        tracking_sampler.start()
        if camera_names:
            camera_ids = resolve_camera_ids(
                None,
                args.rig_config,
                camera_names=camera_names,
            )
            if len(camera_ids) != len(set(camera_ids)):
                mappings = ", ".join(
                    f"{name}={camera_id}"
                    for name, camera_id in zip(
                        camera_names, camera_ids, strict=True
                    )
                )
                raise SystemExit(
                    f"Selected cameras must use distinct devices ({mappings})."
                )
            camera_specs, _ = build_camera_specs(
                camera_ids,
                camera_names=camera_names,
                laptop_camera=False,
                laptop_cam_id=0,
                laptop_cam_name="laptop",
                rig_config=args.rig_config,
                default_fps=args.cam_fps,
                default_width=args.cam_width,
                default_height=args.cam_height,
            )
            cameras = connect_cameras(
                camera_specs,
                fps=args.cam_fps,
                width=args.cam_width,
                height=args.cam_height,
                zero_non_laptop=False,
            )
        if cameras and not args.no_rerun:
            viewer = RerunCameraViewer(
                camera_names,
                application_id="handumi_teleop_record",
            )
            viewer.start()
            camera_views = LiveCameraViews(
                cameras=cameras,
                camera_names=camera_names,
                width=args.cam_width,
                height=args.cam_height,
                viewer=viewer,
            )
            camera_worker = BestEffortPeriodicWorker(
                camera_views.update,
                rate_hz=args.cam_fps,
                thread_name="handumi-record-camera-preview",
            )
            camera_worker.start()
        gripper_pair = connect_feetech(args)
        if gripper_pair is not None:
            grippers = FeetechGripperSampler(
                gripper_pair,
                sample_hz=args.feetech_sample_hz,
            )
            grippers.start()

        real_env.setup(repair=not args.skip_can_repair)
        real_env.connect()
        record_log.info("Selected home pose: %s", home_pose_name)
        real_env.home(home_q)
        joint_filter.reset(home_q)
        startup_widths = _latest_widths(grippers)
        command_stream.submit(
            home_q,
            _gripper_openings(startup_widths),
            time_s=time.perf_counter(),
            active=True,
            new_epoch=True,
        )
        record_log.info(
            "Continuous gripper synchronization started; arms remain at home "
            "until their HandUMI gripper opens."
        )
        record_log.info(
            "Joint trajectory playback: %.1f Hz, %.0f ms delay, "
            "%.0f ms max bridge, %.0f ms EMA; adaptive IK filter "
            "cutoff=%.1f Hz + %.1f*|dq/dt|.",
            args.command_rate_hz,
            args.trajectory_delay_ms,
            args.max_extrapolation_ms,
            args.motion_smoothing_time_constant_s * 1000.0,
            args.joint_filter_min_cutoff_hz,
            args.joint_filter_velocity_coefficient,
        )
        space_listener.start()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        layout = canonical_joint_layout(runtime)
        use_videos = bool(camera_names)
        features = build_features(
            camera_names,
            args.cam_width,
            args.cam_height,
            use_videos,
            layout.names,
            camera_specs=camera_specs,
        )
        encoder_selection = None
        dataset_vcodec = _SOFTWARE_VIDEO_CODEC
        if use_videos:
            output_sizes = [
                camera_output_size(
                    spec,
                    default_width=args.cam_width,
                    default_height=args.cam_height,
                )
                for spec in camera_specs
            ]
            encoder_selection = _select_video_encoder(
                policy="auto",
                requested_vcodec=None,
                width=max(width for width, _ in output_sizes),
                height=max(height for _, height in output_sizes),
                fps=args.fps,
                camera_count=len(camera_names),
                requested_threads=None,
            )
            dataset_vcodec = encoder_selection.vcodec
            record_log.info(
                "LeRobot video encoder: %s (%s).",
                dataset_vcodec,
                "hardware" if encoder_selection.hardware else "CPU",
            )
        dataset_kwargs = {
            "repo_id": args.repo_id,
            "root": args.output_dir,
            "image_writer_processes": 0,
            "image_writer_threads": 0 if use_videos else max(1, 4 * len(camera_names)),
            "vcodec": dataset_vcodec,
            "streaming_encoding": use_videos,
            "encoder_queue_maxsize": max(1, args.fps),
            "encoder_threads": (
                encoder_selection.threads if encoder_selection is not None else None
            ),
        }
        if args.resume:
            dataset = LeRobotDataset.resume(**dataset_kwargs)
        else:
            dataset = LeRobotDataset.create(
                **dataset_kwargs,
                fps=args.fps,
                robot_type=runtime.config.kind,
                features=features,
                use_videos=use_videos,
            )
        if use_videos:
            _install_strict_streaming_encoder(dataset)
        dataset_writer = AsyncLeRobotWriter(
            dataset,
            max_pending_frames=max(2, args.fps * 2),
            use_videos=use_videos,
        )
        existing_episodes = int(dataset.num_episodes)
        record_log.info("Recording LeRobot dataset at: %s", dataset.root)

        clap_detector = DoubleClapDetector()
        clap_arbiter = _BilateralClapArbiter()
        home_standby = GripperHomeStandby(
            hold_s=args.gripper_park_hold_s,
            initial_standby=True,
        )
        recorded = 0
        while (
            args.num_episodes <= 0 or recorded < args.num_episodes
        ) and not stop_event.is_set():
            ep_num = existing_episodes + recorded + 1
            ep_total = (
                "inf"
                if args.num_episodes <= 0
                else str(existing_episodes + args.num_episodes)
            )
            if not _wait_for_tracking_sampler(
                tracking_sampler,
                stop_event,
                enabled_sides=enabled_sides,
                tracking_stale_ms=args.tracking_stale_ms,
            ):
                break
            record_log.info("--- Episode %d/%s ---", ep_num, ep_total)
            _log_episode_interface(
                ep_num,
                ep_total,
                waiting=True,
                space_start=args.space_start,
                park_hold_s=args.gripper_park_hold_s,
            )
            if not args.space_start:
                record_log.info(
                    "  Open either enabled gripper to start episode %d ...",
                    ep_num,
                )
            else:
                record_log.info(
                    "  Press Space%s to start episode %d ...",
                    " or open either gripper" if not args.skip_feetech else "",
                    ep_num,
                )
            clap_arbiter.reset()
            clap_detector.reset()
            _, _, n_frames, status, _ = record_episode(
                tracker=tracker,
                tracking_sampler=tracking_sampler,
                grippers=grippers,
                real_env=real_env,
                controller=controller,
                runtime=runtime,
                home_q=home_q,
                enabled_sides=enabled_sides,
                space_listener=space_listener,
                clap_detector=clap_detector,
                clap_arbiter=clap_arbiter,
                fps=args.fps,
                task=args.task,
                stop_event=stop_event,
                play_sounds=play_sounds,
                initial_start_sides=(),
                sync_lag_s=args.sync_lag_s,
                max_sync_skew_s=args.max_sync_skew_s,
                gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                tracking_loss_timeout_s=args.tracking_loss_timeout_s,
                tracking_stale_ms=args.tracking_stale_ms,
                command_stream=command_stream,
                park_hold_s=args.gripper_park_hold_s,
                park_max_joint_speed_deg_s=args.park_max_joint_speed_deg_s,
                joint_filter=joint_filter,
                home_standby=home_standby,
                dataset_writer=dataset_writer,
                cameras=cameras,
                camera_names=camera_names,
                camera_width=args.cam_width,
                camera_height=args.cam_height,
                camera_stale_timeout_s=args.camera_stale_timeout_s,
                episode_number=ep_num,
                episode_total=ep_total,
            )
            if status == "session_finished":
                record_log.info(
                    "Bilateral double clap detected; finishing recording session."
                )
                dataset_writer.clear_episode()
                log_say("Recording session finished", play_sounds=play_sounds)
                break
            if status == "discarded":
                record_log.warning(
                    "Episode discarded by left double clap (%d frames).", n_frames
                )
                dataset_writer.clear_episode()
                log_say("Episode discarded", play_sounds=play_sounds)
                continue
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "dataset_unhealthy",
                "encoder_unhealthy",
                "interrupted",
            }:
                record_log.warning(
                    "Episode discarded (%s, %d frames).", status, n_frames
                )
                dataset_writer.clear_episode()
                log_say("Episode discarded", play_sounds=play_sounds)
                if status == "interrupted":
                    break
                continue
            try:
                dataset_writer.save_episode(n_frames)
            except DatasetWriteError as exc:
                record_log.error("Episode discarded before commit: %s", exc)
                dataset_writer.clear_episode()
                log_say("Episode discarded", play_sounds=play_sounds)
                continue
            recorded += 1
            record_log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(
                f"Episode {ep_num} saved, {n_frames} frames",
                play_sounds=play_sounds,
            )

        dataset_writer.finalize()
        from handumi.dataset import update_handumi_metadata

        updated_info = update_handumi_metadata(
            dataset.root,
            {
                "recording_device": args.device,
                "capture_schema": HANDUMI_CAPTURE_SCHEMA,
                "state_layout": "yaml_arm_joints_plus_logical_gripper_width_m",
                "state_semantics": "real_robot_joint_feedback",
                "action_semantics": "next_step_teleop_joint_command",
                "trajectory_command_rate_hz": args.command_rate_hz,
                "trajectory_delay_ms": args.trajectory_delay_ms,
                "trajectory_max_extrapolation_ms": args.max_extrapolation_ms,
                "trajectory_ema_time_constant_s": (
                    args.motion_smoothing_time_constant_s
                ),
                "joint_filter_min_cutoff_hz": args.joint_filter_min_cutoff_hz,
                "joint_filter_velocity_coefficient": (
                    args.joint_filter_velocity_coefficient
                ),
                "joint_filter_derivative_cutoff_hz": (
                    args.joint_filter_derivative_cutoff_hz
                ),
                "gripper_park_hold_s": args.gripper_park_hold_s,
                "park_max_joint_speed_deg_s": args.park_max_joint_speed_deg_s,
                "translation_scale": args.translation_scale,
                "observation_action_alignment": (
                    "observation.state[t] is canonical backend feedback; "
                    "action[t] is the next recorded teleop command."
                ),
                "source_kind_ids": {"converted": 0, "teleop": 1, "unknown": -1},
                "calibration_id_semantics": (
                    "-1 means no per-episode calibration artifact is referenced"
                ),
                "sync_lag_s": args.sync_lag_s,
                "max_sync_skew_s": args.max_sync_skew_s,
                "joint_names": layout.names,
                "target_robot": _robot_metadata(args.robot),
                "repo_id": args.repo_id,
                "video_keys": [
                    f"observation.images.{name}" for name in camera_names
                ],
            },
        )
        dataset.meta.info = updated_info
        _write_dataset_readme(
            Path(dataset.root),
            repo_id=args.repo_id,
            task=args.task,
            license_id="other",
        )
        if existing_episodes + recorded > 0:
            _validate_finalized_lerobot_dataset(Path(dataset.root))
            record_log.info("LeRobot v3 integrity validation passed.")
        else:
            record_log.warning(
                "Session finished without completed episodes; skipping dataset "
                "integrity validation."
            )
        record_log.info(
            "Done. Recorded %d episode(s). Dataset at: %s", recorded, args.output_dir
        )
    finally:
        escape_listener.stop()
        space_listener.close()
        try:
            command_stream.stop()
        finally:
            try:
                if dataset_writer is not None:
                    dataset_writer.close()
                real_env.disconnect()
            finally:
                if grippers is not None:
                    grippers.stop()
                if gripper_pair is not None:
                    gripper_pair.close()
                if tracker_started:
                    if tracking_sampler is not None:
                        tracking_sampler.stop()
                    tracker.stop()
                if camera_worker is not None:
                    camera_worker.close()
                if camera_views is not None:
                    camera_views.close()
                elif cameras:
                    disconnect_cameras(cameras)
                log_say("Exiting", play_sounds=play_sounds, blocking=True)


def _existing_episode_count(root: Path) -> int:
    info_path = Path(root) / "meta" / "info.json"
    return int(json.loads(info_path.read_text()).get("total_episodes", 0))


def _validate_resume_dataset(root: Path) -> None:
    """Require a finalized vector dataset before appending teleop episodes."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    required_paths = (
        root / "meta" / "episodes",
        root / "meta" / "tasks.parquet",
        root / "data",
    )
    if not root.is_dir():
        raise SystemExit(f"Cannot resume: dataset directory does not exist: {root}")
    if not all(path.exists() for path in required_paths):
        raise SystemExit(f"Cannot resume incomplete dataset at {root}.")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot resume: invalid {info_path}: {exc}") from exc
    if not isinstance(info, dict) or "total_episodes" not in info:
        raise SystemExit(f"Cannot resume: {info_path} is not a HandUMI dataset.")


def main() -> None:
    _run_record()


if __name__ == "__main__":
    main()
