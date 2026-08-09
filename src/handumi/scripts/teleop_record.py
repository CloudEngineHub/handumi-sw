"""Record joint-level real-robot teleoperation demonstrations.

This is the recording sibling of ``handumi teleop-real``. The operator drives
the real robot with HandUMI tracking and Feetech gripper widths, while each
LeRobot row stores canonical robot joints directly:

* ``observation.state`` is the robot feedback read from the real backend.
* ``action`` is the next joint command produced by the teleop controller.

Before recording, controller->TCP calibration and Feetech calibration must be
available. Episode control is optimized for continuous real-robot collection:

* double-squeeze left: start the first or replacement episode
* double-squeeze right: save the current episode and start the next one
* double-squeeze both grippers: discard the current episode
* ``Esc`` / ``Ctrl+C``: discard the active episode and stop

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
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Real-time IK favors CPU tail latency; an explicit environment override wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from handumi.dataset.canonical import canonical_joint_layout, canonicalize_command
from handumi.dataset.capture import (
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
    _EscapeStopListener,
    _robot_metadata,
    build_tracker,
    connect_feetech,
)
from handumi.synchronization import (
    SustainedHealthGate,
    synchronized_gripper_frame,
)
from handumi.teleop.common import (
    BestEffortPeriodicWorker,
    KeyboardSpaceListener,
    TeleopLoopTimer,
    TeleopMotionSmoother,
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
    add_physical_teleop_arguments,
    validate_physical_teleop_args,
)
from handumi.teleop.session import TeleopSession
from handumi.teleop.tracking import LatestTrackingSampler, TrackingRecoveryPolicy
from handumi.teleop.trajectory import TeleopCommandStream
from handumi.tracking.base import TrackingProvider
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say
from handumi.visualize import LiveCameraViews

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
record_log = logging.getLogger("handumi.record_teleop")


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
    if gesture == "left" and not recording:
        return "start"
    if gesture == "right" and recording:
        return "save"
    if gesture == "both" and recording:
        return "discard"
    return None


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    joint_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    img_dtype = "video" if use_videos else "image"
    features: dict[str, Any] = {}
    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }
    state_action = joint_state_feature(joint_names)
    features["observation.state"] = state_action
    features["action"] = dict(state_action)
    features.update(feetech_features())
    features.update(capture_timing_features())
    features.update(camera_health_features(cam_names))
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
    motion_smoother: TeleopMotionSmoother | None = None,
) -> tuple[np.ndarray, np.ndarray, int, str, np.ndarray]:
    loop_timer = TeleopLoopTimer(fps)
    n_frames = 0
    start_t: float | None = None
    episode_start_ns: int | None = None
    status = "recorded"
    pending_start_sides = initial_start_sides
    tracking_recovery = TrackingRecoveryPolicy()
    health_gate = SustainedHealthGate(sensor_loss_timeout_s)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)
    sync_lag_ns = int(sync_lag_s * 1e9)
    q = controller.q.copy()
    if motion_smoother is None:
        motion_smoother = TeleopMotionSmoother()
    motion_smoother.reset(q)
    teleop_session = TeleopSession(controller, motion_smoother)
    observations: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    last_processed_tracking_time_ns: int | None = None
    timing_next_log_s = time.perf_counter() + 5.0
    timing_ik_total_s = 0.0
    timing_ik_max_s = 0.0
    timing_ik_samples = 0
    timing_tracking_age_max_s = 0.0
    timing_discarded_ik = 0
    command_stream.stop()

    while True:
        loop_start, _ = loop_timer.tick()
        record_time_ns = time.monotonic_ns()
        if episode_start_ns is None:
            episode_start_ns = record_time_ns

        if stop_event.is_set():
            status = "interrupted"
            observations.clear()
            commands.clear()
            break

        tracking_snapshot = tracking_sampler.latest()
        if tracking_snapshot is None:
            immediate_widths = _latest_widths(grippers)
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
            space_listener.consume_space()
            clap_detector.update_sides(
                immediate_widths.left_mm, immediate_widths.right_mm, loop_start
            )
            clap_arbiter.reset()
            loop_timer.sleep(loop_start)
            continue

        if not tracking_ok:
            if tracking_recovery.note_missing(
                loop_start,
                observed_since=(
                    tracking_snapshot.fresh_at_s if tracking_stale else None
                ),
            ):
                command_stream.stop()
                held = real_env.hold(q)
                controller.tracking_lost(held)
                motion_smoother.reset(held)
                q = held
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
                        "Tracking recovered; double clap or Space to re-anchor."
                    )
                    log_say("tracking recovered", play_sounds=play_sounds)
            loop_timer.sleep(loop_start)
            continue
        if tracking_recovery.lost:
            record_log.info("Tracking stream recovered; waiting for a fresh anchor.")
        tracking_recovery.reset()

        immediate_widths = _latest_widths(grippers)
        start_sides = pending_start_sides
        pending_start_sides = ()
        if space_listener.consume_space():
            start_sides = controller.idle_sides()
        clap_gesture = clap_arbiter.update(
            clap_detector, immediate_widths, loop_start
        )
        gesture_action = _episode_gesture_action(
            clap_gesture, recording=start_t is not None
        )
        if gesture_action == "start":
            start_sides = enabled_sides
        elif gesture_action == "save":
            status = "recorded"
            break
        elif gesture_action == "discard":
            status = "discarded"
            observations.clear()
            commands.clear()
            break

        inputs = teleop_session.inputs(sample, immediate_widths)
        fresh_tracking = (
            last_processed_tracking_time_ns is None
            or tracking_snapshot.source_time_ns != last_processed_tracking_time_ns
        )
        if not fresh_tracking and not start_sides:
            command_stream.update_openings(inputs.openings)
            real_env.check_health()
            loop_timer.sleep(loop_start)
            continue

        observation_q = real_env.read(base_q=q)
        controller_q_before_ik = controller.q.copy()
        ik_start_s = time.perf_counter()
        teleop_frame = teleop_session.advance(
            inputs,
            now_s=loop_start,
            start_sides=start_sides,
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
        ):
            controller.q = controller_q_before_ik
            motion_smoother.restore_joint_command(controller_q_before_ik)
            timing_discarded_ik += 1
            real_env.check_health()
            continue
        anchored = teleop_frame.anchored_sides
        if anchored and start_t is None:
            start_t = loop_start
            record_log.info(
                "Teleop episode started after anchoring %s.", "/".join(anchored)
            )
            log_say("recording episode", play_sounds=play_sounds)

        action_q = teleop_frame.q
        command_stream.submit(
            action_q,
            teleop_frame.inputs.openings,
            time_s=tracking_snapshot.fresh_at_s,
            active=controller.active,
            new_epoch=bool(anchored),
        )
        last_processed_tracking_time_ns = tracking_snapshot.source_time_ns
        real_env.check_health()
        q = action_q

        timing_now_s = time.perf_counter()
        if command_stream.running and timing_now_s >= timing_next_log_s:
            output_stats = command_stream.stats()
            record_log.info(
                "Control timing: output=%.1f Hz, missed=%d, "
                "tracking_age_max=%.0f ms, IK_avg/max=%.1f/%.1f ms, "
                "backend_write_max=%.1f ms, stale_IK_discarded=%d.",
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

        target_time_ns = max(episode_start_ns, record_time_ns - sync_lag_ns)
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

        observations.append(
            canonicalize_command(
                observation_q,
                runtime=runtime,
                openings={
                    "left": gripper_frame.widths.left_normalized,
                    "right": gripper_frame.widths.right_normalized,
                },
            )
        )
        commands.append(
            canonicalize_command(
                played_action_q,
                runtime=runtime,
                openings=played_openings,
            )
        )
        n_frames += 1
        loop_timer.sleep(loop_start)

    command_stream.stop()
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
        translation_deadzone_m=args.translation_deadzone_mm / 1000.0,
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
    camera_views: LiveCameraViews | None = None
    camera_worker: BestEffortPeriodicWorker | None = None
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    motion_config = TeleopMotionConfig.from_args(args)
    motion_smoother = motion_config.make_input_smoother()
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
        camera_views = LiveCameraViews.from_args(
            args, application_id="handumi_teleop_record"
        )
        if camera_views is not None:
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
        motion_smoother.reset(home_q)
        record_log.info(
            "Joint trajectory playback: %.1f Hz, %.0f ms delay, "
            "%.0f ms max bridge, %.0f ms EMA.",
            args.command_rate_hz,
            args.trajectory_delay_ms,
            args.max_extrapolation_ms,
            args.motion_smoothing_time_constant_s * 1000.0,
        )
        space_listener.start()

        from handumi.dataset import EpisodeResult, write_dataset

        layout = canonical_joint_layout(runtime)
        existing_episodes = (
            _existing_episode_count(args.output_dir) if args.resume else 0
        )
        results: list[EpisodeResult] = []
        record_log.info("Recording vector dataset at: %s", args.output_dir)

        clap_detector = DoubleClapDetector()
        clap_arbiter = _BilateralClapArbiter()
        recorded = 0
        auto_start_next = False
        while (
            args.num_episodes <= 0 or recorded < args.num_episodes
        ) and not stop_event.is_set():
            ep_num = existing_episodes + recorded + 1
            ep_total = "inf" if args.num_episodes <= 0 else str(args.num_episodes)
            if not _wait_for_tracking_sampler(
                tracking_sampler,
                stop_event,
                enabled_sides=enabled_sides,
                tracking_stale_ms=args.tracking_stale_ms,
            ):
                break
            record_log.info("--- Episode %d/%s ---", ep_num, ep_total)
            start_immediately = auto_start_next
            auto_start_next = False
            if start_immediately:
                record_log.info("  Starting episode %d automatically ...", ep_num)
            elif not args.space_start:
                record_log.info(
                    "  Double-squeeze left gripper to start episode %d ...",
                    ep_num,
                )
            else:
                record_log.info(
                    "  Press Space%s to start episode %d ...",
                    " or double-squeeze left gripper" if not args.skip_feetech else "",
                    ep_num,
                )
            controller.reset()
            clap_arbiter.reset()
            states, actions, n_frames, status, _ = record_episode(
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
                initial_start_sides=enabled_sides if start_immediately else (),
                sync_lag_s=args.sync_lag_s,
                max_sync_skew_s=args.max_sync_skew_s,
                gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                tracking_loss_timeout_s=args.tracking_loss_timeout_s,
                tracking_stale_ms=args.tracking_stale_ms,
                command_stream=command_stream,
                motion_smoother=motion_smoother,
            )
            if status == "discarded":
                record_log.warning(
                    "Episode discarded by bilateral gesture (%d frames).", n_frames
                )
                log_say("Episode discarded", play_sounds=play_sounds)
                real_env.move_home(home_q)
                continue
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "interrupted",
            }:
                record_log.warning(
                    "Episode discarded (%s, %d frames).", status, n_frames
                )
                log_say("Episode discarded", play_sounds=play_sounds)
                if status == "interrupted":
                    break
                real_env.move_home(home_q)
                continue
            results.append(
                EpisodeResult(
                    episode_index=recorded,
                    states=states,
                    actions=actions,
                    task=args.task,
                    calibration_id=-1,
                    source_kind=1,
                )
            )
            recorded += 1
            record_log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(
                f"Episode {ep_num} saved, {n_frames} frames",
                play_sounds=play_sounds,
            )
            real_env.move_home(home_q)
            auto_start_next = True

        if results:
            write_dataset(
                output_root=args.output_dir,
                source_root=args.output_dir,
                source_info={"features": {}, "handumi": {}},
                episodes=results,
                robot_type=runtime.config.kind,
                joint_names=layout.names,
                fps=args.fps,
                resume=args.resume,
                handumi_metadata={
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
                    "translation_scale": args.translation_scale,
                    "translation_deadzone_mm": args.translation_deadzone_mm,
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
                },
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
