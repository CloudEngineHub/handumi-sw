#!/usr/bin/env python3
"""Run live HandUMI teleop on a registered real-robot backend.

The controller TCP pose is anchored at the robot home TCP, then relative
controller motion drives the IK target. The IK solution ``q`` is the source of
truth; the selected backend converts it to hardware commands and streams them
over CAN.

Safety behavior:

* controller->TCP calibration is required;
* the robot homes before teleop starts;
* arms stay idle until double-clap or explicit ``--space-start``;
* double-clap during teleop clears anchors and returns home.

Examples:

    handumi teleop-real --device pico --robot piper
    handumi teleop-real --device pico --robot piper --space-start
"""

import argparse
import logging
import os
import sys
import time

# Also cover direct execution instead of the ``handumi`` command router.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from handumi.feetech import FeetechGripperSampler
from handumi.feetech.setup import list_feetech_serial_ports
from handumi.real.registry import make_real_backend
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.teleop.common import (
    BestEffortPeriodicWorker,
    KeyboardSpaceListener,
    TeleopLoopTimer,
    enabled_sides as _enabled_sides,
    enabled_tracking_ok as _enabled_tracking_ok,
    latest_widths as _latest_widths,
    tracking_world_map as _tracking_world_map,
)
from handumi.teleop.core import TeleopController
from handumi.teleop.motion import (
    TeleopMotionConfig,
    validate_teleop_motion_args,
)
from handumi.teleop.physical import (
    add_physical_teleop_arguments,
    validate_physical_teleop_args,
)
from handumi.teleop.session import TeleopSession
from handumi.teleop.hardware import (
    load_required_controller_tcp_calibration as _load_required_calibration,
    validate_feetech_ports_exist,
    validate_feetech_ready as _validate_feetech_ready,
)
from handumi.teleop.tracking import LatestTrackingSampler, TrackingRecoveryPolicy
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say
from handumi.visualize import LiveCameraViews

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
real_log = logging.getLogger("handumi.teleop_real")

def _parse_real_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = argparse.ArgumentParser(
        description="Teleoperate a supported physical robot with HandUMI."
    )
    parser.add_argument(
        "--help-advanced", action="store_true", help="Show expert hardware options."
    )
    add_physical_teleop_arguments(parser)
    parser.add_argument(
        "--duration-s", type=float, default=0.0, help="0 means run until Ctrl+C."
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
            "cameras",
            "no_rerun",
        }
        for action in parser._actions:
            if action.dest not in normal:
                action.help = argparse.SUPPRESS
    else:
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args(raw_argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_real_args(argv)


def _validate_real_args(args: argparse.Namespace) -> None:
    validate_teleop_motion_args(args)
    validate_physical_teleop_args(args)
    if args.duration_s < 0.0:
        raise SystemExit("--duration-s must be >= 0.")


def _validate_feetech_ports_exist(feetech_config, *, robot: str = "piper") -> None:
    return validate_feetech_ports_exist(
        feetech_config,
        robot=robot,
        list_ports=list_feetech_serial_ports,
    )


def _run_real() -> None:
    args = _parse_real_args()
    _validate_real_args(args)
    from handumi.scripts.record import build_tracker, connect_feetech

    real_log.info("Loading %s IK solver.", args.robot)
    runtime = load_embodiment(args.robot)
    import jax

    real_log.info("IK compute platform: %s.", jax.default_backend())
    try:
        home_pose_name, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    controller = TeleopController(
        runtime,
        home_q=home_q,
        enabled_sides=_enabled_sides(args.side),
        source_world_to_robot_world=_tracking_world_map(args.device),
        translation_scale=args.translation_scale,
        translation_deadzone_m=args.translation_deadzone_mm / 1000.0,
    )
    q = home_q.copy()
    real_log.info("Selected home pose: %s", home_pose_name)
    real_log.info("Warming IK solver before touching hardware.")
    controller.warmup()
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    gripper_pair = None
    grippers = None
    enabled_sides = _enabled_sides(args.side)
    real_env = make_real_backend(
        args.robot,
        runtime=runtime,
        rig_config=args.rig_config,
        active_sides=enabled_sides,
    )
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    tracker_started = False
    tracking_sampler: LatestTrackingSampler | None = None
    camera_views: LiveCameraViews | None = None
    camera_worker: BestEffortPeriodicWorker | None = None

    clap = DoubleClapDetector()
    play_sounds = not args.no_sounds
    motion_config = TeleopMotionConfig.from_args(args)
    loop_timer = TeleopLoopTimer(motion_config.input_rate_hz)
    motion_smoother = motion_config.make_input_smoother()
    teleop_session = TeleopSession(controller, motion_smoother)
    command_stream = motion_config.make_command_stream(real_env.write)
    episode_start: float | None = None
    frame = 0
    last_processed_tracking_time_ns: int | None = None
    tracking_recovery = TrackingRecoveryPolicy()
    timing_next_log_s = time.perf_counter() + 5.0
    timing_ik_total_s = 0.0
    timing_ik_max_s = 0.0
    timing_ik_samples = 0
    timing_tracking_age_max_s = 0.0
    timing_discarded_ik = 0

    try:
        real_log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        tracking_sampler = LatestTrackingSampler(
            tracker.latest,
            sample_rate_hz=motion_config.input_rate_hz,
        )
        tracking_sampler.start()
        camera_views = LiveCameraViews.from_args(
            args, application_id="handumi_teleop_real"
        )
        if camera_views is not None:
            camera_worker = BestEffortPeriodicWorker(
                camera_views.update,
                rate_hz=args.cam_fps,
                thread_name="handumi-live-camera-preview",
            )
            camera_worker.start()
        gripper_pair = connect_feetech(args)
        if gripper_pair is not None:
            # Keep serial retries off the real-time control loop.  In
            # particular, a transient Feetech timeout must not abort teleop:
            # the sampler retains the last valid widths while it retries in
            # the background, just like the recording paths do.
            grippers = FeetechGripperSampler(
                gripper_pair,
                sample_hz=max(100.0, motion_config.input_rate_hz),
            )
            grippers.start()

        real_env.setup(repair=not args.skip_can_repair)
        real_env.connect()
        real_env.home(home_q)
        motion_smoother.reset(home_q)
        real_log.info(
            "Joint trajectory playback: %.1f Hz, %.0f ms delay, "
            "%.0f ms max bridge, %.0f ms EMA.",
            args.command_rate_hz,
            args.trajectory_delay_ms,
            args.max_extrapolation_ms,
            args.motion_smoothing_time_constant_s * 1000.0,
        )

        space_listener.start()
        if args.space_start:
            real_log.info(
                "Real %s is at home. Start idle arms with Space, or double clap "
                "to start enabled arms.",
                args.robot,
            )
        else:
            real_log.info(
                "Real %s is at home. Double clap a gripper to start enabled arms.",
                args.robot,
            )

        while True:
            loop_start, _ = loop_timer.tick()
            if episode_start is not None and args.duration_s > 0.0:
                if loop_start - episode_start >= args.duration_s:
                    break

            tracking_snapshot = tracking_sampler.latest()
            if tracking_snapshot is None:
                widths = _latest_widths(grippers)
                if args.space_start:
                    space_listener.consume_space()
                clap.update(widths.left_mm, widths.right_mm, loop_start)
                loop_timer.sleep(loop_start)
                continue
            sample = tracking_snapshot.sample
            tracking_age_s = tracking_snapshot.age_s(loop_start)
            timing_tracking_age_max_s = max(
                timing_tracking_age_max_s, tracking_age_s
            )
            tracking_stale = tracking_age_s > args.tracking_stale_ms / 1000.0
            side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
            tracking_ok = (
                not tracking_stale
                and _enabled_tracking_ok(side_tracked, enabled_sides)
            )

            # Tracker startup and a fresh SDK reconnect can briefly expose an
            # empty sample.  Before an operator anchors an arm, there is no
            # robot motion to cancel, so do not hold the robot or restart the
            # PICO service for that transient state.
            if not controller.active and not tracking_ok:
                tracking_recovery.reset()
                widths = _latest_widths(grippers)
                if args.space_start:
                    space_listener.consume_space()
                clap.update(widths.left_mm, widths.right_mm, loop_start)
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
                    real_log.warning(
                        "Tracking lost%s; pending motion cancelled at the current "
                        "robot command. Re-anchor after recovery.",
                        (
                            f" (sample age {tracking_age_s * 1000.0:.0f} ms)"
                            if tracking_stale
                            else ""
                        ),
                    )
                    log_say("tracking lost", play_sounds=play_sounds)
                recover = getattr(tracker, "recover", None)
                if callable(recover) and tracking_recovery.should_recover(loop_start):
                    acquired, recovered = tracking_sampler.try_source_call(recover)
                    if acquired and recovered:
                        real_log.info(
                            "Tracking recovered; double clap or Space to re-anchor."
                        )
                        log_say("tracking recovered", play_sounds=play_sounds)
                loop_timer.sleep(loop_start)
                continue
            if tracking_recovery.lost:
                real_log.info("Tracking stream is valid again; waiting for a fresh anchor.")
            tracking_recovery.reset()

            widths = _latest_widths(grippers)
            inputs = teleop_session.inputs(sample, widths)
            start_sides: tuple[str, ...] = ()
            if args.space_start and space_listener.consume_space():
                start_sides = controller.idle_sides()
                if start_sides:
                    real_log.info("Space pressed; starting %s.", "/".join(start_sides))
            if clap.update(widths.left_mm, widths.right_mm, loop_start):
                if controller.active:
                    command_stream.stop()
                    q = controller.reset()
                    motion_smoother.reset(home_q)
                    episode_start = None
                    frame = 0
                    last_processed_tracking_time_ns = None
                    real_log.info(
                        "Double clap detected; teleop reset, robot returning home slowly."
                    )
                    log_say("returning home", play_sounds=play_sounds)
                    real_env.move_home(home_q)
                    log_say("teleop reset", play_sounds=play_sounds)
                    continue
                start_sides = enabled_sides
                real_log.info("Double clap detected; starting %s.", "/".join(start_sides))

            fresh_tracking = (
                last_processed_tracking_time_ns is None
                or tracking_snapshot.source_time_ns
                != last_processed_tracking_time_ns
            )
            if not fresh_tracking and not start_sides:
                command_stream.update_openings(inputs.openings)
                real_env.check_health()
                loop_timer.sleep(loop_start)
                continue

            controller_q_before_ik = controller.q.copy()
            ik_start_s = time.perf_counter()
            teleop_frame = teleop_session.advance(
                inputs, now_s=loop_start, start_sides=start_sides
            )
            ik_elapsed_s = time.perf_counter() - ik_start_s
            timing_ik_total_s += ik_elapsed_s
            timing_ik_max_s = max(timing_ik_max_s, ik_elapsed_s)
            timing_ik_samples += 1
            newest_after_ik = tracking_sampler.latest()
            if (
                newest_after_ik is not None
                and newest_after_ik.source_time_ns
                != tracking_snapshot.source_time_ns
                and not start_sides
            ):
                # Never publish an IK result for a frame superseded while the
                # solve was running. It would be a visibly delayed correction.
                controller.q = controller_q_before_ik
                motion_smoother.restore_joint_command(controller_q_before_ik)
                timing_discarded_ik += 1
                real_env.check_health()
                continue
            anchored_sides = teleop_frame.anchored_sides
            anchored_this_frame = bool(anchored_sides)
            for side in anchored_sides:
                real_log.info("%s arm anchored; real robot follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            if episode_start is None and anchored_this_frame:
                episode_start = loop_start
                frame = 0
                timing_next_log_s = time.perf_counter() + 5.0
                timing_ik_total_s = 0.0
                timing_ik_max_s = 0.0
                timing_ik_samples = 0
                timing_tracking_age_max_s = 0.0
                timing_discarded_ik = 0
                real_log.info("Teleop timer started.")

            q = teleop_frame.q
            command_stream.submit(
                q,
                inputs.openings,
                time_s=tracking_snapshot.fresh_at_s,
                active=controller.active,
                new_epoch=anchored_this_frame,
            )
            last_processed_tracking_time_ns = tracking_snapshot.source_time_ns
            real_env.check_health()

            timing_now_s = time.perf_counter()
            if command_stream.running and timing_now_s >= timing_next_log_s:
                output_stats = command_stream.stats()
                real_log.info(
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

            loop_timer.sleep(loop_start)
            if episode_start is not None:
                frame += 1
    except KeyboardInterrupt:
        real_log.info("Stopping.")
    finally:
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


def main() -> None:
    _run_real()


if __name__ == "__main__":
    main()
