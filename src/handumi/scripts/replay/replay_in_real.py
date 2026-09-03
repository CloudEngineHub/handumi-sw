#!/usr/bin/env python3
"""Replay a converted joint-level dataset on the physical robot.

This is the last check before training: the joint vectors a policy will learn
from are streamed to the real arms through the robot's hardware backend while
the measured joints are logged and compared with the command. No IK runs and
nothing is re-solved. What moves is exactly what the dataset stores.

The robot is selected like ``teleop-real``: ``--robot`` names a registered
hardware backend (Piper, OpenArm v1, ...) and the dataset must have been
converted for that embodiment. A dataset written in an external LeRobot layout
(``handumi convert --output-layout bi_piper_follower``) is decoded back to
radians and meters first, so the deployable vector is what gets validated.

Safety behavior:

* every frame is checked against the URDF joint limits and the backend's joint
  speed limit before any hardware connection opens (``--dry-run`` stops there);
* the robot homes at its slow homing speed, then ramps into the first frame;
* ``--speed`` slows playback uniformly;
* Ctrl+C or a backend fault holds the arms where they are and disables them
  without a return-home motion; a completed replay returns home slowly.

Examples:

    handumi replay-real outputs/datasets/tblock-all-piper-clean-bi_piper_follower \\
        --episode 0 --dry-run
    handumi replay-real outputs/datasets/tblock-all-piper-clean-bi_piper_follower \\
        --robot piper --episode 0 --speed 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

# Also cover direct execution instead of the ``handumi`` command router.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset import handumi_metadata
from handumi.dataset.canonical import (
    canonical_joint_layout,
    expand_canonical_trajectory,
)
from handumi.dataset.external_layouts import (
    EXTERNAL_LAYOUTS,
    ExternalJointLayout,
    to_canonical,
)
from handumi.dataset.selection import resolve_dataset_selection
from handumi.real.registry import (
    REAL_BACKEND_NAMES,
    TeleopRobotBackend,
    make_real_backend,
)
from handumi.real.streamer import AccelerationLimitedJointTrajectory, step_toward
from handumi.robots.kinematics import pose_error_arrays
from handumi.robots.registry import (
    EMBODIMENT_NAMES,
    RobotRuntime,
    load_embodiment,
    resolve_home_q,
)
from handumi.scripts.replay.replay_in_sim import joint_approach_ramp
from handumi.scripts.replay.replay_in_sim_joints import (
    dataset_info,
    load_joint_episode,
    resolve_state_layout,
    resolved_robot,
)
from handumi.teleop.common import SIDE_CHOICES, TeleopLoopTimer, enabled_sides
from handumi.teleop.motion import TeleopMotionConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.replay_real")

SIDES = ("left", "right")
DEFAULT_OUT_DIR = Path("outputs/replay_in_real")
# Export clips IK overshoot inside this band, so a converted dataset is
# expected to sit at most this far outside the URDF limits.
LIMIT_TOLERANCE_RAD = 2e-3
# One input frame plus the scheduling margin teleop uses at its own cadence.
TRAJECTORY_SCHEDULING_MARGIN_S = 0.0067
# The tracking lag is searched inside this window: the command stream delay
# plus the motor response comfortably fits, while a longer window would let a
# periodic motion alias onto a later repetition.
MAX_TRACKING_LAG_S = 0.4
TRACKING_LAG_STEP_S = 0.005
CONFIRMATION = "REPLAY"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        description=(
            "Stream a converted joint-level dataset to the physical robot it "
            "was converted for and report how closely the arms followed it."
        )
    )
    parser.add_argument(
        "dataset",
        help="Local path or Hugging Face repo id of a converted joints dataset.",
    )
    parser.add_argument(
        "--help-advanced", action="store_true", help="Show expert options."
    )
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        default=[0],
        help="Episode index(es) to replay, in order. Default: 0.",
    )
    parser.add_argument(
        "--robot",
        choices=EMBODIMENT_NAMES,
        default=None,
        help=(
            "Hardware backend to drive. Defaults to the robot the dataset was "
            f"converted for. Backends exist for: {', '.join(REAL_BACKEND_NAMES)}."
        ),
    )
    parser.add_argument("--side", choices=SIDE_CHOICES, default="both")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed factor in (0, 1]. 0.5 plays at half the recorded rate.",
    )
    parser.add_argument(
        "--approach-seconds",
        type=float,
        default=3.0,
        help=(
            "Seconds to ramp from home into the first frame, and between "
            "episodes. Checked against the backend joint speed limit."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("observation.state", "action"),
        default="observation.state",
        help=advanced(
            "Joint column to play. observation.state[t] is the command at t; "
            "action[t] is the command at t+1."
        ),
    )
    parser.add_argument(
        "--state-layout",
        choices=("auto", "handumi", *sorted(EXTERNAL_LAYOUTS)),
        default="auto",
        help=(
            "How the joint columns are encoded. 'handumi' is the canonical "
            "converted layout (radians + meters); an external layout name "
            "denormalizes that stack's vector first. 'auto' reads the dataset "
            "metadata."
        ),
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help=(
            "With an explicit --state-layout: the dataset was recorded with the "
            "plugin's use_degrees option (arm joints in degrees)."
        ),
    )
    parser.add_argument("--revision", default="main", help=advanced("Hub revision."))
    parser.add_argument(
        "--start-frame", type=int, default=0, help=advanced("First frame.")
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help=advanced("Maximum frames.")
    )
    parser.add_argument("--stride", type=int, default=1, help=advanced("Frame stride."))
    parser.add_argument(
        "--home-pose",
        default=None,
        help="Override a legacy named home pose. Omit to use the robot home_q.",
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local robot CAN configuration.",
    )
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Validate but do not auto-repair CAN with sudo before connecting.",
    )
    parser.add_argument(
        "--trajectory-delay-ms",
        type=float,
        default=None,
        help=advanced(
            "Delay of the interpolated command stream. Defaults to one playback "
            "frame plus the teleop scheduling margin."
        ),
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=None,
        metavar="DEG_S2",
        help=(
            "Override the backend's command-stream acceleration limit (deg/s^2) for this "
            "run. Its braking envelope lags a moving target by v^2/(2a), so the "
            "teleop value (720 for Piper) cannot follow fast frames; the dry run "
            "predicts the resulting error. Ignored by rate-limited backends."
        ),
    )
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=3.0,
        help=(
            "Maximum lag-compensated joint tracking error for a PASS verdict."
        ),
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=10.0,
        help="Maximum lag-compensated TCP tracking error for a PASS verdict.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any replayed episode fails the tolerances.",
    )
    parser.add_argument(
        "--no-return-home",
        action="store_true",
        help="Leave the robot at the last frame instead of returning home.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Do not ask for confirmation before moving the robot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where to write the tracking logs. Default: {DEFAULT_OUT_DIR}/<dataset>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decode and check the episodes without touching hardware.",
    )
    return parser


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeCommands:
    """One episode decoded to full URDF joints and normalized openings."""

    episode: int
    frame_indices: np.ndarray
    qpos: np.ndarray
    openings: np.ndarray
    fps: float
    clipped_values: int
    max_clip_rad: float

    @property
    def frames(self) -> int:
        return int(len(self.qpos))

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps


@dataclass(frozen=True)
class ReplayPlan:
    """Everything the hardware run needs, validated before it starts."""

    robot: str
    runtime: RobotRuntime
    external: ExternalJointLayout | None
    home_pose: str
    home_q: np.ndarray
    sides: tuple[str, ...]
    speed: float
    approach_seconds: float
    episodes: tuple[EpisodeCommands, ...]
    motion_joint_indices: tuple[int, ...]
    joint_speed_limit_deg_s: float
    joint_acceleration_limit_deg_s2: float | None
    required_speed_deg_s: float
    approach_speed_deg_s: float
    predictions: tuple[TrackingPrediction, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def playback_rate_hz(self) -> float:
        return self.episodes[0].fps * self.speed


def motion_joint_indices(runtime: RobotRuntime, sides: tuple[str, ...]) -> tuple[int, ...]:
    """Arm joints of the enabled sides, excluding gripper fingers."""
    finger_indices = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    return tuple(
        dict.fromkeys(
            index
            for side in sides
            for index in runtime.arm_joint_indices(side)
            if index not in finger_indices
        )
    )


def joint_speed_limit_deg_s(runtime: RobotRuntime) -> float:
    """The joint speed the backend streamer will not exceed.

    Backends declare it in their own units under ``real`` in the robot YAML:
    Piper as ``max_joint_speed_deg_s``, OpenArm as
    ``control.max_joint_speed_rad_s``.
    """
    options = runtime.config.real_options or {}
    control = options.get("control")
    if isinstance(control, dict) and control.get("max_joint_speed_rad_s") is not None:
        return float(np.rad2deg(float(control["max_joint_speed_rad_s"])))
    if options.get("max_joint_speed_rad_s") is not None:
        return float(np.rad2deg(float(options["max_joint_speed_rad_s"])))
    return float(runtime.config.real.max_joint_speed_deg_s)


def joint_acceleration_limit_deg_s2(
    runtime: RobotRuntime, override: float | None = None
) -> float | None:
    """The stream acceleration limit, or None for a purely rate-limited backend.

    Piper's streamer plans an acceleration-limited move toward the latest
    target (``real.max_joint_acceleration_deg_s2``); OpenArm's steps toward
    it at a fixed maximum rate and has no acceleration setting.
    """
    options = runtime.config.real_options or {}
    if isinstance(options.get("control"), dict):
        return None
    if override is not None:
        return float(override)
    return float(runtime.config.real.max_joint_acceleration_deg_s2)


def runtime_with_acceleration(runtime: RobotRuntime, value: float) -> RobotRuntime:
    """A copy of the runtime whose backend settings carry a new stream limit."""
    real = replace(runtime.config.real, max_joint_acceleration_deg_s2=float(value))
    return replace(runtime, config=replace(runtime.config, real=real))


@dataclass(frozen=True)
class TrackingPrediction:
    """Tracking error the command stream itself will cause, before any motor."""

    episode: int
    lag_s: float
    max_deg: float
    p99_deg: float


def predict_tracking(
    qpos: np.ndarray,
    rate_hz: float,
    indices: tuple[int, ...],
    *,
    max_velocity_deg_s: float,
    max_acceleration_deg_s2: float | None,
    command_rate_hz: float = 100.0,
    episode: int = 0,
) -> TrackingPrediction:
    """Run the frames through the backend's own trajectory generator.

    The stream interpolates the frames at the command rate and the backend
    limits velocity (and, for Piper, acceleration with a braking envelope
    that lags a moving target by ``v^2 / 2a``). Playing the episode through
    the same generator offline predicts that part of the tracking error, so
    a dry run can say whether a speed or acceleration is worth trying.
    """
    q = np.rad2deg(np.asarray(qpos, dtype=np.float64)[:, list(indices)])
    if len(q) < 2 or not indices:
        return TrackingPrediction(episode, 0.0, 0.0, 0.0)
    command_time = np.arange(len(q)) / rate_hz
    output_time = np.arange(0.0, command_time[-1] + 0.5, 1.0 / command_rate_hz)
    target = interpolate_commands(command_time, q, output_time)
    if max_acceleration_deg_s2 is None:
        step = max_velocity_deg_s / command_rate_hz
        output = np.empty_like(target)
        current = target[0]
        for row, value in enumerate(target):
            current = step_toward(current, value, step)
            output[row] = current
    else:
        trajectory = AccelerationLimitedJointTrajectory(
            target[0],
            sample_rate_hz=command_rate_hz,
            max_velocity=max_velocity_deg_s,
            max_acceleration=max_acceleration_deg_s2,
        )
        output = np.stack([trajectory.step(value) for value in target])
    lag = estimate_tracking_lag(command_time, q, output_time, output)
    error = np.abs(output - interpolate_commands(command_time, q, output_time - lag))
    return TrackingPrediction(
        episode=episode,
        lag_s=lag,
        max_deg=float(error.max()),
        p99_deg=float(np.percentile(error.max(axis=1), 99)),
    )


def clip_to_joint_limits(
    qpos: np.ndarray,
    runtime: RobotRuntime,
    *,
    tolerance_rad: float = LIMIT_TOLERANCE_RAD,
) -> tuple[np.ndarray, int, float]:
    """Clip IK overshoot inside ``tolerance_rad``; refuse anything larger.

    Backends do not clip for us: OpenArm holds the previous target on an
    out-of-limit frame, silently skipping motion, and Piper firmware rejects
    it. Both would corrupt the tracking comparison.
    """
    lower = np.asarray(runtime.robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(runtime.robot.joints.upper_limits, dtype=np.float32)
    below = np.maximum(lower - qpos, 0.0)
    above = np.maximum(qpos - upper, 0.0)
    overshoot = np.maximum(below, above)
    worst = float(overshoot.max()) if overshoot.size else 0.0
    if worst > tolerance_rad:
        frame, joint = np.unravel_index(int(overshoot.argmax()), overshoot.shape)
        name = runtime.joint_names[int(joint)]
        raise SystemExit(
            f"Frame {int(frame)} puts {name} {np.rad2deg(worst):.3f} deg outside "
            f"its URDF limit [{lower[joint]:.4f}, {upper[joint]:.4f}] rad, more "
            f"than the {np.rad2deg(tolerance_rad):.3f} deg export tolerance. "
            "This dataset was not produced by `handumi convert`, or its limits "
            "changed; do not send it to the robot."
        )
    clipped = np.clip(qpos, lower, upper).astype(np.float32)
    # Values that merely round onto the limit in float32 are not overshoot.
    return clipped, int(np.count_nonzero(overshoot > 1e-5)), worst


def decode_episode(
    args: argparse.Namespace,
    *,
    runtime: RobotRuntime,
    external: ExternalJointLayout | None,
    episode: int,
) -> EpisodeCommands:
    """Read one episode and expand it to the full URDF joint vector."""
    layout = canonical_joint_layout(runtime)
    episode_args = argparse.Namespace(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        revision=args.revision,
        episode=episode,
    )
    states, actions, fps = load_joint_episode(episode_args)
    if states.ndim != 2 or states.shape[1] != layout.size:
        width = states.shape[1] if states.ndim == 2 else states.shape
        raise SystemExit(
            f"Expected {layout.size} joint columns for {runtime.name}, got {width}. "
            "The dataset was converted for a different embodiment."
        )
    selected = states if args.source == "observation.state" else actions
    frame_indices = list(range(args.start_frame, len(selected), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise SystemExit(f"Episode {episode}: no frames selected for replay.")
    canonical = selected[frame_indices]
    if external is not None:
        canonical = to_canonical(canonical, layout=external, runtime=runtime)
    qpos, openings = expand_canonical_trajectory(canonical, runtime=runtime)
    if not np.all(np.isfinite(qpos)):
        raise SystemExit(f"Episode {episode} contains non-finite joint values.")
    qpos, clipped, worst = clip_to_joint_limits(qpos, runtime)
    return EpisodeCommands(
        episode=int(episode),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        qpos=qpos,
        openings=openings.astype(np.float32),
        fps=float(fps),
        clipped_values=clipped,
        max_clip_rad=worst,
    )


def apply_inactive_sides(
    plan_sides: tuple[str, ...],
    runtime: RobotRuntime,
    home_q: np.ndarray,
    qpos: np.ndarray,
    openings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Park the arms the operator did not enable at home with a closed gripper."""
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    openings = np.asarray(openings, dtype=np.float32).copy()
    for side_index, side in enumerate(SIDES):
        if side in plan_sides:
            continue
        indices = runtime.arm_joint_indices(side)
        qpos[:, indices] = np.asarray(home_q, dtype=np.float32)[indices]
        openings[:, side_index] = 0.0
    return qpos, openings


# Frames asking more than the backend limit make the arm lag briefly, which
# the tracking report shows; only a sustained overrun makes the run pointless.
MAX_FAST_FRAME_FRACTION = 0.05


@dataclass(frozen=True)
class SpeedDemand:
    """Per-frame joint speed a playback asks of the backend."""

    peak_deg_s: float
    peak_frame: int
    peak_joint: str
    fast_frames: int
    frames: int

    @property
    def fast_fraction(self) -> float:
        return self.fast_frames / self.frames if self.frames else 0.0


def speed_demand(
    qpos: np.ndarray,
    rate_hz: float,
    indices: tuple[int, ...],
    *,
    joint_names: tuple[str, ...],
    limit_deg_s: float,
) -> SpeedDemand:
    if len(qpos) < 2 or not indices:
        return SpeedDemand(0.0, 0, "", 0, max(len(qpos) - 1, 0))
    speeds = np.rad2deg(np.abs(np.diff(qpos[:, list(indices)], axis=0))) * rate_hz
    frame, column = np.unravel_index(int(speeds.argmax()), speeds.shape)
    return SpeedDemand(
        peak_deg_s=float(speeds.max()),
        peak_frame=int(frame) + 1,
        peak_joint=joint_names[indices[int(column)]],
        fast_frames=int(np.count_nonzero(speeds.max(axis=1) > limit_deg_s)),
        frames=len(speeds),
    )


def eased_alphas(frames: int) -> np.ndarray:
    """The smoothstep profile ``joint_approach_ramp`` uses, for the grippers."""
    if frames <= 0:
        return np.empty(0, dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, frames + 1, dtype=np.float32)[:-1]
    return alpha * alpha * (3.0 - 2.0 * alpha)


def approach_segment(
    from_q: np.ndarray,
    from_openings: np.ndarray,
    to_q: np.ndarray,
    to_openings: np.ndarray,
    *,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint and gripper lead-in from one pose to the next episode start."""
    ramp = joint_approach_ramp(from_q, to_q, frames=frames)
    alphas = eased_alphas(len(ramp))
    openings = (
        np.asarray(from_openings, dtype=np.float32)[None]
        + alphas[:, None]
        * (np.asarray(to_openings, dtype=np.float32) - np.asarray(from_openings, dtype=np.float32))[None]
    )
    return ramp, openings.astype(np.float32)


def build_plan(args: argparse.Namespace) -> ReplayPlan:
    """Decode every requested episode and check it against the robot limits."""
    info = args.dataset_info
    external = args.external_layout
    robot = args.robot
    if external is not None and external.robot != robot:
        raise SystemExit(
            f"The dataset is written in the {external.robot_type} layout, which "
            f"describes {external.robot!r}; it cannot drive {robot!r}."
        )
    target = handumi_metadata(info).get("target_robot")
    if external is None and isinstance(target, dict) and target.get("name") not in (None, robot):
        raise SystemExit(
            f"The dataset was converted for {target.get('name')!r}; it cannot drive "
            f"{robot!r}."
        )
    if robot not in REAL_BACKEND_NAMES:
        raise SystemExit(
            f"No real hardware backend is registered for {robot!r}. Backends "
            f"exist for: {', '.join(REAL_BACKEND_NAMES)}."
        )

    runtime = load_embodiment(robot)
    try:
        home_pose, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    sides = enabled_sides(args.side)
    indices = motion_joint_indices(runtime, sides)

    episodes: list[EpisodeCommands] = []
    for episode in args.episode:
        decoded = decode_episode(args, runtime=runtime, external=external, episode=episode)
        qpos, openings = apply_inactive_sides(sides, runtime, home_q, decoded.qpos, decoded.openings)
        episodes.append(
            EpisodeCommands(
                episode=decoded.episode,
                frame_indices=decoded.frame_indices,
                qpos=qpos,
                openings=openings,
                fps=decoded.fps,
                clipped_values=decoded.clipped_values,
                max_clip_rad=decoded.max_clip_rad,
            )
        )
    fps_values = {episode.fps for episode in episodes}
    if len(fps_values) != 1:
        raise SystemExit(f"Episodes disagree on fps: {sorted(fps_values)}.")

    limit = joint_speed_limit_deg_s(runtime)
    acceleration = joint_acceleration_limit_deg_s2(
        runtime, args.accel
    )
    rate = episodes[0].fps * args.speed
    demands = [
        speed_demand(e.qpos, rate, indices, joint_names=runtime.joint_names, limit_deg_s=limit)
        for e in episodes
    ]
    required = max(demand.peak_deg_s for demand in demands)
    # Home -> first frame, and last frame -> next first frame, both eased over
    # the same duration; smoothstep peaks at 1.5x the mean speed.
    hops = [(home_q, episodes[0].qpos[0])]
    hops += [(a.qpos[-1], b.qpos[0]) for a, b in zip(episodes, episodes[1:], strict=False)]
    approach = 0.0
    if args.approach_seconds > 0.0:
        approach = max(
            1.5 * float(np.rad2deg(np.abs(b - a)[list(indices)].max())) / args.approach_seconds
            for a, b in hops
        )
    notes: list[str] = []
    for episode, demand in zip(episodes, demands, strict=True):
        if demand.fast_fraction > MAX_FAST_FRAME_FRACTION:
            suggested = args.speed * limit / demand.peak_deg_s * 0.9
            raise SystemExit(
                f"Episode {episode.episode} asks more than the {limit:.1f} deg/s the "
                f"{robot} backend streams on {demand.fast_frames} of {demand.frames} "
                f"frames (peak {demand.peak_deg_s:.1f} deg/s on {demand.peak_joint}); "
                "the arm would lag through most of it and the comparison would be "
                f"meaningless. Rerun with --speed {suggested:.2f} or lower."
            )
        if demand.fast_frames:
            notes.append(
                f"episode {episode.episode} asks more than {limit:.1f} deg/s on "
                f"{demand.fast_frames} frame(s) (peak {demand.peak_deg_s:.1f} deg/s on "
                f"{demand.peak_joint} at frame {demand.peak_frame}); the arm will "
                "lag briefly there. Lower --speed to remove the spikes."
            )
    if approach > limit:
        needed = args.approach_seconds * approach / limit * 1.1
        raise SystemExit(
            f"The lead-in from home needs {approach:.1f} deg/s over "
            f"{args.approach_seconds:.1f}s but the backend allows {limit:.1f} deg/s. "
            f"Rerun with --approach-seconds {needed:.1f} or more."
        )
    if args.approach_seconds <= 0.0:
        first_jump = float(np.rad2deg(np.abs(episodes[0].qpos[0] - home_q)[list(indices)].max()))
        if first_jump > 1.0:
            raise SystemExit(
                f"--approach-seconds 0 would jump {first_jump:.1f} deg from home "
                "into the first frame. Use a positive lead-in on hardware."
            )
    predictions = tuple(
        predict_tracking(
            e.qpos, rate, indices,
            max_velocity_deg_s=limit, max_acceleration_deg_s2=acceleration,
            command_rate_hz=float(runtime.config.real.command_rate_hz), episode=e.episode,
        )
        for e in episodes
    )
    worst = max(predictions, key=lambda p: p.max_deg)
    if worst.max_deg > args.tolerance_deg:
        hint = (
            "lower --speed"
            if acceleration is None
            else "lower --speed or raise --accel"
        )
        notes.append(
            f"the command stream alone will lag episode {worst.episode} by up to "
            f"{worst.max_deg:.1f} deg (p99 {worst.p99_deg:.1f}) at these limits, above "
            f"the {args.tolerance_deg:g} deg tolerance; {hint} before blaming the robot."
        )
    if args.accel is not None and acceleration is None:
        notes.append(
            f"{robot} streams at a fixed rate; --accel has no effect."
        )
    clipped = sum(e.clipped_values for e in episodes)
    if clipped:
        worst = max(e.max_clip_rad for e in episodes)
        notes.append(
            f"{clipped} joint value(s) sat up to {np.rad2deg(worst):.4f} deg outside "
            "the URDF limits (IK overshoot) and were clipped to the limit."
        )
    return ReplayPlan(
        robot=robot,
        runtime=runtime,
        external=external,
        home_pose=home_pose,
        home_q=np.asarray(home_q, dtype=np.float32),
        sides=sides,
        speed=float(args.speed),
        approach_seconds=float(args.approach_seconds),
        episodes=tuple(episodes),
        motion_joint_indices=indices,
        joint_speed_limit_deg_s=limit,
        joint_acceleration_limit_deg_s2=acceleration,
        required_speed_deg_s=required,
        approach_speed_deg_s=approach,
        predictions=predictions,
        notes=tuple(notes),
    )


def describe_table_placement(info: dict[str, object]) -> list[str]:
    """Where conversion put the table, so the physical one can match it.

    The placement is baked into the stored joints by the absolute-table
    retargeting; replay cannot change it, only tell the operator what it was.
    """
    deployment = handumi_metadata(info).get("deployment_calibration")
    if not isinstance(deployment, dict):
        return ["  Table placement: not recorded by the conversion"]
    path = deployment.get("path")
    profile = deployment.get("profile", "?")
    verified = bool(deployment.get("verified"))
    lines = [
        f"  Table placement: profile {profile} "
        f"({'verified' if verified else 'unverified'}, scope {deployment.get('scope', '?')}"
        f"{', lab ' + str(deployment['lab']) if deployment.get('lab') else ''}) "
        f"from {path}"
    ]
    if isinstance(path, str) and Path(path).is_file():
        try:
            import yaml

            data = yaml.safe_load(Path(path).read_text()) or {}
            pose = ((data.get("calibration") or {}).get("robot_from_table") or {})
            position = pose.get("position")
            if position is not None:
                x, y, z = (float(v) for v in position)
                lines.append(
                    f"    table origin at x={x:.2f} m forward, y={y:.2f} m left, "
                    f"z={z:.2f} m from the robot world origin (between the arm bases)"
                )
        except Exception:  # noqa: BLE001 - informational only
            pass
    if deployment.get("scope") == "simulation" or not verified:
        lines.append(
            "    This is not a measured installation: place the physical table "
            "exactly so, clear it for the first run, or reconvert with a "
            "local calibration (configs/calibration/table/local/<robot>.yaml)."
        )
    return lines


def describe_plan(plan: ReplayPlan, args: argparse.Namespace) -> str:
    total_frames = sum(e.frames for e in plan.episodes)
    total_s = sum(e.duration_s for e in plan.episodes) / plan.speed
    layout = plan.external.name if plan.external else "handumi canonical"
    lines = [
        "Real replay plan",
        f"  Dataset: {args.dataset_root}",
        f"  Repository: {args.repo_id}",
        f"  Robot backend: {plan.robot} (home pose {plan.home_pose})",
        f"  Arms: {'/'.join(plan.sides)}",
        f"  State layout: {layout}",
        f"  Joint column: {args.source}",
        f"  Episodes: {', '.join(str(e.episode) for e in plan.episodes)} "
        f"({total_frames} frames, {total_s:.1f}s at speed {plan.speed:g})",
        f"  Playback rate: {plan.playback_rate_hz:.1f} Hz",
        f"  Joint speed: needs {plan.required_speed_deg_s:.1f} deg/s, "
        f"backend allows {plan.joint_speed_limit_deg_s:.1f} deg/s"
        + (
            f" at {plan.joint_acceleration_limit_deg_s2:.0f} deg/s^2"
            if plan.joint_acceleration_limit_deg_s2 is not None
            else " (rate-limited stream)"
        ),
        f"  Lead-in: {plan.approach_seconds:.1f}s from home "
        f"(peak {plan.approach_speed_deg_s:.1f} deg/s)",
        *describe_table_placement(args.dataset_info),
    ]
    for episode, prediction in zip(plan.episodes, plan.predictions, strict=True):
        opening = episode.openings
        lines.append(
            f"  Episode {episode.episode}: {episode.frames} frames, "
            f"gripper left {opening[:, 0].min() * 100:.0f}..{opening[:, 0].max() * 100:.0f}% "
            f"right {opening[:, 1].min() * 100:.0f}..{opening[:, 1].max() * 100:.0f}%, "
            f"predicted stream lag {prediction.lag_s * 1000:.0f} ms, error max "
            f"{prediction.max_deg:.1f} / p99 {prediction.p99_deg:.1f} deg"
        )
    for note in plan.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracking analysis
# ---------------------------------------------------------------------------


@dataclass
class FeedbackLog:
    """Measured joints and gripper openings sampled while streaming."""

    time_s: list[float] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    openings: list[np.ndarray] = field(default_factory=list)

    def append(self, time_s: float, q: np.ndarray, openings: dict[str, float]) -> None:
        self.time_s.append(float(time_s))
        self.qpos.append(np.asarray(q, dtype=np.float32).copy())
        self.openings.append(
            np.asarray(
                [openings.get(side, np.nan) for side in SIDES], dtype=np.float32
            )
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.time_s:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
        return (
            np.asarray(self.time_s, dtype=np.float64),
            np.stack(self.qpos),
            np.stack(self.openings),
        )


def interpolate_commands(
    command_time_s: np.ndarray, commands: np.ndarray, sample_time_s: np.ndarray
) -> np.ndarray:
    """Piecewise-linear command value at each sample time (held outside)."""
    commands = np.asarray(commands, dtype=np.float64)
    return np.stack(
        [
            np.interp(sample_time_s, command_time_s, commands[:, column])
            for column in range(commands.shape[1])
        ],
        axis=1,
    )


def estimate_tracking_lag(
    command_time_s: np.ndarray,
    commands: np.ndarray,
    sample_time_s: np.ndarray,
    samples: np.ndarray,
    *,
    max_lag_s: float = MAX_TRACKING_LAG_S,
    step_s: float = TRACKING_LAG_STEP_S,
) -> float:
    """Lag that best aligns the measured joints with the command they follow.

    Every sample is scored at every candidate lag, with the command held at
    its first and last value outside the streamed window: the samples taken
    while the arm settles on the last frame are part of what it had to reach,
    and the same sample set for every lag keeps the comparison fair.
    """
    if len(sample_time_s) == 0 or len(command_time_s) < 2:
        return 0.0
    best_lag, best_error = 0.0, np.inf
    for lag in np.arange(0.0, max_lag_s + 1e-9, step_s):
        expected = interpolate_commands(command_time_s, commands, sample_time_s - lag)
        error = float(np.mean(np.abs(samples - expected)))
        if error < best_error - 1e-9:
            best_lag, best_error = float(lag), error
    return best_lag


@dataclass(frozen=True)
class TrackingReport:
    episode: int
    frames: int
    samples: int
    lag_s: float
    joint_mean_deg: float
    joint_max_deg: float
    joint_max_name: str
    joint_max_frame: int
    raw_joint_max_deg: float
    tcp_mean_mm: dict[str, float]
    tcp_max_mm: dict[str, float]
    tcp_max_rot_deg: dict[str, float]
    gripper_mean_pct: float | None
    gripper_max_pct: float | None
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "frames": self.frames,
            "feedback_samples": self.samples,
            "tracking_lag_ms": round(self.lag_s * 1000.0, 1),
            "joint_error_deg": {
                "mean": round(self.joint_mean_deg, 3),
                "max": round(self.joint_max_deg, 3),
                "max_joint": self.joint_max_name,
                "max_frame": self.joint_max_frame,
                "max_without_lag_compensation": round(self.raw_joint_max_deg, 3),
            },
            "tcp_error": {
                side: {
                    "mean_mm": round(self.tcp_mean_mm[side], 2),
                    "max_mm": round(self.tcp_max_mm[side], 2),
                    "max_rot_deg": round(self.tcp_max_rot_deg[side], 2),
                }
                for side in self.tcp_mean_mm
            },
            "gripper_error_pct": (
                None
                if self.gripper_mean_pct is None
                else {"mean": round(self.gripper_mean_pct, 2), "max": round(self.gripper_max_pct or 0.0, 2)}
            ),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def analyze_tracking(
    plan: ReplayPlan,
    episode: EpisodeCommands,
    command_time_s: np.ndarray,
    feedback: FeedbackLog,
    *,
    tolerance_deg: float,
    tolerance_mm: float,
) -> tuple[TrackingReport, dict[str, np.ndarray]]:
    """Compare what the robot measured with what the dataset commanded."""
    sample_time_s, measured_q, measured_openings = feedback.arrays()
    indices = list(plan.motion_joint_indices)
    names = plan.runtime.joint_names
    reasons: list[str] = []
    if len(sample_time_s) < 2:
        report = TrackingReport(
            episode=episode.episode, frames=episode.frames, samples=len(sample_time_s),
            lag_s=0.0, joint_mean_deg=float("nan"), joint_max_deg=float("nan"),
            joint_max_name="", joint_max_frame=-1, raw_joint_max_deg=float("nan"),
            tcp_mean_mm={}, tcp_max_mm={}, tcp_max_rot_deg={}, gripper_mean_pct=None,
            gripper_max_pct=None, passed=False, reasons=("no joint feedback was read",),
        )
        return report, {}

    lag = estimate_tracking_lag(
        command_time_s, episode.qpos[:, indices], sample_time_s, measured_q[:, indices]
    )
    # Every sample counts, including the settle after the last frame: the
    # hold there is part of what the robot had to reach.
    expected_q = interpolate_commands(command_time_s, episode.qpos, sample_time_s - lag)
    raw_expected_q = interpolate_commands(command_time_s, episode.qpos, sample_time_s)
    joint_error = np.abs(measured_q[:, indices] - expected_q[:, indices])
    raw_joint_error = np.abs(measured_q[:, indices] - raw_expected_q[:, indices])
    worst = np.unravel_index(int(joint_error.argmax()), joint_error.shape)
    worst_frame = int(np.searchsorted(command_time_s, sample_time_s[worst[0]] - lag))
    joint_max_deg = float(np.rad2deg(joint_error.max()))
    joint_mean_deg = float(np.rad2deg(joint_error.mean()))

    solver = plan.runtime.solver_cls()
    expected_full = expected_q.astype(np.float32)
    measured_full = measured_q.astype(np.float32)
    # Fingers are not measured joints on every backend; compare arm poses
    # with the commanded finger values on both sides of the comparison.
    finger_indices = [
        finger.index
        for fingers in (plan.runtime.finger_joints or {}).values()
        for finger in fingers
    ]
    measured_full[:, finger_indices] = expected_full[:, finger_indices]
    target = {side: np.empty((len(expected_full), 7), dtype=np.float32) for side in SIDES}
    achieved = {side: np.empty((len(expected_full), 7), dtype=np.float32) for side in SIDES}
    for row, (q_target, q_measured) in enumerate(zip(expected_full, measured_full, strict=True)):
        target["left"][row], target["right"][row] = solver.fk_pose7(q_target)
        achieved["left"][row], achieved["right"][row] = solver.fk_pose7(q_measured)
    errors = pose_error_arrays(target["left"], target["right"], achieved["left"], achieved["right"])
    tcp_mean_mm = {
        side: float(errors[f"{side}_pos_error_m"].mean() * 1000.0) for side in plan.sides
    }
    tcp_max_mm = {
        side: float(errors[f"{side}_pos_error_m"].max() * 1000.0) for side in plan.sides
    }
    tcp_max_rot_deg = {
        side: float(errors[f"{side}_rot_error_deg"].max()) for side in plan.sides
    }

    gripper_mean = gripper_max = None
    side_columns = [SIDES.index(side) for side in plan.sides]
    measured_grip = measured_openings[:, side_columns]
    if np.any(np.isfinite(measured_grip)):
        expected_grip = interpolate_commands(
            command_time_s, episode.openings[:, side_columns], sample_time_s - lag
        )
        grip_error = np.abs(measured_grip - expected_grip)
        gripper_mean = float(np.nanmean(grip_error) * 100.0)
        gripper_max = float(np.nanmax(grip_error) * 100.0)

    if joint_max_deg > tolerance_deg:
        reasons.append(
            f"joint error {joint_max_deg:.2f} deg on {names[indices[worst[1]]]} "
            f"exceeds {tolerance_deg:.2f} deg"
        )
    for side in plan.sides:
        if tcp_max_mm[side] > tolerance_mm:
            reasons.append(
                f"{side} TCP error {tcp_max_mm[side]:.1f} mm exceeds {tolerance_mm:.1f} mm"
            )
    report = TrackingReport(
        episode=episode.episode,
        frames=episode.frames,
        samples=len(sample_time_s),
        lag_s=lag,
        joint_mean_deg=joint_mean_deg,
        joint_max_deg=joint_max_deg,
        joint_max_name=names[indices[worst[1]]],
        joint_max_frame=min(worst_frame, episode.frames - 1),
        raw_joint_max_deg=float(np.rad2deg(raw_joint_error.max())),
        tcp_mean_mm=tcp_mean_mm,
        tcp_max_mm=tcp_max_mm,
        tcp_max_rot_deg=tcp_max_rot_deg,
        gripper_mean_pct=gripper_mean,
        gripper_max_pct=gripper_max,
        passed=not reasons,
        reasons=tuple(reasons),
    )
    arrays = {
        "measured_time_s": sample_time_s,
        "measured_qpos": measured_q,
        "measured_openings": measured_openings,
        "expected_qpos_at_lag": expected_q.astype(np.float32),
        "left_pos_error_m": errors["left_pos_error_m"],
        "right_pos_error_m": errors["right_pos_error_m"],
        "left_rot_error_deg": errors["left_rot_error_deg"],
        "right_rot_error_deg": errors["right_rot_error_deg"],
        "fk_target_left_pose7": target["left"],
        "fk_target_right_pose7": target["right"],
        "fk_measured_left_pose7": achieved["left"],
        "fk_measured_right_pose7": achieved["right"],
    }
    return report, arrays


def format_report(report: TrackingReport) -> str:
    if report.samples < 2:
        return (
            f"[replay-real] episode {report.episode}: FAIL "
            f"({'; '.join(report.reasons)})"
        )
    lines = [
        f"[replay-real] episode {report.episode}: {report.frames} frames, "
        f"{report.samples} feedback samples, tracking lag {report.lag_s * 1000:.0f} ms",
        f"  joints: mean {report.joint_mean_deg:.2f} deg, max {report.joint_max_deg:.2f} deg "
        f"({report.joint_max_name} @ frame {report.joint_max_frame}); "
        f"without lag compensation max {report.raw_joint_max_deg:.2f} deg",
    ]
    for side in report.tcp_mean_mm:
        lines.append(
            f"  {side} TCP: mean {report.tcp_mean_mm[side]:.1f} mm, max "
            f"{report.tcp_max_mm[side]:.1f} mm / {report.tcp_max_rot_deg[side]:.1f} deg"
        )
    if report.gripper_mean_pct is not None:
        lines.append(
            f"  gripper: mean {report.gripper_mean_pct:.1f}%, max {report.gripper_max_pct:.1f}% of full opening"
        )
    else:
        lines.append("  gripper: no physical opening feedback on this backend")
    verdict = "PASS" if report.passed else "FAIL (" + "; ".join(report.reasons) + ")"
    lines.append(f"  verdict: {verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardware run
# ---------------------------------------------------------------------------


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return DEFAULT_OUT_DIR / Path(str(args.dataset_root)).name


def save_episode(
    directory: Path,
    plan: ReplayPlan,
    episode: EpisodeCommands,
    command_time_s: np.ndarray,
    report: TrackingReport,
    arrays: dict[str, np.ndarray],
    *,
    repo_id: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"episode_{episode.episode:06d}_{plan.robot}"
    payload: dict[str, Any] = {
        "repo_id": np.asarray([repo_id]),
        "robot": np.asarray([plan.robot]),
        "episode": np.asarray([episode.episode], dtype=np.int64),
        "joint_names": np.asarray(list(plan.runtime.joint_names)),
        "sides": np.asarray(list(plan.sides)),
        "fps": np.asarray([episode.fps], dtype=np.float32),
        "speed": np.asarray([plan.speed], dtype=np.float32),
        "frame_indices": episode.frame_indices,
        "commanded_time_s": command_time_s,
        "commanded_qpos": episode.qpos,
        "commanded_openings": episode.openings,
        "tracking_lag_s": np.asarray([report.lag_s], dtype=np.float32),
        **arrays,
    }
    np.savez_compressed(stem.with_suffix(".npz"), **payload)
    stem.with_suffix(".json").write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    return stem.with_suffix(".npz")


def confirm_motion(plan: ReplayPlan, *, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit("Refusing to move the robot without a terminal; pass --yes.")
    answer = input(
        f"The {plan.robot} will home, then replay "
        f"{len(plan.episodes)} episode(s) with the {'/'.join(plan.sides)} arm(s). "
        f"Clear the workspace and hold the emergency stop ready. Type {CONFIRMATION}: "
    ).strip()
    if answer != CONFIRMATION:
        raise SystemExit("Replay cancelled; the robot was not connected.")


class _Playback:
    """Stream frames at the playback rate while logging robot feedback."""

    def __init__(
        self,
        plan: ReplayPlan,
        backend: TeleopRobotBackend,
        *,
        trajectory_delay_s: float,
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.rate_hz = plan.playback_rate_hz
        self.stream = TeleopMotionConfig(
            input_rate_hz=self.rate_hz,
            command_rate_hz=float(plan.runtime.config.real.command_rate_hz),
            trajectory_delay_s=trajectory_delay_s,
        ).make_command_stream(backend.write)
        self.trajectory_delay_s = trajectory_delay_s
        self.timer = TeleopLoopTimer(self.rate_hz)
        self.q = plan.home_q.copy()
        self.openings = np.zeros(2, dtype=np.float32)
        self._started = False

    def _submit(self, q: np.ndarray, openings: np.ndarray, *, now: float) -> None:
        self.q = np.asarray(q, dtype=np.float32)
        self.openings = np.asarray(openings, dtype=np.float32)
        self.stream.submit(
            self.q,
            {side: float(self.openings[index]) for index, side in enumerate(SIDES)},
            time_s=now,
            active=True,
            new_epoch=not self._started,
        )
        self._started = True

    def _sample(self, feedback: FeedbackLog | None, now: float) -> None:
        if feedback is None:
            return
        feedback.append(now, self.backend.read(base_q=self.q), self.backend.read_gripper_openings())

    def play(
        self,
        qpos: np.ndarray,
        openings: np.ndarray,
        *,
        feedback: FeedbackLog | None,
    ) -> np.ndarray:
        """Submit every frame at the playback rate; return their submit times."""
        times = np.empty(len(qpos), dtype=np.float64)
        for index, (q, opening) in enumerate(zip(qpos, openings, strict=True)):
            loop_start, _ = self.timer.tick()
            self._sample(feedback, loop_start)
            self._submit(q, opening, now=loop_start)
            times[index] = loop_start
            self.backend.check_health()
            self.timer.sleep(loop_start)
        return times

    def settle(self, seconds: float, *, feedback: FeedbackLog | None) -> None:
        """Keep sampling while the stream drains its delay and the arm arrives.

        The last frame is re-submitted every tick: the command buffer holds
        the newest *predicted* command after its last sample, and a repeated
        frame makes that prediction the frame itself.
        """
        deadline = time.perf_counter() + seconds
        while True:
            loop_start, _ = self.timer.tick()
            self._sample(feedback, loop_start)
            self._submit(self.q, self.openings, now=loop_start)
            self.backend.check_health()
            if loop_start >= deadline:
                return
            self.timer.sleep(loop_start)

    def stop(self) -> None:
        self.stream.stop()


def run_plan(plan: ReplayPlan, args: argparse.Namespace) -> list[TrackingReport]:
    """Home, replay every episode, analyze tracking, and return home."""
    runtime = plan.runtime
    if (
        args.accel is not None
        and plan.joint_acceleration_limit_deg_s2 is not None
    ):
        runtime = runtime_with_acceleration(runtime, plan.joint_acceleration_limit_deg_s2)
        log.info(
            "Command stream acceleration limit for this run: %.0f deg/s^2 (robot YAML: %.0f).",
            plan.joint_acceleration_limit_deg_s2,
            plan.runtime.config.real.max_joint_acceleration_deg_s2,
        )
    try:
        backend = make_real_backend(
            plan.robot,
            runtime=runtime,
            rig_config=args.rig_config,
            active_sides=plan.sides,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rate = plan.playback_rate_hz
    delay_s = (
        args.trajectory_delay_ms / 1000.0
        if args.trajectory_delay_ms is not None
        else 1.0 / rate + TRAJECTORY_SCHEDULING_MARGIN_S
    )
    approach_frames = int(round(rate * plan.approach_seconds))
    settle_s = delay_s + 0.5
    directory = output_directory(args)
    reports: list[TrackingReport] = []
    playback: _Playback | None = None
    connected = False
    completed = False
    fault: BaseException | None = None
    try:
        log.info("Preparing %s transports.", plan.robot)
        backend.setup(repair=not args.skip_can_repair)
        backend.connect()
        connected = True
        log.info("Homing %s at its slow homing speed.", plan.robot)
        backend.home(plan.home_q)
        playback = _Playback(plan, backend, trajectory_delay_s=delay_s)
        log.info(
            "Command stream: %.1f Hz playback -> %.0f Hz interpolated, %.0f ms delay.",
            rate,
            plan.runtime.config.real.command_rate_hz,
            delay_s * 1000.0,
        )
        previous_q = plan.home_q
        previous_openings = np.zeros(2, dtype=np.float32)
        for episode in plan.episodes:
            ramp_q, ramp_openings = approach_segment(
                previous_q, previous_openings, episode.qpos[0], episode.openings[0],
                frames=approach_frames,
            )
            if len(ramp_q):
                log.info(
                    "Episode %d: %.1fs lead-in into the first frame.",
                    episode.episode,
                    len(ramp_q) / rate,
                )
                playback.play(ramp_q, ramp_openings, feedback=None)
            log.info(
                "Episode %d: streaming %d frames (%.1fs).",
                episode.episode,
                episode.frames,
                episode.duration_s / plan.speed,
            )
            feedback = FeedbackLog()
            command_time_s = playback.play(episode.qpos, episode.openings, feedback=feedback)
            playback.settle(settle_s, feedback=feedback)
            report, arrays = analyze_tracking(
                plan, episode, command_time_s, feedback,
                tolerance_deg=args.tolerance_deg, tolerance_mm=args.tolerance_mm,
            )
            saved = save_episode(
                directory, plan, episode, command_time_s, report, arrays, repo_id=args.repo_id
            )
            print(format_report(report))
            print(f"[replay-real] saved: {saved}")
            reports.append(report)
            previous_q = episode.qpos[-1]
            previous_openings = episode.openings[-1]
        completed = True
    except KeyboardInterrupt:
        log.warning("Interrupted; holding the arms where they are.")
    except Exception as exc:  # noqa: BLE001 - any backend fault ends the run
        fault = exc
        log.error("Replay aborted: %s", exc)
    finally:
        if playback is not None:
            try:
                playback.stop()
            except Exception as exc:  # noqa: BLE001 - report, keep shutting down
                log.error("Command stream stopped with an error: %s", exc)
        if connected:
            try:
                if completed and not args.no_return_home:
                    log.info("Returning %s home slowly.", plan.robot)
                    backend.move_home(plan.home_q)
                elif playback is not None:
                    backend.hold(playback.q)
            except Exception as exc:  # noqa: BLE001 - never skip disconnect
                log.error("Final motion failed: %s", exc)
            finally:
                backend.disconnect()
    if not completed:
        reason = f": {fault}" if fault is not None else ""
        raise SystemExit(
            f"Replay did not complete{reason}; the arms were held and disabled."
        )
    return reports


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = build_parser(show_advanced=show_advanced)
    if show_advanced:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(raw_argv)
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be in (0, 1]; a real robot is never replayed faster than recorded.")
    if args.approach_seconds < 0.0:
        parser.error("--approach-seconds must be >= 0.")
    if args.stride < 1:
        parser.error("--stride must be >= 1.")
    if args.tolerance_deg <= 0.0 or args.tolerance_mm <= 0.0:
        parser.error("--tolerance-deg and --tolerance-mm must be > 0.")
    if args.accel is not None and args.accel <= 0.0:
        parser.error("--accel must be > 0.")
    if args.trajectory_delay_ms is not None and args.trajectory_delay_ms < 0.0:
        parser.error("--trajectory-delay-ms must be >= 0.")
    if len(set(args.episode)) != len(args.episode):
        parser.error("--episode lists the same episode twice.")
    try:
        selection = resolve_dataset_selection(args.dataset, revision=args.revision)
    except ValueError as exc:
        parser.error(str(exc))
    args.repo_id = selection.repo_id
    args.dataset_root = selection.root
    args.dataset_info = dataset_info(selection.root)
    args.external_layout = resolve_state_layout(args, args.dataset_info)
    args.robot = resolved_robot(args, args.dataset_info)
    return args


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    print(describe_plan(plan, args))
    if args.dry_run:
        print("[replay-real] dry run: no hardware was touched.")
        return
    confirm_motion(plan, assume_yes=args.yes)
    reports = run_plan(plan, args)
    failed = [report.episode for report in reports if not report.passed]
    if failed:
        print(
            f"[replay-real] {len(failed)} of {len(reports)} episode(s) exceeded the "
            f"tolerances: {', '.join(map(str, failed))}."
        )
        if args.strict:
            raise SystemExit(1)
    else:
        print(
            f"[replay-real] all {len(reports)} episode(s) tracked within "
            f"{args.tolerance_deg:g} deg / {args.tolerance_mm:g} mm."
        )


if __name__ == "__main__":
    main()
