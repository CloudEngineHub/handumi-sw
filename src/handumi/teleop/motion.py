"""Shared motion-pipeline configuration for every live teleop frontend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

from handumi.teleop.common import DEFAULT_TELEOP_FPS, TeleopMotionSmoother
from handumi.teleop.trajectory import TeleopCommandStream

DEFAULT_COMMAND_RATE_HZ = 100.0
# One 30 Hz source interval plus a small scheduling margin is enough to
# interpolate between consecutive IK solutions without the 80 ms latency that
# was previously used by simulation and recording.
DEFAULT_TRAJECTORY_DELAY_MS = 60.0
DEFAULT_COMMAND_EMA_TIME_CONSTANT_S = 0.02
DEFAULT_POSITION_DEADBAND_MM = 0.5
DEFAULT_ORIENTATION_DEADBAND_DEG = 0.25


@dataclass(frozen=True)
class TeleopMotionConfig:
    """Rates and filters shared by sim, real, and recording teleoperation."""

    input_rate_hz: float = DEFAULT_TELEOP_FPS
    command_rate_hz: float = DEFAULT_COMMAND_RATE_HZ
    trajectory_delay_s: float = DEFAULT_TRAJECTORY_DELAY_MS / 1000.0
    command_ema_time_constant_s: float = DEFAULT_COMMAND_EMA_TIME_CONSTANT_S
    position_deadband_m: float = DEFAULT_POSITION_DEADBAND_MM / 1000.0
    orientation_deadband_rad: float = np.deg2rad(DEFAULT_ORIENTATION_DEADBAND_DEG)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TeleopMotionConfig:
        """Build normalized SI-unit configuration from shared CLI arguments."""
        return cls(
            input_rate_hz=float(args.fps),
            command_rate_hz=float(args.command_rate_hz),
            trajectory_delay_s=float(args.trajectory_delay_ms) / 1000.0,
            command_ema_time_constant_s=float(
                args.motion_smoothing_time_constant_s
            ),
            position_deadband_m=float(args.motion_position_deadband_mm) / 1000.0,
            orientation_deadband_rad=float(
                np.deg2rad(args.motion_orientation_deadband_deg)
            ),
        )

    def make_input_smoother(self) -> TeleopMotionSmoother:
        """Create the common pre-IK jitter gate.

        EMA belongs to the 100 Hz command stream so it advances between IK
        results. The input stage therefore applies only pose deadbands.
        """
        return TeleopMotionSmoother(
            0.0,
            position_deadband_m=self.position_deadband_m,
            orientation_deadband_rad=self.orientation_deadband_rad,
        )

    def make_command_stream(
        self,
        write: Callable[[np.ndarray, dict[str, float]], None],
    ) -> TeleopCommandStream:
        """Create the common interpolated, fixed-rate output stage."""
        return TeleopCommandStream(
            write,
            command_rate_hz=self.command_rate_hz,
            delay_s=self.trajectory_delay_s,
            ema_time_constant_s=self.command_ema_time_constant_s,
        )


def add_teleop_motion_arguments(
    parser: argparse.ArgumentParser,
    *,
    help_transform: Callable[[str], str] | None = None,
) -> None:
    """Add the identical live-motion controls to a teleop CLI parser."""
    transform = help_transform or (lambda text: text)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_TELEOP_FPS,
        help=transform("Tracking/IK processing frequency."),
    )
    parser.add_argument(
        "--command-rate-hz",
        type=float,
        default=DEFAULT_COMMAND_RATE_HZ,
        help=transform("Fixed-rate playback frequency for interpolated joints."),
    )
    parser.add_argument(
        "--trajectory-delay-ms",
        type=float,
        default=DEFAULT_TRAJECTORY_DELAY_MS,
        help=transform("Small playback window used to interpolate adjacent IK results."),
    )
    parser.add_argument(
        "--motion-smoothing-time-constant-s",
        type=float,
        default=DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
        help=transform("100 Hz joint-command EMA time constant; 0 disables it."),
    )
    parser.add_argument(
        "--motion-position-deadband-mm",
        type=float,
        default=DEFAULT_POSITION_DEADBAND_MM,
        help=transform("Ignore controller translation jitter below this distance."),
    )
    parser.add_argument(
        "--motion-orientation-deadband-deg",
        type=float,
        default=DEFAULT_ORIENTATION_DEADBAND_DEG,
        help=transform("Ignore controller rotation jitter below this angle."),
    )


def validate_teleop_motion_args(args: argparse.Namespace) -> None:
    """Validate the shared motion arguments with CLI-friendly errors."""
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.command_rate_hz <= 0.0:
        raise SystemExit("--command-rate-hz must be > 0.")
    if args.trajectory_delay_ms < 0.0:
        raise SystemExit("--trajectory-delay-ms must be >= 0.")
    if args.motion_smoothing_time_constant_s < 0.0:
        raise SystemExit("--motion-smoothing-time-constant-s must be >= 0.")
    if args.motion_position_deadband_mm < 0.0:
        raise SystemExit("--motion-position-deadband-mm must be >= 0.")
    if args.motion_orientation_deadband_deg < 0.0:
        raise SystemExit("--motion-orientation-deadband-deg must be >= 0.")


__all__ = [
    "DEFAULT_COMMAND_EMA_TIME_CONSTANT_S",
    "DEFAULT_COMMAND_RATE_HZ",
    "DEFAULT_ORIENTATION_DEADBAND_DEG",
    "DEFAULT_POSITION_DEADBAND_MM",
    "DEFAULT_TRAJECTORY_DELAY_MS",
    "TeleopMotionConfig",
    "add_teleop_motion_arguments",
    "validate_teleop_motion_args",
]
