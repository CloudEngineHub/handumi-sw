import argparse

import numpy as np
import pytest

from handumi.teleop.motion import (
    DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_MAX_EXTRAPOLATION_MS,
    DEFAULT_ORIENTATION_DEADBAND_DEG,
    DEFAULT_POSITION_DEADBAND_MM,
    DEFAULT_TRAJECTORY_DELAY_MS,
    TeleopMotionConfig,
    add_teleop_motion_arguments,
    validate_teleop_motion_args,
)


def _args(*values: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_teleop_motion_arguments(parser)
    return parser.parse_args(values)


def test_shared_motion_defaults_build_one_normalized_configuration():
    args = _args()

    config = TeleopMotionConfig.from_args(args)

    assert config.input_rate_hz == 30.0
    assert config.command_rate_hz == DEFAULT_COMMAND_RATE_HZ == 100.0
    assert config.trajectory_delay_s == pytest.approx(
        DEFAULT_TRAJECTORY_DELAY_MS / 1000.0
    )
    assert config.max_extrapolation_s == pytest.approx(
        DEFAULT_MAX_EXTRAPOLATION_MS / 1000.0
    )
    assert config.command_ema_time_constant_s == DEFAULT_COMMAND_EMA_TIME_CONSTANT_S
    assert config.position_deadband_m == pytest.approx(
        DEFAULT_POSITION_DEADBAND_MM / 1000.0
    )
    assert config.orientation_deadband_rad == pytest.approx(
        np.deg2rad(DEFAULT_ORIENTATION_DEADBAND_DEG)
    )


def test_input_smoother_only_gates_jitter_before_ik():
    config = TeleopMotionConfig()

    smoother = config.make_input_smoother()

    assert smoother.time_constant_s == 0.0
    assert smoother.position_deadband_m == config.position_deadband_m
    assert smoother.orientation_deadband_rad == config.orientation_deadband_rad


def test_config_builds_the_fixed_rate_ema_command_stream():
    config = TeleopMotionConfig()

    stream = config.make_command_stream(lambda q, openings: None)

    assert stream.player.command_rate_hz == config.command_rate_hz
    assert stream.player.buffer.delay_s == config.trajectory_delay_s
    assert (
        stream.player.buffer.max_extrapolation_s == config.max_extrapolation_s
    )
    assert (
        stream.player.ema_time_constant_s == config.command_ema_time_constant_s
    )


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--fps", "0"),
        ("--command-rate-hz", "0"),
        ("--trajectory-delay-ms", "-1"),
        ("--max-extrapolation-ms", "-1"),
        ("--motion-smoothing-time-constant-s", "-1"),
        ("--motion-position-deadband-mm", "-1"),
        ("--motion-orientation-deadband-deg", "-1"),
    ),
)
def test_shared_motion_validation_rejects_invalid_values(option, value):
    with pytest.raises(SystemExit):
        validate_teleop_motion_args(_args(option, value))
