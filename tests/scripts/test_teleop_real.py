import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.feetech import GripperSample, GripperWidths
from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.scripts.teleop_real import (
    _enabled_tracking_ok,
    _load_required_calibration,
    _validate_feetech_ports_exist,
    _validate_real_args as _validate_args,
    parse_args,
)
from handumi.teleop.common import start_sides as _start_sides
from handumi.teleop.common import latest_widths as _latest_widths
from handumi.teleop.motion import (
    DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_ORIENTATION_DEADBAND_DEG,
    DEFAULT_POSITION_DEADBAND_MM,
    DEFAULT_TRAJECTORY_DELAY_MS,
)
from handumi.teleop.physical import (
    DEFAULT_TRACKING_STALE_MS,
    DEFAULT_TRANSLATION_DEADZONE_MM,
    DEFAULT_TRANSLATION_SCALE,
)


class TeleopRealArgsTest(unittest.TestCase):
    def test_latest_widths_uses_sampler_cache(self):
        widths = GripperWidths.zero()
        sampler = mock.Mock()
        sampler.latest.return_value = GripperSample(
            widths=widths,
            sample_time_ns=123,
            sequence=1,
        )

        self.assertIs(_latest_widths(sampler), widths)
        sampler.read_normalized_widths.assert_not_called()

    def test_defaults_target_piper_without_space_start(self):
        args = parse_args(["--device", "pico"])

        self.assertEqual(args.robot, "piper")
        self.assertEqual(args.fps, 30)
        self.assertEqual(args.command_rate_hz, DEFAULT_COMMAND_RATE_HZ)
        self.assertEqual(args.translation_scale, DEFAULT_TRANSLATION_SCALE)
        self.assertEqual(
            args.translation_deadzone_mm, DEFAULT_TRANSLATION_DEADZONE_MM
        )
        self.assertEqual(args.tracking_stale_ms, DEFAULT_TRACKING_STALE_MS)
        self.assertEqual(args.trajectory_delay_ms, DEFAULT_TRAJECTORY_DELAY_MS)
        self.assertEqual(
            args.motion_smoothing_time_constant_s,
            DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
        )
        self.assertEqual(
            args.motion_position_deadband_mm, DEFAULT_POSITION_DEADBAND_MM
        )
        self.assertEqual(
            args.motion_orientation_deadband_deg,
            DEFAULT_ORIENTATION_DEADBAND_DEG,
        )
        self.assertFalse(args.space_start)
        _validate_args(args)

    def test_skip_cameras_is_available(self):
        args = parse_args(["--device", "pico", "--skip-cameras"])

        self.assertTrue(args.skip_cameras)
        _validate_args(args)

    def test_space_start_is_opt_in(self):
        args = parse_args(["--device", "pico", "--space-start"])

        self.assertTrue(args.space_start)
        _validate_args(args)

    def test_translation_scale_cli_override_is_parsed(self):
        args = parse_args(["--device", "pico", "--translation-scale", "2.25"])

        self.assertEqual(args.translation_scale, 2.25)

    def test_translation_scale_must_be_positive(self):
        args = parse_args(["--device", "pico", "--translation-scale", "0"])

        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_tracking_freshness_and_deadzone_are_validated(self):
        for option, value in (
            ("--tracking-stale-ms", "0"),
            ("--translation-deadzone-mm", "-1"),
        ):
            with self.assertRaises(SystemExit):
                _validate_args(parse_args(["--device", "pico", option, value]))

    def test_smoothing_configuration_cannot_be_negative(self):
        for option in (
            "--motion-smoothing-time-constant-s",
            "--motion-position-deadband-mm",
            "--motion-orientation-deadband-deg",
        ):
            args = parse_args(["--device", "pico", option, "-0.01"])
            with self.assertRaises(SystemExit):
                _validate_args(args)

    def test_trajectory_configuration_is_validated(self):
        args = parse_args(["--device", "pico", "--command-rate-hz", "0"])
        with self.assertRaises(SystemExit):
            _validate_args(args)

        args = parse_args(["--device", "pico", "--trajectory-delay-ms", "-1"])
        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_default_calibration_comes_from_piper_robot_tool_setup(self):
        args = parse_args(["--device", "meta"])

        calibration = _load_required_calibration(args)

        np.testing.assert_allclose(
            calibration.left[:3],
            [0.1206525, 0.02460851, -0.20575515],
        )

    def test_accepts_registered_openarm_backend(self):
        args = parse_args(["--device", "pico", "--robot", "openarmv1", "--space-start"])

        _validate_args(args)
        self.assertEqual(args.robot, "openarmv1")

    def test_skip_feetech_requires_space_start(self):
        args = parse_args(["--device", "pico", "--skip-feetech"])

        with self.assertRaises(SystemExit):
            _validate_args(args)

        args = parse_args(["--device", "pico", "--skip-feetech", "--space-start"])
        _validate_args(args)

    def test_space_starts_only_idle_arms(self):
        anchors = {"left": {"source": object()}, "right": None}

        self.assertEqual(_start_sides(anchors, ("left", "right")), ("right",))

    def test_tracking_loss_policy_requires_all_enabled_sides(self):
        self.assertFalse(
            _enabled_tracking_ok({"left": True, "right": False}, ("left", "right"))
        )

    def test_single_side_mode_only_requires_that_side_tracked(self):
        self.assertTrue(_enabled_tracking_ok({"left": True, "right": False}, ("left",)))
        self.assertFalse(
            _enabled_tracking_ok({"left": True, "right": False}, ("right",))
        )

    def test_feetech_port_validation_reports_missing_rig_ports(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyACM9"),
            right=GripperCalibration(1, 900, 1900, 75.0, "/dev/ttyACM8"),
        )

        with (
            mock.patch(
                "handumi.scripts.teleop_real.list_feetech_serial_ports",
                return_value={"/dev/ttyACM0"},
            ),
            self.assertRaisesRegex(SystemExit, "Remap Feetech"),
        ):
            _validate_feetech_ports_exist(config)

    def test_feetech_port_validation_accepts_existing_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "ttyACM0"
            right = Path(tmp) / "ttyACM1"
            left.touch()
            right.touch()
            config = FeetechConfig(
                port=None,
                baudrate=1_000_000,
                protocol_version=0,
                left=GripperCalibration(0, 1000, 2000, 80.0, str(left)),
                right=GripperCalibration(1, 900, 1900, 75.0, str(right)),
            )

            _validate_feetech_ports_exist(config)


if __name__ == "__main__":
    unittest.main()
