import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handumi.dataset.capture import SYNC_LAG_S
from handumi.feetech import GripperWidths
from handumi.scripts.teleop_real import parse_args as parse_real_args
from handumi.scripts.teleop_record import (
    _validate_record_args,
    _validate_resume_dataset,
    build_features,
    build_joint_frame,
    joint_state_feature,
    parse_args,
)
from handumi.teleop import (
    DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_ORIENTATION_DEADBAND_DEG,
    DEFAULT_POSITION_DEADBAND_MM,
    DEFAULT_TELEOP_FPS,
    DEFAULT_TRAJECTORY_DELAY_MS,
)


def _widths() -> GripperWidths:
    return GripperWidths(
        left=0.01,
        right=0.02,
        left_mm=10.0,
        right_mm=20.0,
        left_normalized=0.25,
        right_normalized=0.5,
        left_ticks=11,
        right_ticks=22,
    )


class TeleopRecordSchemaTest(unittest.TestCase):
    def test_physical_teleop_defaults_match_live_teleop(self):
        args = parse_args(["--device", "pico", "--output-dir", "outputs/capture"])
        real_args = parse_real_args(["--device", "pico"])

        self.assertEqual(args.pico_mode, "mandos")
        self.assertTrue(args.pico_adb)
        self.assertFalse(args.pico_wifi)
        self.assertFalse(args.skip_feetech)
        self.assertFalse(args.space_start)
        self.assertEqual(args.translation_scale, 1.5)
        self.assertEqual(args.fps, DEFAULT_TELEOP_FPS)
        self.assertEqual(args.command_rate_hz, DEFAULT_COMMAND_RATE_HZ)
        self.assertEqual(args.trajectory_delay_ms, DEFAULT_TRAJECTORY_DELAY_MS)
        self.assertEqual(
            args.motion_smoothing_time_constant_s,
            DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
        )
        self.assertEqual(args.motion_position_deadband_mm, DEFAULT_POSITION_DEADBAND_MM)
        self.assertEqual(
            args.motion_orientation_deadband_deg,
            DEFAULT_ORIENTATION_DEADBAND_DEG,
        )
        self.assertEqual(args.fps, 30)
        self.assertEqual(args.sync_lag_s, SYNC_LAG_S)
        self.assertEqual(args.feetech_sample_hz, 100.0)

        for attribute in (
            "robot",
            "home_pose",
            "side",
            "fps",
            "command_rate_hz",
            "trajectory_delay_ms",
            "motion_smoothing_time_constant_s",
            "motion_position_deadband_mm",
            "motion_orientation_deadband_deg",
            "translation_scale",
            "translation_deadzone_mm",
            "tracking_stale_ms",
            "space_start",
            "no_sounds",
            "controller_tcp_calibration",
            "rig_config",
            "feetech_port",
            "skip_feetech",
            "quest_ip",
            "tcp_port",
            "sync_port",
            "pico_mode",
            "pico_wifi",
            "skip_adb_check",
            "skip_can_repair",
        ):
            self.assertEqual(getattr(args, attribute), getattr(real_args, attribute))

    def test_physical_teleop_options_are_configurable(self):
        args = parse_args(
            [
                "--device",
                "pico",
                "--output-dir",
                "outputs/capture",
                "--space-start",
                "--skip-feetech",
                "--pico-wifi",
                "--translation-scale",
                "2.25",
                "--no-sounds",
                "--skip-can-repair",
            ]
        )

        _validate_record_args(args)
        self.assertTrue(args.space_start)
        self.assertTrue(args.skip_feetech)
        self.assertTrue(args.pico_wifi)
        self.assertFalse(args.pico_adb)
        self.assertEqual(args.translation_scale, 2.25)
        self.assertTrue(args.no_sounds)
        self.assertTrue(args.skip_can_repair)

    def test_rerun_camera_options_are_shared_with_live_teleop(self):
        record_args = parse_args(
            [
                "--device",
                "pico",
                "--output-dir",
                "outputs/capture",
                "--cameras",
                "left_wrist,workspace",
                "--cam-width",
                "320",
                "--cam-height",
                "240",
            ]
        )
        real_args = parse_real_args(
            [
                "--device",
                "pico",
                "--cameras",
                "left_wrist,workspace",
                "--cam-width",
                "320",
                "--cam-height",
                "240",
            ]
        )

        self.assertEqual(record_args.cameras, ["left_wrist", "workspace"])
        self.assertEqual(record_args.cameras, real_args.cameras)
        self.assertEqual(record_args.cam_width, real_args.cam_width)
        self.assertEqual(record_args.cam_height, real_args.cam_height)

    def test_skip_cameras_matches_live_teleop(self):
        record_args = parse_args(
            [
                "--device",
                "pico",
                "--output-dir",
                "outputs/capture",
                "--skip-cameras",
            ]
        )
        real_args = parse_real_args(["--device", "pico", "--skip-cameras"])

        self.assertTrue(record_args.skip_cameras)
        self.assertEqual(record_args.skip_cameras, real_args.skip_cameras)

    def test_explicit_camera_selection_conflicts_with_no_rerun(self):
        args = parse_args(
            [
                "--device",
                "pico",
                "--output-dir",
                "outputs/capture",
                "--cameras",
                "workspace",
                "--no-rerun",
            ]
        )

        with self.assertRaises(SystemExit):
            _validate_record_args(args)

    def test_skip_feetech_requires_space_start(self):
        args = parse_args(
            ["--device", "pico", "--output-dir", "outputs/capture", "--skip-feetech"]
        )
        with self.assertRaises(SystemExit):
            _validate_record_args(args)

    def test_trajectory_configuration_is_validated(self):
        args = parse_args(
            [
                "--device",
                "pico",
                "--output-dir",
                "outputs/capture",
                "--command-rate-hz",
                "0",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_record_args(args)

        args = parse_args(
            [
                "--device",
                "pico",
                "--output-dir",
                "outputs/capture",
                "--trajectory-delay-ms",
                "-1",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_record_args(args)

    def test_resume_requires_a_finalized_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            with self.assertRaisesRegex(SystemExit, "does not exist"):
                _validate_resume_dataset(root)

            (root / "meta" / "episodes").mkdir(parents=True)
            (root / "meta" / "tasks.parquet").touch()
            (root / "data").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 2}), encoding="utf-8"
            )
            _validate_resume_dataset(root)

    def test_joint_state_feature_uses_robot_joint_names(self):
        self.assertEqual(
            joint_state_feature(["left_joint1", "left_joint2"]),
            {
                "dtype": "float32",
                "shape": (2,),
                "names": ["left_joint1", "left_joint2"],
            },
        )

    def test_features_store_joint_feedback_and_joint_action(self):
        features = build_features(
            ["left_wrist"],
            cam_width=320,
            cam_height=240,
            use_videos=True,
            joint_names=["left_joint1", "right_joint1"],
        )

        self.assertEqual(features["observation.state"]["shape"], (2,))
        self.assertEqual(features["action"]["shape"], (2,))
        self.assertEqual(
            features["observation.state"]["names"],
            ["left_joint1", "right_joint1"],
        )
        self.assertEqual(features["action"]["names"], ["left_joint1", "right_joint1"])
        self.assertEqual(features["observation.images.left_wrist"]["dtype"], "video")
        self.assertFalse(
            any(
                key.startswith("observation.tracking") or key == "observation.valid"
                for key in features
            )
        )

    def test_joint_frame_keeps_observation_and_action_separate(self):
        frame = build_joint_frame(
            observation_q=np.array([1.0, 2.0], dtype=np.float64),
            action_q=np.array([3.0, 4.0], dtype=np.float64),
            widths=_widths(),
        )

        np.testing.assert_array_equal(
            frame["observation.state"], np.array([1.0, 2.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            frame["action"], np.array([3.0, 4.0], dtype=np.float32)
        )
        self.assertEqual(frame["observation.feetech.left_ticks"].item(), 11)
        self.assertEqual(frame["observation.feetech.right_normalized"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
