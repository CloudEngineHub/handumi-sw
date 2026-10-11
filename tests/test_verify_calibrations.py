from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

from handumi.calibration.spatial import calibration_hash, write_yaml
from handumi.scripts import cli
from handumi.scripts.setup import verify_calibrations
from handumi.tracking.base import TrackingProvider


def _tcp_args(path: Path | None = None) -> Namespace:
    return Namespace(
        controller_tcp_calibration=path,
        mirror_tolerance_m=0.005,
        max_tip_distance_m=0.35,
    )


def test_cli_exposes_complete_calibration_verifier() -> None:
    command = cli.COMMANDS[("calibrate", "verify")]

    assert command.module == "handumi.scripts.setup.verify_calibrations"


def test_piper_pico_selects_assembly_specific_calibration() -> None:
    checks: list[verify_calibrations.Check] = []

    calibration = verify_calibrations._verify_tcp(
        _tcp_args(),
        device="pico",
        robot="piper",
        checks=checks,
    )

    assert calibration is not None
    assert not [check for check in checks if check.status == "FAIL"]
    # Installation measurements may change; mirror diagnostics are advisory.
    mirror = next(check for check in checks if check.name == "TCP mirror")
    assert mirror.status in {"PASS", "WARN"}
    selection = next(check for check in checks if check.name == "TCP selection")
    assert selection.detail.endswith(
        "configs/calibration/controller_tcp/pico_piper_beta.yaml"
    )


def test_tcp_verifier_warns_about_nominal_mirror_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "tcp.yaml"
    path.write_text(
        """\
calibration:
  controller_to_gripper_tcp:
    left:
      position: [0.10, -0.20, -0.04]
      quaternion: [0.0, 0.0, 0.0, 1.0]
    right:
      position: [0.10, -0.20, -0.04]
      quaternion: [0.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )
    checks: list[verify_calibrations.Check] = []

    verify_calibrations._verify_tcp(
        _tcp_args(path),
        device="pico",
        robot="piper",
        checks=checks,
    )

    mirror = next(check for check in checks if check.name == "TCP mirror")
    assert mirror.status == "WARN"
    assert "200.00 mm" in mirror.detail


def test_spatial_verifier_does_not_require_workspace_camera(tmp_path: Path) -> None:
    rig = tmp_path / "rig.yaml"
    spatial_path = tmp_path / "spatial.yaml"
    session_path = tmp_path / "session.yaml"
    board = {
        "id": "test-board",
        "squares_x": 5,
        "squares_y": 7,
        "dictionary": "DICT_5X5_100",
        "square_length_m": 0.03,
        "marker_length_m": 0.015,
        "legacy_pattern": False,
    }
    rig.write_text(
        """\
cameras:
  left: {index_or_path: 0}
  right: {index_or_path: 1}
  top: {index_or_path: 2}
spatial_calibration:
  board:
    id: test-board
    squares_x: 5
    squares_y: 7
    dictionary: DICT_5X5_100
    square_length_m: 0.03
    marker_length_m: 0.015
    legacy_pattern: false
""",
        encoding="utf-8",
    )
    spatial = {
        "kind": "handumi_spatial_calibration",
        "board": board,
        "cameras": {
            "left": {"mean_error_px": 0.2},
            "right": {"mean_error_px": 0.3},
        },
        "controller_camera": {
            "left": {"metrics": {"translation_rms_mm": 1.0}},
            "right": {"metrics": {"translation_rms_mm": 1.5}},
        },
    }
    write_yaml(spatial_path, spatial)
    write_yaml(
        session_path,
        {
            "kind": "handumi_session_calibration",
            "tracking_device": "pico",
            "spatial_calibration_sha256": calibration_hash(spatial),
            "table_from_device": {
                "translation_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "metrics": {"translation_rms_mm": 2.0},
        },
    )
    args = Namespace(
        spatial=spatial_path,
        session=session_path,
        rig_config=rig,
        max_intrinsics_error_px=0.8,
        max_mount_rms_mm=8.0,
        max_session_rms_mm=8.0,
    )
    checks: list[verify_calibrations.Check] = []

    verify_calibrations._verify_spatial_and_session(args, "pico", checks)

    assert [check.status for check in checks] == ["PASS", "PASS"]
    assert all("workspace" not in check.name.lower() for check in checks)


def test_capture_position_uses_tcp_pose_median() -> None:
    poses = iter(
        [
            [0.10, 0.20, 0.30],
            [0.11, 0.19, 0.30],
            [0.09, 0.21, 0.30],
        ]
    )

    class Tracker:
        def latest(self):
            position = np.asarray(next(poses), dtype=np.float32)
            pose = np.r_[position, [0.0, 0.0, 0.0, 1.0]]
            return SimpleNamespace(
                streaming=True,
                left_tracked=True,
                left_tcp_pose=pose,
            )

    center, jitter_mm = verify_calibrations._capture_position(
        cast(TrackingProvider, Tracker()),
        "left",
        samples=3,
        interval_s=0.0,
    )

    np.testing.assert_allclose(center, [0.10, 0.20, 0.30])
    assert 11.0 < jitter_mm < 12.0
