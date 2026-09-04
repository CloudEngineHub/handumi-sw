from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

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
    selection = next(check for check in checks if check.name == "TCP selection")
    assert selection.detail.endswith(
        "configs/calibration/controller_tcp/pico_piper_beta.yaml"
    )


def test_tcp_verifier_rejects_broken_device_mirror(tmp_path: Path) -> None:
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
    assert mirror.status == "FAIL"
    assert "200.00 mm" in mirror.detail


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
