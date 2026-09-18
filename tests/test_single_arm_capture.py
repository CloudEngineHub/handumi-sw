"""Right-only capture must not require or silently validate a left sensor."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from handumi.config import dataset_active_sides, resolve_active_sides
from handumi.dataset.quality import validate_episode
from handumi.dataset.reader import _restore_enabled_signals
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    assert_calibrated,
)
from handumi.feetech.gripper import FeetechGripperPair, GripperWidths
from handumi.scripts.record import _tracking_healthy, build_observation
from handumi.tracking.base import ControllerPairSample


def test_side_selection_and_legacy_default():
    assert dataset_active_sides({}) == ("left", "right")
    assert resolve_active_sides(available=("right",)) == ("right",)
    for side in ("left", "both"):
        with pytest.raises(ValueError, match="unavailable"):
            resolve_active_sides(side, available=("right",))
    with pytest.raises(ValueError):
        dataset_active_sides({"active_sides": []})


def test_only_right_encoder_is_opened_read_and_calibrated(monkeypatch):
    opened = []
    reads = []

    class Bus:
        def __init__(self, *, port, **kwargs):
            assert port == "right-device"

        def open(self):
            opened.append("right")

        def close(self):
            pass

        def read_position(self, servo_id, **kwargs):
            reads.append(servo_id)
            return 150

    monkeypatch.setattr("handumi.feetech.gripper.FeetechBus", Bus)
    config = FeetechConfig(
        port=None,
        baudrate=1_000_000,
        protocol_version=0,
        left=GripperCalibration(servo_id=0),
        right=GripperCalibration(
            servo_id=1,
            port="right-device",
            closed_ticks=100,
            open_ticks=200,
            max_width_mm=80,
        ),
    )
    assert_calibrated(config, active_sides=("right",))
    with pytest.raises(SystemExit, match="left"):
        assert_calibrated(config)
    with FeetechGripperPair(config, active_sides=("right",)) as grippers:
        widths = grippers.read_normalized_widths_fast()
    assert opened == ["right"] and reads == [1]
    assert widths.left == 0 and widths.right == pytest.approx(0.04)


def test_capture_neutralizes_absent_controller_and_keeps_it_untracked():
    sample = replace(
        ControllerPairSample.empty("pico"),
        right_tracked=True,
        left_controller_pose=np.full(7, np.nan),
        left_tracked=True,
    )
    assert _tracking_healthy(replace(sample, left_tracked=False), ("right",))
    assert not _tracking_healthy(replace(sample, left_tracked=False))
    frame = build_observation(sample, GripperWidths.zero(), ("right",))
    np.testing.assert_array_equal(frame["observation.state"][:7], [0, 0, 0, 0, 0, 0, 1])
    assert frame["observation.tracking.left_tracked"][0] == 0
    assert np.isfinite(frame["observation.state"]).all()


def test_quality_only_checks_active_side_but_still_rejects_right_tracking_loss():
    n = 120
    states = np.zeros((n, 16))
    states[:, [6, 13]] = 1
    states[:, 7] = np.linspace(0.3, 0.4, n)
    states[:, 15] = np.linspace(0, 0.06, n)
    signals = {
        "observation.tracking.left_tracked": np.zeros(n),
        "observation.tracking.right_tracked": np.ones(n),
    }
    _restore_enabled_signals(signals, {"active_sides": ["right"], "sources": {}}, n)
    report = validate_episode(states, fps=30, signals=signals)
    assert report.accepted
    assert not any("freeze" in finding.code for finding in report.findings)
    assert "max_pose_freeze_s.left" not in report.metrics
    signals["observation.tracking.right_tracked"][:] = 0
    rejected = validate_episode(states, fps=30, signals=signals)
    assert not rejected.accepted
    assert any(f.code == "tracking_quality_fraction" for f in rejected.findings)
    signals["observation.tracking.right_tracked"][:] = 1
    states[:, 7] = 0.3
    frozen = validate_episode(states, fps=30, signals=signals)
    assert any(f.code == "full_pose_freeze" for f in frozen.findings)


def test_recording_defaults_and_resume_preserve_side(tmp_path, monkeypatch):
    import sys

    import yaml

    from handumi.scripts import record

    rig = tmp_path / "rig.yaml"
    rig.write_text(yaml.safe_dump({"recording": {"robot": "franka_emika_panda"}}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["record", "--rig-config", str(rig), "--output-dir", str(tmp_path / "data")],
    )
    args = record._resolve_recording_args(record.parse_args())
    assert args.active_sides == ("right",)
    assert args.cameras == ["right_wrist"]
    values = record._recording_values_from_dataset({}, {"active_sides": ["right"]})
    assert values["side"] == "right"
    # An older recording still requires both sides, even on a single-arm rig.
    assert record._recording_values_from_dataset({}, {})["side"] == "both"


def test_merge_rejects_different_capture_sides(tmp_path):
    from handumi.dataset.merging import _require_compatible

    with pytest.raises(ValueError, match="active_sides"):
        _require_compatible(
            {"handumi": {}},
            {"handumi": {"active_sides": ["right"]}},
            tmp_path,
            tmp_path,
        )


def test_empty_capture_is_reported_as_rejected():
    signals = {}
    _restore_enabled_signals(signals, {"active_sides": ["right"], "sources": {}}, 0)
    report = validate_episode(np.zeros((0, 16)), fps=30, signals=signals)
    assert not report.accepted
    assert any(f.code == "episode_too_short" for f in report.findings)
