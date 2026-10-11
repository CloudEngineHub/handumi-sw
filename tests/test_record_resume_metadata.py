"""Regression tests for camera metadata used by ``handumi record --resume``."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from handumi.scripts import record


def test_camera_metadata_uses_per_camera_output_sizes():
    args = SimpleNamespace(
        resume=False,
        device="meta",
        active_sides=("left", "right"),
        record_audio=False,
        skip_feetech=False,
        cam_fps=30,
        cam_width=640,
        cam_height=480,
        tracking_loss_timeout_s=1.0,
        sync_lag_s=0.04,
        max_sync_skew_s=0.06,
        camera_stale_timeout_s=0.25,
        gripper_stale_timeout_s=0.1,
        sensor_loss_timeout_s=1.0,
        feetech_sample_hz=100.0,
    )
    specs = [
        {
            "name": "left_wrist",
            "id": "/dev/video0",
            "type": "opencv",
            "width": 1280,
            "height": 960,
            "output_width": 1280,
            "output_height": 960,
            "fps": 30,
        },
        {
            "name": "workspace",
            "id": "/dev/video4",
            "type": "zedmini",
            "width": 1344,
            "height": 376,
            "output_width": 672,
            "output_height": 376,
            "fps": 30,
        },
    ]

    metadata = record._resume_handumi_metadata(
        args=args,
        camera_specs=specs,
        calibration_metadata={},
        spatial_session_metadata=None,
        robot_metadata={},
    )

    assert "camera_resolution" not in metadata
    assert [
        (camera["output_height"], camera["output_width"])
        for camera in metadata["cameras"]
    ] == [(960, 1280), (376, 672)]


def test_resume_ignores_incorrect_legacy_global_camera_resolution(
    tmp_path, monkeypatch
):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "robot_type": "handumi_raw",
        "features": {},
        "handumi": {"camera_resolution": [480, 640]},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    monkeypatch.setattr(record, "_validate_finalized_lerobot_dataset", lambda _: None)

    record._validate_resume_target(
        root,
        fps=30,
        features={},
        handumi={},
    )


def test_updating_metadata_removes_legacy_global_camera_resolution(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"handumi": {"camera_resolution": [480, 640]}})
    )

    updated = record._update_info_json(root, {"camera_fps": 30})

    assert updated is not None
    assert "camera_resolution" not in updated["handumi"]


def test_resume_loads_current_session_file_instead_of_snapshot(tmp_path, monkeypatch):
    old = {"path": str(tmp_path / "session.yaml"), "sha256": "old"}
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30, "handumi": {"spatial_session_calibration": old},
    }))
    monkeypatch.setattr(record, "load_optional_rig_section", lambda *a: {})
    monkeypatch.setattr(record, "_robot_metadata", lambda _: {
        "configuration": {"arms": {"left": {}, "right": {}}},
    })
    args = record._resolve_recording_args(record.parse_args([
        "--output-dir", str(root), "--resume",
    ]))
    assert args.session_calibration == Path(old["path"])
    current = {**old, "sha256": "new"}
    monkeypatch.setattr(record, "session_calibration_metadata", lambda p: current)
    assert record._resolve_spatial_session_metadata(args) == current


def test_resume_accepts_changed_session_but_rejects_changed_mount(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    old = {"sha256": "old", "spatial_calibration_sha256": "mount",
           "tracking_device": "pico", "workspace_frame": "table", "board_id": "board"}
    metadata = {"spatial_session_calibration": old}
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30, "robot_type": "handumi_raw", "features": {}, "handumi": metadata,
    }))
    monkeypatch.setattr(record, "_validate_finalized_lerobot_dataset", lambda _: None)
    record._validate_resume_target(root, fps=30, features={}, handumi=metadata)
    new = deepcopy(metadata)
    new["spatial_session_calibration"]["sha256"] = "new"
    record._validate_resume_target(root, fps=30, features={}, handumi=new)
    new["spatial_session_calibration"]["spatial_calibration_sha256"] = "different-mount"
    with pytest.raises(SystemExit, match="spatial_session_calibration"):
        record._validate_resume_target(root, fps=30, features={}, handumi=new)


def test_resume_missing_session_does_not_fall_back_to_old_snapshot(tmp_path):
    args = SimpleNamespace(resume=True, session_calibration=tmp_path / "missing.yaml",
                           _resume_handumi={"spatial_session_calibration": {"sha256": "old"}})
    with pytest.raises(SystemExit, match="Invalid session calibration"):
        record._resolve_spatial_session_metadata(args)


def test_history_preserves_legacy_and_each_new_episode_range():
    old = {"sha256": "old", "table_from_device": {"translation_m": [0, 0, 0]}}
    new = {"sha256": "new", "table_from_device": {"translation_m": [0, 0, 1]}}
    previous = {"spatial_session_calibration": old}
    current = {"spatial_session_calibration": new}
    record._record_session_history(current, previous=previous, first_episode=5, end_episode=8)
    assert current["spatial_session_calibration"] == old
    assert current["recording_sessions"] == [
        {"episode_from": 0, "episode_to": 5, "spatial_session_calibration": old},
        {"episode_from": 5, "episode_to": 8, "spatial_session_calibration": new},
    ]
    aborted = {"spatial_session_calibration": {"sha256": "unused"}}
    record._record_session_history(aborted, previous=current, first_episode=8, end_episode=8)
    assert aborted == current
    assert previous == {"spatial_session_calibration": old}
