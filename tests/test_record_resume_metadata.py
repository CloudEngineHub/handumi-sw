"""Regression tests for camera metadata used by ``handumi record --resume``."""

from __future__ import annotations

import json
from types import SimpleNamespace

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
