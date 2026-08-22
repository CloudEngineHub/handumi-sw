from pathlib import Path

import pytest

from handumi.calibration.spatial import CharucoBoardSpec
from handumi.scripts.setup import calibrate_spatial


def _spatial(board: CharucoBoardSpec, *, source: int = 1) -> dict:
    return {
        "board": board.to_dict(),
        "cameras": {"left_wrist": {"index_or_path": source}},
    }


def test_validate_spatial_rig_rejects_board_scale_mismatch(monkeypatch) -> None:
    spatial_board = CharucoBoardSpec(square_length_m=0.03, marker_length_m=0.015)
    rig_board = CharucoBoardSpec(square_length_m=0.06, marker_length_m=0.03)
    monkeypatch.setattr(calibrate_spatial, "_board_from_rig", lambda _path: rig_board)

    with pytest.raises(
        SystemExit, match=r"(?s)square=30\.0 mm.*square=60\.0 mm"
    ):
        calibrate_spatial._validate_spatial_rig(
            _spatial(spatial_board),
            spatial_path=Path("spatial.yaml"),
            rig_path=Path("rig.yaml"),
            camera="left_wrist",
        )


def test_validate_spatial_rig_warns_when_camera_mapping_changed(
    monkeypatch, caplog
) -> None:
    board = CharucoBoardSpec(square_length_m=0.06, marker_length_m=0.03)
    monkeypatch.setattr(calibrate_spatial, "_board_from_rig", lambda _path: board)
    monkeypatch.setattr(calibrate_spatial, "_camera_source", lambda _path, _camera: 1)

    calibrate_spatial._validate_spatial_rig(
        _spatial(board, source=2),
        spatial_path=Path("spatial.yaml"),
        rig_path=Path("rig.yaml"),
        camera="left_wrist",
    )

    assert "mapped to 1" in caplog.text
    assert "captured from 2" in caplog.text
