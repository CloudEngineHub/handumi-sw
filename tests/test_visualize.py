from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rerun as rr

from handumi.scripts import visualize


@pytest.fixture
def hardware(monkeypatch):
    config = SimpleNamespace(
        left=SimpleNamespace(max_width_mm=66),
        right=SimpleNamespace(max_width_mm=66),
    )
    pair = Mock()
    pair.__enter__ = Mock(return_value=pair)
    pair.__exit__ = Mock(return_value=False)
    pair.read_normalized_widths.return_value = SimpleNamespace(left_mm=12, right_mm=34)
    cameras = [Mock(), Mock()]
    for camera in cameras:
        camera.sample_at.return_value = SimpleNamespace(
            image=np.zeros((8, 8, 3), dtype=np.uint8)
        )
    monkeypatch.setattr(visualize, "load_config", lambda _: config)
    monkeypatch.setattr(visualize, "assert_calibrated", lambda _: None)
    monkeypatch.setattr(visualize, "resolve_camera_ids", lambda *a, **k: [0, 2])
    monkeypatch.setattr(
        visualize, "build_camera_specs", lambda *a, **k: ([{}, {}], None)
    )
    monkeypatch.setattr(visualize, "FeetechGripperPair", lambda _: pair)
    monkeypatch.setattr(visualize, "make_camera_device", Mock(side_effect=cameras))
    for name in ("init", "log", "send_blueprint", "set_time", "rerun_shutdown"):
        monkeypatch.setattr(rr, name, Mock())
    return pair, cameras


def test_camera_and_width_preview_logs_both_sides_and_closes_on_interrupt(hardware):
    pair, cameras = hardware
    pair.read_normalized_widths.side_effect = [
        SimpleNamespace(left_mm=12, right_mm=34),
        KeyboardInterrupt,
    ]
    with pytest.raises(KeyboardInterrupt):
        visualize.run(visualize.parse_args([]))
    paths = [call.args[0] for call in rr.log.call_args_list]
    assert "observation.images.left_wrist" in paths
    assert "observation.images.right_wrist" in paths
    assert paths.count("/observation.feetech.left_width_mm") == 2
    assert paths.count("/observation.feetech.right_width_mm") == 2
    for camera in cameras:
        camera.disconnect.assert_called_once()
    pair.__exit__.assert_called_once()
    rr.rerun_shutdown.assert_called_once()


def test_camera_start_failure_releases_already_open_devices(hardware):
    pair, cameras = hardware
    cameras[1].connect.side_effect = RuntimeError("camera unavailable")
    with pytest.raises(RuntimeError, match="camera unavailable"):
        visualize.run(visualize.parse_args([]))
    for camera in cameras:
        camera.disconnect.assert_called_once()
    pair.__exit__.assert_called_once()
    rr.init.assert_not_called()
