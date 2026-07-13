from argparse import Namespace
from pathlib import Path

import numpy as np

from handumi.scripts.replay.replay_in_sim import (
    _metadata_tcp_calibration,
    _render_task_scene,
    _resolve_gripper_openings,
    _resolved_controller_device,
    _resolved_retarget_mode,
    _resolved_tcp_calibration,
    load_robot_from_table,
)


def test_recorded_normalized_grippers_take_precedence():
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 14:16] = 0.08
    recorded = np.array([[0.2, 0.7], [0.3, 0.8]], dtype=np.float32)

    openings, source = _resolve_gripper_openings(
        states, recorded, max_width_m=0.08
    )

    np.testing.assert_allclose(openings, recorded)
    assert source == "recorded Feetech normalized"


def test_grippers_fall_back_to_widths_in_meters():
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 14:16] = [[0.0, 0.033], [0.066, 0.099]]

    openings, source = _resolve_gripper_openings(
        states, None, max_width_m=0.066
    )

    np.testing.assert_allclose(openings, [[0.0, 0.5], [1.0, 1.0]], atol=1e-6)
    assert source == "state widths in meters"


def test_pose_only_state_has_no_gripper_opening():
    states = np.zeros((2, 14), dtype=np.float32)

    openings, source = _resolve_gripper_openings(
        states, None, max_width_m=0.08
    )

    assert openings is None
    assert source == "unavailable (legacy pose-only state)"


def test_controller_device_is_read_from_dataset_metadata():
    args = Namespace(controller_device=None)
    info = {"handumi": {"recording_device": "meta"}}

    assert _resolved_controller_device(args, info) == "meta"


def test_explicit_controller_device_overrides_metadata():
    args = Namespace(controller_device="pico")
    info = {"handumi": {"recording_device": "meta"}}

    assert _resolved_controller_device(args, info) == "pico"


def test_auto_retarget_uses_absolute_table_for_calibrated_table_dataset():
    args = Namespace(retarget_mode="auto")
    info = {"handumi": {"tracking_workspace": "table"}}

    assert _resolved_retarget_mode(args, info) == "absolute-table"


def test_auto_retarget_falls_back_to_local_relative_without_table_frame():
    args = Namespace(retarget_mode="auto")

    assert _resolved_retarget_mode(args, {"handumi": {}}) == "local-relative"


def test_explicit_retarget_mode_overrides_dataset_metadata():
    args = Namespace(retarget_mode="anchored")
    info = {"handumi": {"tracking_workspace": "table"}}

    assert _resolved_retarget_mode(args, info) == "anchored"


def test_controller_tcp_calibration_is_loaded_from_metadata():
    info = {
        "handumi": {
            "controller_tcp_calibration": {
                "sha256": "abc123",
                "applied_to_state": False,
                "controller_to_gripper_tcp": {
                    "left": {
                        "position": [0.1, 0.2, 0.3],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                    "right": {
                        "position": [-0.1, -0.2, -0.3],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        }
    }

    resolved = _metadata_tcp_calibration(info)

    assert resolved is not None
    calibration, source = resolved
    np.testing.assert_allclose(calibration.left[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(calibration.right[:3], [-0.1, -0.2, -0.3])
    assert source == "dataset metadata sha256=abc123"


def _metadata_calibration_info() -> dict[str, object]:
    return {
        "handumi": {
            "controller_tcp_calibration": {
                "sha256": "old-snapshot",
                "applied_to_state": False,
                "controller_to_gripper_tcp": {
                    "left": {
                        "position": [0.01, 0.02, 0.03],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                    "right": {
                        "position": [-0.01, -0.02, -0.03],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        }
    }


def test_piper_meta_configured_tcp_calibration_precedes_dataset_snapshot():
    from handumi.robots.registry import load_robot_config

    args = Namespace(
        controller_tcp_calibration=None,
        use_dataset_tcp_calibration=False,
    )
    configured = load_robot_config("piper").controller_tcp_calibrations["meta"]

    calibration, source = _resolved_tcp_calibration(
        args,
        _metadata_calibration_info(),
        robot="piper",
        controller_device="meta",
        configured_path=configured,
    )

    np.testing.assert_allclose(
        calibration.left[:3], [0.12068467, 0.02142489, -0.21669616]
    )
    assert source.startswith("configured piper/meta:")


def test_dataset_tcp_snapshot_can_be_requested_explicitly():
    args = Namespace(
        controller_tcp_calibration=None,
        use_dataset_tcp_calibration=True,
    )

    calibration, source = _resolved_tcp_calibration(
        args,
        _metadata_calibration_info(),
        robot="piper",
        controller_device="meta",
        configured_path=Path("configs/calibration/meta_controller_tcp.yaml"),
    )

    np.testing.assert_allclose(calibration.left[:3], [0.01, 0.02, 0.03])
    assert source == "dataset metadata sha256=old-snapshot"


def test_explicit_tcp_path_precedes_configured_and_dataset(tmp_path: Path):
    explicit = tmp_path / "explicit_tcp.yaml"
    explicit.write_text(
        """\
calibration:
  controller_to_gripper_tcp:
    left:
      position: [0.4, 0.5, 0.6]
      quaternion: [0.0, 0.0, 0.0, 1.0]
    right:
      position: [-0.4, -0.5, -0.6]
      quaternion: [0.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )
    args = Namespace(
        controller_tcp_calibration=explicit,
        use_dataset_tcp_calibration=False,
    )

    calibration, source = _resolved_tcp_calibration(
        args,
        _metadata_calibration_info(),
        robot="piper",
        controller_device="meta",
        configured_path=Path("configs/calibration/meta_controller_tcp.yaml"),
    )

    np.testing.assert_allclose(calibration.left[:3], [0.4, 0.5, 0.6])
    assert source == str(explicit)


def test_load_robot_from_table(tmp_path: Path):
    path = tmp_path / "deployment.yaml"
    path.write_text(
        """\
calibration:
  robot_from_table:
    position: [0.3, 0.0, 0.1]
    quaternion: [0.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )

    pose = load_robot_from_table(path)

    np.testing.assert_allclose(pose, [0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])


def test_absolute_table_parser_defaults_prepare_start_and_align_tools():
    from handumi.scripts.replay.replay_in_sim import build_parser

    args = build_parser().parse_args([])

    assert args.retarget_mode == "auto"
    assert args.use_dataset_tcp_calibration is False
    assert args.absolute_orientation == "relative-start"
    assert args.initial_solve_iterations == 12
    assert args.initial_position_tolerance_m == 0.01
    assert args.max_ik_position_error_m == 0.03
    assert args.max_ik_rotation_error_deg == 45.0
    assert args.table_clearance_warning_m == 0.10


def test_render_task_scene_maps_table_bodies_into_robot_world():
    class FakeScene:
        def __init__(self):
            self.frames = []
            self.boxes = []

        def add_frame(self, name, **kwargs):
            self.frames.append((name, kwargs))
            return object()

        def add_box(self, name, **kwargs):
            self.boxes.append((name, kwargs))
            return object()

    class FakeServer:
        def __init__(self):
            self.scene = FakeScene()

    server = FakeServer()
    args = Namespace(scene="cube_in_box")
    rollout = {
        "robot_from_table_pose7": np.array(
            [[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )
    }

    _render_task_scene(server, args, rollout)

    assert len(server.scene.frames) == 2
    assert len(server.scene.boxes) == 6
    cube = next(item for item in server.scene.frames if item[0].endswith("cube"))
    np.testing.assert_allclose(cube[1]["position"], [0.3, -0.1, 0.0], atol=1e-6)
