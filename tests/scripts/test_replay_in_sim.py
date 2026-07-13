from argparse import Namespace
from pathlib import Path

import numpy as np

from handumi.scripts.replay.replay_in_sim import (
    _metadata_tcp_calibration,
    _render_task_scene,
    _resolve_gripper_openings,
    _resolved_controller_device,
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
