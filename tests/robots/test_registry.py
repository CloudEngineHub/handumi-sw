"""Registry-level embodiment checks (docs/source/development/new_embodiment.md §3).

Currently exercises the ``metal`` embodiment; extend the module-level list as
new simulation-only robots are added.
"""

from __future__ import annotations

import numpy as np
import pytest

from handumi.robots.registry import available_robot_names, load_embodiment

EMBODIMENTS = ("metal",)


@pytest.fixture(scope="module", params=EMBODIMENTS)
def runtime(request):
    return load_embodiment(request.param)


def test_registered(runtime) -> None:
    assert runtime.name in available_robot_names()


def test_arm_joint_order_and_indices(runtime) -> None:
    names = list(runtime.robot.joints.actuated_names)
    for side in ("left", "right"):
        arm_names = runtime.arm_joint_names(side)
        indices = runtime.arm_joint_indices(side)
        assert [names[i] for i in indices] == arm_names
        assert all(name.startswith(f"{side}_") for name in arm_names)
    left = set(runtime.arm_joint_indices("left"))
    right = set(runtime.arm_joint_indices("right"))
    assert not left & right
    assert len(left | right) == runtime.robot.joints.num_actuated_joints


def test_home_q_within_limits(runtime) -> None:
    home = runtime.home_q()
    assert home.shape == (runtime.robot.joints.num_actuated_joints,)
    lower = np.asarray(runtime.robot.joints.lower_limits)
    upper = np.asarray(runtime.robot.joints.upper_limits)
    assert np.all(home >= lower - 1e-6)
    assert np.all(home <= upper + 1e-6)


def test_home_fk_is_left_right_symmetric(runtime) -> None:
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    left, right = solver.fk_pose7(runtime.home_q())
    # Mirror across the X=0 world plane: same height and forward reach.
    np.testing.assert_allclose(left[0], -right[0], atol=1e-3)
    np.testing.assert_allclose(left[1:3], right[1:3], atol=1e-3)


def test_gripper_mapping(runtime) -> None:
    q = runtime.home_q()
    runtime.set_finger_positions(q, {"left": 1.0, "right": 0.0})
    for finger in runtime.finger_joints["left"]:
        assert q[finger.index] == pytest.approx(finger.open_value)
    for finger in runtime.finger_joints["right"]:
        assert q[finger.index] == pytest.approx(finger.closed_value)


def test_urdf_meshes_resolve(runtime) -> None:
    urdf = runtime.load_urdf(load_meshes=True)
    assert len(urdf.scene.geometry) > 0
    for name, geometry in urdf.scene.geometry.items():
        assert geometry.vertices.shape[0] > 0, f"empty mesh for {name}"


def test_ik_converges_near_home(runtime) -> None:
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    offset = np.array([0.0, 0.03, -0.03], dtype=np.float32)
    left_target = (left[:3] + offset, left[3:7])
    right_target = (right[:3] + offset, right[3:7])
    q = home
    for _ in range(5):
        q = np.asarray(
            solver.ik(q, left_pose=left_target, right_pose=right_target),
            dtype=np.float32,
        )
    l_sol, r_sol = solver.fk_pose7(q)
    assert float(np.linalg.norm(l_sol[:3] - left_target[0])) < 0.002
    assert float(np.linalg.norm(r_sol[:3] - right_target[0])) < 0.002
