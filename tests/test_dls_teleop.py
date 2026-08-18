"""Safety and continuity checks for the real-teleop DLS follower."""

import numpy as np
import pytest

from handumi.robots.registry import load_embodiment
from handumi.teleop.dls import make_real_teleop_dls_solver


@pytest.fixture(scope="module")
def piper_dls():
    runtime = load_embodiment("piper")
    home_q = runtime.home_q()
    solver = make_real_teleop_dls_solver(runtime, runtime.solver_cls(), home_q)
    solver.warmup(home_q)
    return runtime, solver, home_q


def test_dls_step_is_bounded_and_does_not_move_inactive_arm(piper_dls) -> None:
    runtime, solver, home_q = piper_dls
    solver.set_timestep(1.0 / 30.0)
    left_pose, _ = solver.fk_pose7(home_q)
    target = left_pose.copy()
    target[:3] += np.array((0.08, -0.04, 0.03), dtype=np.float32)

    actual = solver.ik(home_q, left_pose=(target[:3], target[3:7]))

    left_indices = np.asarray(solver.side_joint_indices["left"], dtype=np.intp)
    right_indices = np.asarray(runtime.arm_joint_indices("right"), dtype=np.intp)
    speed_limits = solver.side_joint_speed_limits_rad_s["left"]
    assert np.all(
        np.abs(actual[left_indices] - home_q[left_indices])
        <= speed_limits / 30.0 + 1e-7
    )
    np.testing.assert_array_equal(actual[right_indices], home_q[right_indices])
    assert np.all(actual >= np.asarray(runtime.robot.joints.lower_limits) - 1e-7)
    assert np.all(actual <= np.asarray(runtime.robot.joints.upper_limits) + 1e-7)


def test_dls_repeated_steps_reduce_cartesian_error(piper_dls) -> None:
    _, solver, home_q = piper_dls
    solver.set_timestep(1.0 / 30.0)
    left_pose, _ = solver.fk_pose7(home_q)
    target = left_pose.copy()
    target[0] += 0.02
    q = home_q.copy()

    initial_error = float(np.linalg.norm(left_pose[:3] - target[:3]))
    for _ in range(20):
        q = solver.ik(q, left_pose=(target[:3], target[3:7]))
    achieved, _ = solver.fk_pose7(q)
    final_error = float(np.linalg.norm(achieved[:3] - target[:3]))

    assert final_error < initial_error * 0.1
    assert np.isfinite(q).all()


def test_dls_joint_cap_scales_with_source_frame_time() -> None:
    runtime = load_embodiment("piper")
    home_q = runtime.home_q()
    solver = make_real_teleop_dls_solver(
        runtime,
        runtime.solver_cls(),
        home_q,
        input_rate_hz=72.0,
    )
    left_pose, _ = solver.fk_pose7(home_q)
    target = left_pose.copy()
    target[:3] += np.array((0.20, -0.10, 0.10), dtype=np.float32)
    solver.set_timestep(1.0 / 72.0)

    actual = solver.ik(home_q, left_pose=(target[:3], target[3:7]))

    indices = np.asarray(solver.side_joint_indices["left"], dtype=np.intp)
    speeds = solver.side_joint_speed_limits_rad_s["left"]
    assert np.all(np.abs(actual[indices] - home_q[indices]) <= speeds / 72.0 + 1e-7)


def test_dls_speed_limits_are_derived_from_robot_and_backend() -> None:
    for robot_name in ("piper", "openarmv1"):
        runtime = load_embodiment(robot_name)
        solver = make_real_teleop_dls_solver(
            runtime,
            runtime.solver_cls(),
            runtime.home_q(),
            input_rate_hz=72.0,
        )
        urdf = np.asarray(runtime.robot.joints.velocity_limits)
        for side in ("left", "right"):
            indices = np.asarray(solver.side_joint_indices[side], dtype=np.intp)
            resolved = solver.side_joint_speed_limits_rad_s[side]
            assert np.all(resolved > 0.0)
            assert np.all(resolved <= urdf[indices])
            assert np.all(resolved <= 1.0)
