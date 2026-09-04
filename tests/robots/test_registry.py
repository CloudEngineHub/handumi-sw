"""Registry-level embodiment checks (docs/source/development/new_embodiment.md §3).

Currently exercises the ``metal`` embodiment; extend the module-level list as
new simulation-only robots are added.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from handumi.robots.registry import (
    CONFIG_DIR,
    available_robot_names,
    load_embodiment,
    robot_config_metadata,
)

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


def test_posture_q_defaults_to_home(runtime) -> None:
    """A robot without a declared working posture keeps the legacy behavior."""
    np.testing.assert_allclose(runtime.config.posture_q, runtime.home_q())
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    np.testing.assert_allclose(solver.posture_q, runtime.home_q())


def test_piper_posture_is_the_teleop_arch_and_within_limits() -> None:
    piper = load_embodiment("piper")
    posture = piper.config.posture_q
    lower = np.asarray(piper.robot.joints.lower_limits)
    upper = np.asarray(piper.robot.joints.upper_limits)
    fixed = ~np.isnan(posture)
    assert np.all(posture[fixed] >= lower[fixed] - 1e-6)
    assert np.all(posture[fixed] <= upper[fixed] + 1e-6)
    for side in ("left", "right"):
        j1, j2, j3, j4, j5, j6, grip = posture[piper.arm_joint_indices(side)]
        assert j1 == 0.0  # the base is biased to stay put
        assert 80.0 < np.degrees(j2) < 120.0 and -100.0 < np.degrees(j3) < -70.0
        assert all(np.isnan(v) for v in (j4, j5, j6, grip))  # wrist left free
    assert piper.config.replay_ik_weights["posture"] > 0.0


def test_posture_cost_pulls_the_solution_toward_the_declared_posture() -> None:
    """Same TCP targets, warm start at the folded home: the posture term decides.

    The targets are the FK of posture_q itself, so posture_q solves the pose
    cost exactly and a posture term makes it the unique optimum. Without the
    term the solver settles wherever the warm start leads, so only the reach
    is checked there.
    """
    from dataclasses import replace

    piper = load_embodiment("piper")
    cfg = replace(piper.config.ik_weights, max_joint_delta=None, posture_weight=0.0)
    plain = piper.solver_cls(config=cfg)
    with_posture = piper.solver_cls(config=replace(cfg, posture_weight=20.0))
    from handumi.robots.kinematics import posture_seed

    posture = posture_seed(piper.config.posture_q, piper.home_q())
    left_target, right_target = plain.fk_pose7(posture)
    poses = {
        "left_pose": (left_target[:3], left_target[3:7]),
        "right_pose": (right_target[:3], right_target[3:7]),
    }
    q_plain = piper.home_q()
    q_post = piper.home_q()
    for _ in range(10):
        q_plain = plain.ik(q_plain, **poses)
        q_post = with_posture.ik(q_post, **poses)
    biased = np.flatnonzero(~np.isnan(piper.config.posture_q))
    assert np.max(np.abs(np.degrees(q_post[biased] - posture[biased]))) < 5.0
    for q in (q_plain, q_post):
        left, right = with_posture.fk_pose7(q)
        assert np.linalg.norm(left[:3] - left_target[:3]) < 0.01
        assert np.linalg.norm(right[:3] - right_target[:3]) < 0.01


def test_limit_margin_cost_keeps_joints_off_their_stops() -> None:
    """A target the wrist reaches on its stop is re-solved with the stop unused."""
    from dataclasses import replace

    from handumi.robots.kinematics import limit_margin_terms

    piper = load_embodiment("piper")
    base = replace(piper.config.ik_weights, max_joint_delta=None, limit_margin_weight=0.0)
    assert limit_margin_terms(piper.robot, base) is None
    margin = replace(base, limit_margin_weight=10.0, limit_margin_rad=np.radians(10.0))
    lower, upper, weights = (np.asarray(t) for t in limit_margin_terms(piper.robot, margin))
    fingers = [f.index for fs in piper.finger_joints.values() for f in fs]
    assert np.all(weights[fingers] == 0.0)  # a 3.5 cm range cannot hold a margin
    arm = piper.arm_joint_indices("left")[:6]
    assert np.all(weights[arm] == 10.0)
    assert np.all(lower[arm] > np.asarray(piper.robot.joints.lower_limits)[arm])
    assert np.all(upper[arm] < np.asarray(piper.robot.joints.upper_limits)[arm])

    # Pose the left wrist on its pitch stop and ask both solvers to hold that TCP.
    plain = piper.solver_cls(config=base)
    soft = piper.solver_cls(config=margin)
    q_stop = piper.home_q().copy()
    q_stop[arm[1]] = np.radians(90.0)
    q_stop[arm[2]] = np.radians(-60.0)
    q_stop[arm[4]] = np.asarray(piper.robot.joints.upper_limits)[arm[4]]  # +70 deg
    left_target, right_target = plain.fk_pose7(q_stop)
    poses = {"left_pose": (left_target[:3], left_target[3:7]), "right_pose": (right_target[:3], right_target[3:7])}
    q_plain, q_soft = q_stop.copy(), q_stop.copy()
    for _ in range(6):
        q_plain = plain.ik(q_plain, **poses)
        q_soft = soft.ik(q_soft, **poses)
    wrist_limit = np.asarray(piper.robot.joints.upper_limits)[arm[4]]
    assert wrist_limit - q_plain[arm[4]] < np.radians(1.0)  # stays parked on the stop
    # Backed off into the margin: the pose target still pins the wrist, so the
    # retreat is a couple of degrees, not the full margin.
    assert wrist_limit - q_soft[arm[4]] > np.radians(1.5)
    achieved, _ = soft.fk_pose7(q_soft)
    assert np.linalg.norm(achieved[:3] - left_target[:3]) < 0.01  # still on target


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


@pytest.mark.parametrize("other", ("piper", "yam"))
def test_collision_stays_opt_in_for_other_embodiments(other) -> None:
    """Robots without collision weights keep the legacy solver path exactly."""
    rt = load_embodiment(other)
    assert not rt.config.ik_weights.collision_enabled
    solver = rt.solver_cls(config=rt.config.ik_weights)
    assert solver.robot_collision is None
    assert solver._collision_gate is None


def test_collision_model_clean_at_home(runtime) -> None:
    """Structurally-overlapping capsule pairs are auto-ignored at build time."""
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    if solver.robot_collision is None:
        pytest.skip("collision not enabled for this embodiment")
    distances = np.asarray(
        solver.robot_collision.compute_self_collision_distance(
            runtime.robot, runtime.home_q()
        )
    )
    margin = runtime.config.ik_weights.self_collision_margin
    assert float(distances.min()) >= margin - 1e-6


def test_self_collision_cost_prevents_interpenetration(runtime) -> None:
    """Commanding both TCPs to one point must stop at contact, not merge."""
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    if solver.robot_collision is None:
        pytest.skip("collision not enabled for this embodiment")
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    center = ((left[:3] + right[:3]) / 2).astype(np.float32)
    q = home
    for _ in range(20):
        q = np.asarray(
            solver.ik(
                q,
                left_pose=(center, left[3:7]),
                right_pose=(center, right[3:7]),
            ),
            dtype=np.float32,
        )
    separation = float(
        np.asarray(
            solver.robot_collision.compute_self_collision_distance(runtime.robot, q)
        ).min()
    )
    assert separation > -0.005  # capsules may touch but must not interpenetrate


def test_world_collision_cost_keeps_tcp_above_table(runtime) -> None:
    """Commanding TCPs below z=0 must stop at the tabletop halfspace."""
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    if solver.robot_collision is None:
        pytest.skip("collision not enabled for this embodiment")
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    below_left = (np.array([left[0], left[1], -0.05], dtype=np.float32), left[3:7])
    below_right = (np.array([right[0], right[1], -0.05], dtype=np.float32), right[3:7])
    q = home
    for _ in range(20):
        q = np.asarray(
            solver.ik(q, left_pose=below_left, right_pose=below_right),
            dtype=np.float32,
        )
    l_sol, r_sol = solver.fk_pose7(q)
    assert l_sol[2] > -0.005
    assert r_sol[2] > -0.005


@pytest.mark.parametrize("name", ("piper", "metal"))
def test_replay_ik_weights_never_leak_into_the_teleop_profile(name) -> None:
    """Real backends must keep exactly the weights their own YAML declares."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    declared = data["ik_weights"]
    rt = load_embodiment(name)
    assert rt.config.ik_weights.ori_weight == pytest.approx(declared["ori"])
    assert rt.config.ik_weights.rest_weight == pytest.approx(declared["rest"])
    overrides = (data.get("replay") or {}).get("ik_weights") or {}
    assert rt.config.replay_ik_weights == {
        key: float(value) for key, value in overrides.items()
    }


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


def test_robot_config_metadata_identifies_the_profile_it_names() -> None:
    piper = robot_config_metadata("piper")
    openarm = robot_config_metadata("openarmv1")
    assert piper["name"] == "piper" and piper["configuration"]["kind"] == "piper"
    assert openarm["name"] == "openarmv1" and openarm["sha256"] != piper["sha256"]
    assert piper["config_path"].endswith("piper.yaml")
    with pytest.raises(SystemExit, match="Unknown robot"):
        robot_config_metadata("not-a-robot")
