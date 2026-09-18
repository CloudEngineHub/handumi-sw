"""Single-arm Panda parity against Menagerie and end-to-end retargeting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from handumi.dataset.canonical import (
    canonical_joint_layout,
    canonicalize_joint_trajectory,
    expand_canonical_trajectory,
)
from handumi.robots.registry import load_embodiment


@pytest.fixture(scope="module")
def panda():
    return load_embodiment("franka_emika_panda")


def test_topology_and_canonical_roundtrip(panda):
    assert panda.active_sides == ("right",)
    assert panda.arm_joint_indices("left") == []
    assert panda.arm_joint_indices("right") == list(range(8))
    assert panda.joint_names == tuple(f"right_joint{i}" for i in range(1, 8)) + (
        "right_finger_joint1",
    )
    q = panda.home_q()
    assert np.all(q >= np.asarray(panda.robot.joints.lower_limits))
    assert np.all(q <= np.asarray(panda.robot.joints.upper_limits))
    for opening in (0.0, 0.5, 1.0):
        panda.set_finger_positions(q, {"left": 1.0, "right": opening})
        assert q[-1] == pytest.approx(0.04 * opening)
        canonical = canonicalize_joint_trajectory(q[None], runtime=panda)
        assert canonical.shape == (1, 8)
        assert canonical[0, -1] == pytest.approx(0.08 * opening)
        expanded, openings = expand_canonical_trajectory(canonical, runtime=panda)
        np.testing.assert_allclose(expanded, q[None], atol=1e-6)
        assert openings[0, 1] == pytest.approx(opening)
    assert canonical_joint_layout(panda).names[-1] == "right_gripper.width_m"
    assert panda.config.real.backend is None


def test_meshes_and_mimic(panda):
    urdf = panda.load_urdf(load_meshes=True)
    assert len(urdf.scene.geometry) > 40
    for geometry in urdf.scene.geometry.values():
        assert len(geometry.vertices) > 0
    assert urdf.joint_map["right_finger_joint2"].mimic.joint == "right_finger_joint1"


def test_fk_matches_upstream_menagerie(panda):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(panda.urdf_path.parent / "panda.xml"))
    data = mujoco.MjData(model)
    solver = panda.solver_cls()
    for offsets in (
        np.zeros(7),
        np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2, 0.1]),
        np.array([-0.3, 0.25, -0.15, 0.2, -0.2, -0.3, 0.2]),
    ):
        q = panda.home_q()
        q[:7] += offsets
        data.qpos[:] = np.r_[q, q[-1]]
        mujoco.mj_forward(model, data)
        hand = data.body("hand")
        expected_pos = hand.xpos + hand.xmat.reshape(3, 3) @ [0, 0, 0.1029]
        _, pose = solver.fk_pose7(q)
        np.testing.assert_allclose(pose[:3], expected_pos, atol=2e-6)
        wxyz = pose[[6, 3, 4, 5]]
        assert abs(np.dot(wxyz, hand.xquat)) == pytest.approx(1.0, abs=2e-6)


def test_ik_ignores_left_targets(panda):
    solver = panda.solver_cls(
        config=replace(panda.config.ik_weights, max_joint_delta=None)
    )
    target_q = panda.home_q()
    target_q[0] += 0.08
    _, target = solver.fk_pose7(target_q)
    home = panda.home_q()
    desired = (target[:3], target[3:])
    only_right = solver.ik(home, right_pose=desired)
    both = solver.ik(
        home,
        left_pose=(np.array([100, -100, 100]), np.array([0, 0, 0, 1])),
        right_pose=desired,
    )
    np.testing.assert_allclose(both, only_right, atol=1e-6)
    _, achieved = solver.fk_pose7(only_right)
    assert np.linalg.norm(achieved[:3] - target[:3]) < 0.001


@pytest.mark.parametrize("mode", ["absolute-table", "local-relative", "anchored"])
def test_replay_right_only_and_canonical_conversion(panda, monkeypatch, mode):
    from handumi.scripts.replay import replay_in_sim as replay

    solver = panda.solver_cls()
    states = np.zeros((12, 16), dtype=np.float32)
    states[:, 6] = 1
    # An arbitrary left controller must never influence the Panda trajectory
    # or make strict replay fail.
    states[:, :3] = [20, -30, 40]
    for i in range(len(states)):
        q = panda.home_q()
        q[0] += i * 0.002
        _, states[i, 7:14] = solver.fk_pose7(q)
    states[:, 15] = np.linspace(0.01, 0.07, len(states))
    info = {
        "handumi": {
            "active_sides": ["right"],
            "recording_device": "pico",
            "tracking_workspace": "table",
        }
    }
    monkeypatch.setattr(
        replay, "load_episode_states", lambda args: (states, 30, info, None)
    )
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "franka_emika_panda",
            "--episode",
            "0",
            "--raw-controller-debug",
            "--retarget-mode",
            mode,
            "--deployment-profile",
            "sim",
            "--strict-ik",
        ]
    )
    # CLI main resolves DATASET; the source itself is replaced above.
    args.repo_id = "local/test"
    args.root = Path("unused")
    rollout = replay.solve_episode(args)
    assert rollout["qpos"].shape == (12, 8)
    assert np.max(rollout["right_pos_error_m"]) < 0.01
    assert np.max(rollout["left_pos_error_m"]) == 0
    canonical = canonicalize_joint_trajectory(rollout["qpos"], runtime=panda)
    np.testing.assert_allclose(canonical[:, -1], states[:, 15], atol=1e-6)


def test_physics_actuator_mapping_and_aperture(panda):
    mujoco = pytest.importorskip("mujoco")
    from handumi.sim.mujoco_sim import MujocoPhysics

    names = [panda.mjcf_actuator_name(name) for name in panda.joint_names]
    physics = MujocoPhysics(mjcf_path=panda.config.mjcf, actuator_names=names)
    visual = physics.model.geom_group == 2
    assert np.all(physics.model.geom_contype[visual] == 0)
    assert np.all(physics.model.geom_conaffinity[visual] == 0)
    mujoco.mj_resetDataKeyframe(physics.model, physics.data, 0)
    q = panda.home_q()
    q[-1] = 0.02
    physics.set_ctrl(dict(zip(names, q, strict=True)))
    for _ in range(300):
        mujoco.mj_step(physics.model, physics.data)
    assert physics.joint_positions()["finger_joint1"] == pytest.approx(0.02, abs=0.002)
    assert physics.data.qpos[-1] == pytest.approx(0.02, abs=0.002)
    assert np.all(np.isfinite(physics.data.qpos))
