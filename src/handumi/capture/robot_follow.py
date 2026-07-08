"""Live robot follow-along: 16D raw state -> bimanual IK -> Viser (Phase 2B).

Consumes the same 16D HandUMI raw state the recorders emit, so any tracking
source that produces it (Quest today, PICO later) can drive the robot view.
Runs alongside Rerun: Rerun keeps cameras/series/controller trails, Viser
renders the URDF arms following your hands (http://localhost:<port>).

Teleop mapping is ABSOLUTE: one fixed workspace -> robot-world transform
(``configs/teleop.yaml``, calibrated once with handumi-calibrate-workspace)
maps every tracked TCP pose to the same robot-world point, every session.
There is deliberately no per-session or per-hand re-anchoring — that would
break the correspondence between a real task scene and the simulated one
(``--scene``), which is the whole point: a pick & place done with the real
HandUMI replays 1:1 on the simulated arms when the cube/box sit at the same
coordinates in both worlds.

Frame mapping (verified against Piper FK): the dual-Piper URDF world and
``handumi_workspace`` share the same right-handed X-forward / Y-left / Z-up
convention, so positions need the calibrated translation (+ optional yaw).
Orientations get one constant alignment: the gripper-TCP identity
(X-forward) maps to the Piper EE rest orientation (EE Z forward, X down).

The heavy IK stack (JAX/pyroki, ~30s JIT warmup) is imported lazily inside
:class:`RobotFollower`; importing this module stays cheap so the pure
transform below is unit-testable without JAX installed.
"""

from __future__ import annotations

import asyncio
import logging
import webbrowser
from pathlib import Path

import numpy as np
import yaml

from handumi.retargeting.handumi_to_robot import raw_state_target_poses

log = logging.getLogger("handumi.robot_follow")

DEFAULT_SCENE_CONFIG_PATH = Path("configs/scene.yaml")
DEFAULT_TELEOP_CONFIG_PATH = Path("configs/teleop.yaml")

# Identity controller orientation (gripper X-forward, workspace frame) -> Piper
# EE rest orientation (EE Z axis forward, X axis down). Columns are the EE
# frame axes expressed in the world frame; right-handed (det = +1).
WRIST_ALIGN = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# HandUMI gripper full opening (m) used to normalize widths into [0, 1].
DEFAULT_GRIPPER_MAX_WIDTH_M = 0.08


def load_workspace_to_robot(
    path: str | Path = DEFAULT_TELEOP_CONFIG_PATH,
) -> tuple[np.ndarray, float]:
    """Load the fixed workspace->robot transform: (translation, yaw_deg).

    Missing file or keys fall back to the uncalibrated default (see
    configs/teleop.yaml).
    """
    translation = np.array([0.0, 0.0, 0.55], dtype=np.float32)
    yaw_deg = 0.0
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        section = data.get("workspace_to_robot") or {}
        translation = np.asarray(
            section.get("translation", translation), dtype=np.float32
        )
        yaw_deg = float(section.get("yaw_deg", yaw_deg))
    return translation, yaw_deg


def _yaw_matrix(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def raw_state_to_robot_targets(
    state: np.ndarray,
    *,
    translation: np.ndarray,
    yaw_deg: float = 0.0,
    gripper_max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M,
) -> dict:
    """Map one 16D raw state from ``handumi_workspace`` into robot-world targets.

    Applies the fixed workspace->robot transform (yaw about Z, then
    translation) to both TCP positions, composes orientations with the yaw
    and :data:`WRIST_ALIGN`, and normalizes gripper widths to [0, 1].
    Returns ``{"left"/"right": (pos, rot_3x3), "left_grip"/"right_grip": float}``.
    Pure numpy — no solver, no JAX.
    """
    (left_pos, left_rot), (right_pos, right_rot) = raw_state_target_poses(state)
    yaw_rot = _yaw_matrix(yaw_deg)
    offset = np.asarray(translation, dtype=np.float32)
    arr = np.asarray(state, dtype=np.float32)
    max_w = max(gripper_max_width_m, 1e-6)
    return {
        "left": (yaw_rot @ left_pos + offset, yaw_rot @ left_rot @ WRIST_ALIGN),
        "right": (yaw_rot @ right_pos + offset, yaw_rot @ right_rot @ WRIST_ALIGN),
        "left_grip": float(np.clip(arr[14] / max_w, 0.0, 1.0)),
        "right_grip": float(np.clip(arr[15] / max_w, 0.0, 1.0)),
    }


class RobotFollower:
    """Owns the IK solver + Viser sim and advances them one raw state at a time.

    Construction is slow (URDF load + JAX JIT warmup, ~30s on CPU); each
    subsequent :meth:`step` solves in ~10-20ms, well inside a 30 Hz budget.
    """

    def __init__(
        self,
        *,
        embodiment: str = "piper",
        port: int | None = None,
        gripper_max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M,
        open_browser: bool = True,
        scene_config_path: Path = DEFAULT_SCENE_CONFIG_PATH,
        scene_name: str | None = None,
        teleop_config_path: Path = DEFAULT_TELEOP_CONFIG_PATH,
    ) -> None:
        import dataclasses

        from handumi.sim.mujoco_sim import SceneConfig
        from handumi.robots.registry import load_embodiment

        self._gripper_max_width_m = gripper_max_width_m
        self._translation, self._yaw_deg = load_workspace_to_robot(teleop_config_path)
        log.info(
            "workspace->robot transform: translation=%s yaw=%.1f deg (%s)",
            np.round(self._translation, 4).tolist(), self._yaw_deg, teleop_config_path,
        )

        scene_config = SceneConfig.from_yaml(scene_config_path)
        if scene_name is not None:
            # --scene overrides configs/scene.yaml (which defaults to no
            # scene) so task props only appear when explicitly requested.
            scene_config = dataclasses.replace(scene_config, name=scene_name)

        log.info("Loading %s IK solver (JAX JIT warmup, ~30s on CPU)...", embodiment)
        runtime = load_embodiment(embodiment)
        self._solver = runtime.solver_cls()
        self._command_size = runtime.command_size
        self._tcp_offset_ee = np.asarray(runtime.tcp_offset_ee, dtype=np.float32)
        self._gripper_index = runtime.command_size - 1
        self._q = np.zeros(self._solver.num_joints, dtype=np.float32)

        self._aio = asyncio.new_event_loop()

        # Real rigid-body physics (contact/grasp) is only available for
        # embodiments that ship an MJCF (runtime.mjcf_path) — make_physics()
        # returns None otherwise (e.g. axol, which only has a URDF today),
        # and Viser then stays kinematics-only, same as before physics existed.
        self._physics = runtime.make_physics(scene_config=scene_config)
        if self._physics is not None:
            self._aio.run_until_complete(self._physics.enable())

        scene_bodies = self._physics.scene_bodies if self._physics is not None else None
        self._sim = runtime.make_sim(port=port, scene_bodies=scene_bodies)
        self._aio.run_until_complete(self._sim.enable())
        resolved_port = port if port is not None else runtime.default_port
        url = f"http://localhost:{resolved_port}"
        log.info("Robot view ready: %s", url)
        if open_browser:
            webbrowser.open(url)

    def reset(self) -> None:
        """Return the arms to the rest pose (used on workspace reset).

        The workspace->robot transform is fixed, so nothing is re-anchored:
        tracking resumes wherever the mapped hand poses actually are.
        """
        self._q = np.zeros(self._solver.num_joints, dtype=np.float32)
        rest = np.zeros(self._command_size, dtype=np.float32)
        if self._physics is not None:
            self._aio.run_until_complete(self._physics.reset())
            self._push_physics_state_to_viser()
        else:
            self._aio.run_until_complete(self._sim.motion_control(left=rest, right=rest))

    def step(
        self,
        state: np.ndarray,
        *,
        left_tracked: bool,
        right_tracked: bool,
    ) -> None:
        """Solve IK toward the tracked side(s) and push the pose to Viser.

        Untracked sides get no pose target, so the rest cost holds them at the
        current joint angles instead of chasing a frozen/stale pose.
        """
        targets = raw_state_to_robot_targets(
            state,
            translation=self._translation,
            yaw_deg=self._yaw_deg,
            gripper_max_width_m=self._gripper_max_width_m,
        )
        self._q = self._solver.ik(
            self._q,
            left_pose=targets["left"] if left_tracked else None,
            right_pose=targets["right"] if right_tracked else None,
        )

        left_cmd = np.zeros(self._command_size, dtype=np.float32)
        right_cmd = np.zeros(self._command_size, dtype=np.float32)
        left_cmd[: len(self._solver.left_indices)] = self._q[self._solver.left_indices]
        right_cmd[: len(self._solver.right_indices)] = self._q[self._solver.right_indices]
        left_cmd[self._gripper_index] = targets["left_grip"]
        right_cmd[self._gripper_index] = targets["right_grip"]

        if self._physics is not None:
            # IK targets become actuator setpoints; MuJoCo's own background
            # thread steps contact physics toward them independently of this
            # call. Viser then renders whatever MuJoCo actually settled on
            # (which may lag/differ from the IK command under contact), not
            # the raw IK solution.
            self._aio.run_until_complete(
                self._physics.motion_control(left=left_cmd, right=right_cmd)
            )
            self._push_physics_state_to_viser()
        else:
            self._aio.run_until_complete(
                self._sim.motion_control(left=left_cmd, right=right_cmd)
            )

        # TCP marker + trail (see ViserSim.set_tcp_pose): always the IK-solved
        # FK, not MuJoCo's physically-settled pose, since the point is to
        # visually sanity-check calibration/IK against the tracked hand
        # motion, independent of contact dynamics. FK stops at the wrist
        # flange, so push the marker out to the gripper tip with the
        # embodiment's fixed EE-frame offset (registry tcp_offset_ee).
        left_fk, right_fk = self._solver.fk(self._q)
        for side, fk in (("left", left_fk), ("right", right_fk)):
            position = np.asarray(fk.translation(), dtype=np.float32)
            rotation = np.asarray(fk.rotation().as_matrix(), dtype=np.float32)
            tip = position + rotation @ self._tcp_offset_ee
            self._aio.run_until_complete(self._sim.set_tcp_pose(side, tip))

    def _push_physics_state_to_viser(self) -> None:
        """Read the current MuJoCo state (arms + every dynamic scene body)
        and render it in Viser. Generic over the scene: whatever task asset
        was loaded (see configs/scene.yaml), each of its dynamic bodies gets
        forwarded by name — no per-task code here."""
        arm_qpos = self._aio.run_until_complete(self._physics.get_arm_qpos())
        self._aio.run_until_complete(self._sim.set_actual_joint_positions(arm_qpos))
        for scene_body in self._physics.scene_bodies:
            if not scene_body.dynamic:
                continue
            pose = self._aio.run_until_complete(self._physics.get_body_pose(scene_body.name))
            if pose is not None:
                position, quaternion_wxyz = pose
                self._aio.run_until_complete(
                    self._sim.set_body_pose(scene_body.name, position, quaternion_wxyz)
                )

    def close(self) -> None:
        if self._physics is not None:
            self._physics.close()
        self._aio.close()
