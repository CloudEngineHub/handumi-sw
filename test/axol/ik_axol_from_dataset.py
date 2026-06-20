#!/usr/bin/env python3
"""
Replay Pico controller data from a LeRobot dataset as Axol robot IK trajectories.

Strategy: SE3 delta retargeting.
  - Frame 0 of each controller is the "engage" origin.
  - Per frame: delta_pico = origin_inv @ now  (movement since frame 0, in Pico frame)
  - delta is converted to robot FLU frame, then applied to the robot's initial EE pose.
  - IK is solved for every frame and the result is played back in MuJoCo.

Pico uses OpenXR convention: X=right, Y=up, Z=backward (right-handed, Y-up).
Robot uses FLU convention: X=forward, Y=left, Z=up.

Usage:
    uv run python test/axol/ik_axol_from_dataset.py datasets/my_dataset
    uv run python test/axol/ik_axol_from_dataset.py datasets/my_dataset --episode 0
    uv run python test/axol/ik_axol_from_dataset.py datasets/my_dataset --speed 0.5
    uv run python test/axol/ik_axol_from_dataset.py datasets/my_dataset --plot-only
    uv run python test/axol/ik_axol_from_dataset.py datasets/my_dataset --plot --plot-save ik_joints.png
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dexumi.robots.axol.config import KinematicsConfig
from dexumi.robots.axol.solver import KinematicsSolver

_URDF_PATH = _REPO_ROOT / "assets" / "axol" / "urdf" / "axol.urdf"

# Rotation from Pico (OpenXR: X=right, Y=up, Z=backward)
# to robot FLU (X=forward, Y=left, Z=up):
#   FLU-X = -Pico-Z   (forward = negative backward)
#   FLU-Y = -Pico-X   (left    = negative right)
#   FLU-Z =  Pico-Y   (up      = up)
_R_PICO_TO_FLU = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


# ── Dataset helpers ────────────────────────────────────────────────────────────


def load_episode(
    root: Path, episode: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, left_poses, right_poses) for the selected episode.

    Each pose array has shape (N, 7): [x, y, z, qx, qy, qz, qw].
    """
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"[ERROR] No .parquet files found under {root / 'data'}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    df.sort_values("index", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if episode is None:
        episode = int(df["episode_index"].iloc[0])
    df = df[df["episode_index"] == episode]
    if df.empty:
        sys.exit(f"[ERROR] Episode {episode} not found in dataset.")

    left = np.stack(df["observation.pico.left_controller_pose"].values).astype(
        np.float32
    )
    right = np.stack(df["observation.pico.right_controller_pose"].values).astype(
        np.float32
    )
    ts = df["timestamp"].values.astype(float).ravel()
    print(f"Episode {episode}: {len(ts)} frames")
    return ts, left, right


# ── Retargeting ────────────────────────────────────────────────────────────────


def _pico_pose_in_flu(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Pico pose [x,y,z,qx,qy,qz,qw] → (pos, rot_3x3) in FLU frame."""
    pos = (_R_PICO_TO_FLU @ pose7[:3]).astype(np.float32)
    R_pico = Rotation.from_quat(pose7[3:]).as_matrix().astype(np.float32)
    R_flu = (_R_PICO_TO_FLU @ R_pico @ _R_PICO_TO_FLU.T).astype(np.float32)
    return pos, R_flu


def _format_vec(v: np.ndarray) -> str:
    return np.array2string(v, precision=4, suppress_small=True)


# Joints whose q=0 sits at or near a hard limit boundary in the Axol URDF.
#
#   left_e1_0:  [0.000,  2.618]  → q=0 is AT the lower bound
#   right_e1_0: [-2.618, 0.000]  → q=0 is AT the upper bound
#   left_s2_0:  [-3.142, 0.349]  → q=0 is only 0.35 rad from the upper bound
#   right_s2_0: [-0.349, 3.142]  → q=0 is only 0.35 rad from the lower bound
#
# With limit_weight=75 (the highest cost weight), the optimizer at q=0 sees a
# limit barrier that overwhelms the position cost (pos_weight=50) on the elbow
# joints, which are the most critical for arm reach.  The solver stays at 0.
_HOME_POSE_OFFSETS: dict[str, float] = {
    "left_e1_0": math.pi / 3,    # 60° into [0, 2.618]
    "right_e1_0": -math.pi / 3,  # 60° into [-2.618, 0]
    "left_s2_0": -math.pi / 6,   # pull away from upper limit 0.349 rad
    "right_s2_0": math.pi / 6,   # pull away from lower limit -0.349 rad
}


def _build_home_pose(joint_names: list[str]) -> np.ndarray:
    """Return a starting pose with limit-boundary joints moved to safe values.

    Uses :data:`_HOME_POSE_OFFSETS` to move joints that would otherwise sit at
    their hard limit boundary when q=0.  All other joints remain at 0.
    """
    q = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        if name in _HOME_POSE_OFFSETS:
            q[i] = _HOME_POSE_OFFSETS[name]
    return q


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle (degrees) of a 3×3 rotation matrix."""
    trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(trace)))


def print_dataset_pose_stats(
    left_poses: np.ndarray, right_poses: np.ndarray
) -> None:
    """Print raw Pico pose diagnostics from the loaded episode."""
    print("\n── Dataset pose diagnostics ──")
    for label, poses in (("Left", left_poses), ("Right", right_poses)):
        pos = poses[:, :3]
        pos_delta = pos - pos[0]
        pos_norm = np.linalg.norm(pos_delta, axis=1)
        quat = poses[:, 3:]
        quat_norm = np.linalg.norm(quat, axis=1)

        print(f"  [{label}] shape: {poses.shape}")
        print(f"  [{label}] frame 0 pose: {_format_vec(poses[0])}")
        if len(poses) > 1:
            print(f"  [{label}] frame 1 pose: {_format_vec(poses[1])}")
        print(f"  [{label}] position range (m):")
        print(f"    min: {_format_vec(pos.min(axis=0))}")
        print(f"    max: {_format_vec(pos.max(axis=0))}")
        print(f"  [{label}] max |pos - pos[0]|: {pos_norm.max():.6f} m")
        print(f"  [{label}] mean |pos - pos[0]|: {pos_norm.mean():.6f} m")
        print(f"  [{label}] frames with pos delta > 1 mm: {(pos_norm > 0.001).sum()}")
        print(f"  [{label}] quaternion norm — min: {quat_norm.min():.6f}, max: {quat_norm.max():.6f}")
        if np.any(quat_norm < 0.5):
            print(f"  [{label}] [WARN] Some quaternions look unnormalized or zero.")


def print_retargeting_origin_stats(
    left_poses: np.ndarray,
    right_poses: np.ndarray,
    solver: KinematicsSolver,
) -> None:
    """Print FK rest pose and Pico origin used for delta retargeting."""
    q_zero = np.zeros(solver.num_joints, dtype=np.float32)
    ee_L0, ee_R0 = solver.fk(q_zero)
    robot_L0_pos = np.asarray(ee_L0.translation(), dtype=np.float32)
    robot_R0_pos = np.asarray(ee_R0.translation(), dtype=np.float32)

    pico_L0_pos, pico_L0_rot = _pico_pose_in_flu(left_poses[0])
    pico_R0_pos, pico_R0_rot = _pico_pose_in_flu(right_poses[0])

    print("\n── Retargeting origins ──")
    print(f"  FK q=0 left EE pos (FLU):  {_format_vec(robot_L0_pos)}")
    print(f"  FK q=0 right EE pos (FLU): {_format_vec(robot_R0_pos)}")
    print(f"  Pico origin left  (FLU):   {_format_vec(pico_L0_pos)}")
    print(f"  Pico origin right (FLU):   {_format_vec(pico_R0_pos)}")


def print_retargeting_trajectory_stats(
    left_poses: np.ndarray,
    right_poses: np.ndarray,
    solver: KinematicsSolver,
) -> None:
    """Summarise Pico deltas and IK targets across the episode."""
    q_zero = np.zeros(solver.num_joints, dtype=np.float32)
    ee_L0, ee_R0 = solver.fk(q_zero)
    robot_L0_pos = np.asarray(ee_L0.translation(), dtype=np.float32)
    robot_L0_rot = np.asarray(ee_L0.rotation().as_matrix(), dtype=np.float32)
    robot_R0_pos = np.asarray(ee_R0.translation(), dtype=np.float32)
    robot_R0_rot = np.asarray(ee_R0.rotation().as_matrix(), dtype=np.float32)

    pico_L0_pos, pico_L0_rot = _pico_pose_in_flu(left_poses[0])
    pico_R0_pos, pico_R0_rot = _pico_pose_in_flu(right_poses[0])

    delta_L_norms: list[float] = []
    delta_R_norms: list[float] = []
    delta_L_rot_degs: list[float] = []
    delta_R_rot_degs: list[float] = []
    target_L_shifts: list[float] = []
    target_R_shifts: list[float] = []

    for i in range(len(left_poses)):
        pico_L_pos, pico_L_rot = _pico_pose_in_flu(left_poses[i])
        pico_R_pos, pico_R_rot = _pico_pose_in_flu(right_poses[i])

        delta_L_pos = pico_L0_rot.T @ (pico_L_pos - pico_L0_pos)
        delta_L_rot = pico_L0_rot.T @ pico_L_rot
        delta_R_pos = pico_R0_rot.T @ (pico_R_pos - pico_R0_pos)
        delta_R_rot = pico_R0_rot.T @ pico_R_rot

        target_L_pos = robot_L0_pos + robot_L0_rot @ delta_L_pos
        target_R_pos = robot_R0_pos + robot_R0_rot @ delta_R_pos

        delta_L_norms.append(float(np.linalg.norm(delta_L_pos)))
        delta_R_norms.append(float(np.linalg.norm(delta_R_pos)))
        delta_L_rot_degs.append(_rotation_angle_deg(delta_L_rot))
        delta_R_rot_degs.append(_rotation_angle_deg(delta_R_rot))
        target_L_shifts.append(float(np.linalg.norm(target_L_pos - robot_L0_pos)))
        target_R_shifts.append(float(np.linalg.norm(target_R_pos - robot_R0_pos)))

    delta_L_arr = np.asarray(delta_L_norms)
    delta_R_arr = np.asarray(delta_R_norms)
    target_L_arr = np.asarray(target_L_shifts)
    target_R_arr = np.asarray(target_R_shifts)

    print("\n── Retargeting trajectory stats ──")
    for label, delta_arr, rot_degs, target_arr in (
        ("Left", delta_L_arr, delta_L_rot_degs, target_L_arr),
        ("Right", delta_R_arr, delta_R_rot_degs, target_R_arr),
    ):
        print(f"  [{label}] Pico delta pos — max: {delta_arr.max():.6f} m, mean: {delta_arr.mean():.6f} m")
        print(f"  [{label}] Pico delta rot — max: {max(rot_degs):.3f} deg, mean: {np.mean(rot_degs):.3f} deg")
        print(f"  [{label}] IK target shift from FK(0) — max: {target_arr.max():.6f} m, mean: {target_arr.mean():.6f} m")
        print(f"  [{label}] frames with target shift > 1 mm: {(target_arr > 0.001).sum()}")


def print_ik_joint_trajectory_stats(
    q_traj: np.ndarray, joint_names: list[str]
) -> None:
    """Summarise solved joint trajectories."""
    q_deg = np.rad2deg(q_traj)
    q_delta = np.diff(q_traj, axis=0) if len(q_traj) > 1 else np.zeros((0, q_traj.shape[1]))
    per_joint_range = q_deg.max(axis=0) - q_deg.min(axis=0)
    per_joint_std = q_deg.std(axis=0)
    per_frame_norm = np.linalg.norm(q_traj, axis=1)
    per_step_norm = np.linalg.norm(q_delta, axis=1) if len(q_delta) else np.zeros(0)

    all_zero_frames = int(np.sum(np.all(np.isclose(q_traj, 0.0), axis=1)))
    any_motion_frames = int(np.sum(per_frame_norm > 1e-4))
    nonzero_joints = int(np.sum(per_joint_range > 0.01))

    print("\n── IK joint trajectory stats ──")
    print(f"  frames: {len(q_traj)}, joints: {q_traj.shape[1]}")
    print(f"  frames with all joints ≈ 0: {all_zero_frames} / {len(q_traj)}")
    print(f"  frames with ||q|| > 1e-4 rad: {any_motion_frames} / {len(q_traj)}")
    print(f"  joints with range > 0.01 deg: {nonzero_joints} / {len(joint_names)}")
    print(f"  ||q|| per frame — max: {per_frame_norm.max():.6f} rad, mean: {per_frame_norm.mean():.6f} rad")
    if len(per_step_norm):
        print(f"  ||Δq|| per step — max: {per_step_norm.max():.6f} rad, mean: {per_step_norm.mean():.6f} rad")
    print(f"  joint angle range (deg) — max: {per_joint_range.max():.3f}, mean: {per_joint_range.mean():.3f}")
    print(f"  joint angle std (deg)   — max: {per_joint_std.max():.3f}, mean: {per_joint_std.mean():.3f}")

    print("  per-joint summary (deg):")
    for name, qj_deg in zip(joint_names, q_deg.T):
        short = name.removeprefix("left_").removeprefix("right_")
        side = "L" if name.startswith("left_") else "R"
        print(
            f"    {side}/{short:8s}  "
            f"min={qj_deg.min():7.3f}  max={qj_deg.max():7.3f}  "
            f"range={qj_deg.max()-qj_deg.min():7.3f}  std={qj_deg.std():6.3f}"
        )


def compute_ik_trajectories(
    left_poses: np.ndarray,
    right_poses: np.ndarray,
    solver: KinematicsSolver,
) -> np.ndarray:
    """Compute joint trajectory via IK for every frame.

    Returns array of shape (N, num_joints) in radians.
    """
    N = len(left_poses)
    q_traj = np.zeros((N, solver.num_joints), dtype=np.float32)

    # Use a home pose that keeps joints away from their hard limit boundaries.
    # q=0 places left_e1_0/right_e1_0 exactly AT their limits, which traps the
    # optimizer (limit_weight=75 > pos_weight=50).
    q_current = _build_home_pose(solver.joint_names)
    solver.set_posture_pose(q_current)

    # Robot EE poses at the home pose (used as the retargeting reference)
    ee_L0, ee_R0 = solver.fk(q_current)
    robot_L0_pos = np.asarray(ee_L0.translation(), dtype=np.float32)
    robot_L0_rot = np.asarray(ee_L0.rotation().as_matrix(), dtype=np.float32)
    robot_R0_pos = np.asarray(ee_R0.translation(), dtype=np.float32)
    robot_R0_rot = np.asarray(ee_R0.rotation().as_matrix(), dtype=np.float32)

    # Pico origins (frame 0) expressed in FLU frame
    pico_L0_pos, pico_L0_rot = _pico_pose_in_flu(left_poses[0])
    pico_R0_pos, pico_R0_rot = _pico_pose_in_flu(right_poses[0])

    print("\n── IK loop diagnostics ──")
    print(f"  q_init (home pose): {_format_vec(q_current)}")
    home_offsets = {n: v for n, v in _HOME_POSE_OFFSETS.items() if n in solver.joint_names}
    print(f"  joints moved from boundary: {list(home_offsets.keys())}")
    print(f"  FK home left EE pos:  {_format_vec(robot_L0_pos)}")
    print(f"  FK home right EE pos: {_format_vec(robot_R0_pos)}")
    print(f"  Pico origin left:    {_format_vec(pico_L0_pos)}")
    print(f"  Pico origin right:   {_format_vec(pico_R0_pos)}")

    print(f"\nRunning IK on {N} frames...")
    for i in range(N):
        if i % 100 == 0:
            print(f"  frame {i:4d}/{N}")

        # Current pico poses in FLU
        pico_L_pos, pico_L_rot = _pico_pose_in_flu(left_poses[i])
        pico_R_pos, pico_R_rot = _pico_pose_in_flu(right_poses[i])

        # Delta from origin: delta = R0.T @ (now - origin_pos), R0.T @ R_now
        delta_L_pos = pico_L0_rot.T @ (pico_L_pos - pico_L0_pos)
        delta_L_rot = pico_L0_rot.T @ pico_L_rot
        delta_R_pos = pico_R0_rot.T @ (pico_R_pos - pico_R0_pos)
        delta_R_rot = pico_R0_rot.T @ pico_R_rot

        # Apply delta to robot EE origin
        target_L_pos = robot_L0_pos + robot_L0_rot @ delta_L_pos
        target_L_rot = robot_L0_rot @ delta_L_rot
        target_R_pos = robot_R0_pos + robot_R0_rot @ delta_R_pos
        target_R_rot = robot_R0_rot @ delta_R_rot

        q_prev = q_current.copy()
        q_current = solver.ik(
            q_current,
            left_pose=(target_L_pos, target_L_rot),
            right_pose=(target_R_pos, target_R_rot),
        )
        q_traj[i] = q_current

        if i <= 3:
            print(f"\n  [frame {i}] retarget debug:")
            print(f"    delta_L_pos:     {_format_vec(delta_L_pos)}  (|Δ|={np.linalg.norm(delta_L_pos):.6f} m)")
            print(f"    delta_R_pos:     {_format_vec(delta_R_pos)}  (|Δ|={np.linalg.norm(delta_R_pos):.6f} m)")
            print(f"    delta_L_rot deg: {_rotation_angle_deg(delta_L_rot):.3f}")
            print(f"    delta_R_rot deg: {_rotation_angle_deg(delta_R_rot):.3f}")
            print(f"    target_L_pos:    {_format_vec(target_L_pos)}")
            print(f"    target_R_pos:    {_format_vec(target_R_pos)}")
            print(f"    target_L shift:  {np.linalg.norm(target_L_pos - robot_L0_pos):.6f} m")
            print(f"    target_R shift:  {np.linalg.norm(target_R_pos - robot_R0_pos):.6f} m")
            print(f"    q_prev:          {_format_vec(q_prev)}")
            print(f"    q_current:       {_format_vec(q_current)}")
            print(f"    Δq:              {_format_vec(q_current - q_prev)}  (|Δq|={np.linalg.norm(q_current - q_prev):.6f} rad)")

    print("\nIK complete.")
    return q_traj


# ── Matplotlib visualization ──────────────────────────────────────────────────


def plot_joint_trajectories(
    q_traj: np.ndarray,
    joint_names: list[str],
    ts: np.ndarray | None = None,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot IK joint angles over time (degrees), left arm on top, right on bottom."""
    n_joints = q_traj.shape[1]
    if n_joints != len(joint_names):
        raise ValueError(
            f"q_traj has {n_joints} joints but got {len(joint_names)} names"
        )

    if ts is not None and len(ts) == len(q_traj):
        t = (ts - ts[0]).astype(float)
        x_label = "Time (s)"
    else:
        t = np.arange(len(q_traj), dtype=float)
        x_label = "Frame"

    q_deg = np.rad2deg(q_traj)
    mid = n_joints // 2
    arms = (
        ("Left arm", joint_names[:mid], q_deg[:, :mid]),
        ("Right arm", joint_names[mid:], q_deg[:, mid:]),
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IK joint trajectories", fontsize=14)

    for ax, (title, names, angles) in zip(axes, arms):
        for j, name in enumerate(names):
            short = name.removeprefix("left_").removeprefix("right_")
            ax.plot(t, angles[:, j], label=short, linewidth=1.2)
        ax.set_ylabel("Angle (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=2, fontsize=8)

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── MuJoCo replay ─────────────────────────────────────────────────────────────


def _build_joint_map(
    model: mujoco.MjModel, solver_joint_names: list[str]
) -> list[int]:
    """Return list of qpos indices aligned with solver_joint_names."""
    mj_name_to_qpos: dict[str, int] = {}
    for i in range(model.njnt):
        if model.jnt_type[i] in (2, 3):  # slide or hinge (revolute)
            mj_name_to_qpos[model.joint(i).name] = int(model.jnt_qposadr[i])

    missing = [n for n in solver_joint_names if n not in mj_name_to_qpos]
    if missing:
        print(f"[WARN] Solver joints not found in MuJoCo model: {missing}")

    return [mj_name_to_qpos[n] for n in solver_joint_names if n in mj_name_to_qpos]


def replay_in_mujoco(
    q_traj: np.ndarray,
    solver_joint_names: list[str],
    ts: np.ndarray,
    fps: float,
    speed: float,
) -> None:
    """Open a MuJoCo viewer and replay the joint trajectory."""
    model = mujoco.MjModel.from_xml_path(str(_URDF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    qpos_indices = _build_joint_map(model, solver_joint_names)
    N = len(q_traj)
    dt = 1.0 / fps / speed

    print(f"\nMuJoCo viewer open — {N} frames @ {fps:.0f} fps (speed ×{speed:.1f})")
    print("Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = 0
        while viewer.is_running():
            t0 = time.monotonic()

            for col, qidx in enumerate(qpos_indices):
                data.qpos[qidx] = float(q_traj[i, col])

            mujoco.mj_forward(model, data)
            viewer.sync()

            sleep_s = dt - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

            i = (i + 1) % N


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Root of the LeRobot dataset (e.g. datasets/my_dataset)",
    )
    parser.add_argument(
        "--episode", "-e",
        type=int,
        default=None,
        help="Episode index to replay (default: first available)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time). Default: 1.0",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib plot of joint angles after IK",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Show matplotlib plot and skip MuJoCo replay",
    )
    parser.add_argument(
        "--plot-save",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the joint-angle plot to this file (e.g. ik_joints.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.dataset.resolve()
    if not root.exists():
        sys.exit(f"[ERROR] Dataset path not found: {root}")

    # Larger max_joint_delta for offline batch IK (not real-time smoothing)
    config = KinematicsConfig(max_joint_delta=math.pi / 6)
    print("Initialising IK solver (JIT warm-up)...")
    solver = KinematicsSolver(config=config)
    print(f"Solver ready. Joints ({solver.num_joints}): {solver.joint_names}\n")

    ts, left_poses, right_poses = load_episode(root, args.episode)
    print_dataset_pose_stats(left_poses, right_poses)
    print_retargeting_origin_stats(left_poses, right_poses, solver)

    q_traj = compute_ik_trajectories(left_poses, right_poses, solver)

    print_retargeting_trajectory_stats(left_poses, right_poses, solver)
    print_ik_joint_trajectory_stats(q_traj, solver.joint_names)

    if args.plot or args.plot_only or args.plot_save is not None:
        plot_joint_trajectories(
            q_traj,
            joint_names=solver.joint_names,
            ts=ts,
            save_path=args.plot_save,
            show=args.plot or args.plot_only,
        )

    if not args.plot_only:
        replay_in_mujoco(
            q_traj,
            solver_joint_names=solver.joint_names,
            ts=ts,
            fps=30.0,
            speed=args.speed,
        )


if __name__ == "__main__":
    main()
