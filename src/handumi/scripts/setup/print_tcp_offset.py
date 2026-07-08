"""Print the fixed offset from the IK solver's EE link to the gripper tip.

The solver's FK stops at the wrist flange (e.g. Piper ``link6``), but the
TCP marker in Viser — and anything else "at the TCP" — should sit at the
gripper tip. This script derives that fixed transform straight from the
embodiment's URDF, per side:

  1. FK pose of the solver's EE link at q = 0 (direct kinematics).
  2. Gripper-tip position: the extreme point of the finger meshes along the
     EE approach axis, at q = 0 (gripper closed).
  3. Offset = tip expressed in the EE link frame (their difference,
     rotated into the EE frame).

Both sides are printed so the mirror symmetry acts as a sanity check; the
suggested value goes into ``EmbodimentRuntime.tcp_offset_ee``
(``handumi/robots/registry.py``). Re-run after any URDF/gripper change.

Usage:

    handumi-print-tcp-offset             # piper (default)
    handumi-print-tcp-offset --robot piper
"""

from __future__ import annotations

import argparse

import numpy as np


def _tip_offset_in_ee_frame(
    urdf, ee_link: str, finger_links: list[str], mesh_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (EE world position, tip offset in EE frame) at q = 0."""
    import trimesh

    T_ee = urdf.get_transform(ee_link)
    ee_world = T_ee[:3, 3]
    T_ee_inv = np.linalg.inv(T_ee)

    # Collect every finger-mesh vertex in the EE frame; the tip is the
    # extreme point along the approach axis (EE +Z), averaged laterally.
    points_ee: list[np.ndarray] = []
    for link_name in finger_links:
        link = next(l for l in urdf.robot.links if l.name == link_name)
        T_link = urdf.get_transform(link_name)
        for visual in link.visuals:
            if visual.geometry.mesh is None:
                continue
            mesh = trimesh.load(f"{mesh_dir}/{visual.geometry.mesh.filename}")
            V = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
            points_ee.append((T_ee_inv @ T_link @ V.T).T[:, :3])
    if not points_ee:
        raise SystemExit(f"No finger meshes found on links {finger_links}")
    P = np.vstack(points_ee)
    z_max = P[:, 2].max()
    tip_region = P[P[:, 2] > z_max - 0.002]  # vertices within 2mm of the tip
    tip_ee = np.array([tip_region[:, 0].mean(), tip_region[:, 1].mean(), z_max])
    return ee_world, tip_ee


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--robot", choices=["piper"], default="piper")
    args = p.parse_args()

    import yourdfpy

    from handumi.robots.registry import load_embodiment

    runtime = load_embodiment(args.robot)
    urdf = yourdfpy.URDF.load(
        str(runtime.urdf_path), mesh_dir=str(runtime.urdf_path.parent)
    )
    urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))

    # Piper naming; extend per-embodiment if another robot gains a gripper.
    sides = {
        "left": ("left_link6", ["left_link7", "left_link8"]),
        "right": ("right_link6", ["right_link7", "right_link8"]),
    }

    offsets = {}
    for side, (ee_link, finger_links) in sides.items():
        ee_world, tip_ee = _tip_offset_in_ee_frame(
            urdf, ee_link, finger_links, str(runtime.urdf_path.parent)
        )
        offsets[side] = tip_ee
        print(f"{side}:")
        print(f"  EE link ({ee_link}) world position: {np.round(ee_world, 4)}")
        print(f"  gripper tip offset in EE frame:     {np.round(tip_ee, 4)}")

    mismatch = np.abs(offsets["left"] - offsets["right"]).max()
    print(f"\nleft/right offset mismatch: {mismatch * 1000:.2f} mm "
          f"({'OK' if mismatch < 0.002 else 'CHECK URDF — sides should match'})")
    print(f"current registry tcp_offset_ee: {runtime.tcp_offset_ee}")
    mean = (offsets["left"] + offsets["right"]) / 2.0
    print(f"suggested tcp_offset_ee:        ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")


if __name__ == "__main__":
    main()
