# Franka Emika Panda — right HandUMI

One Panda arm with its parallel gripper, driven by the **right** HandUMI
controller and encoder. Supports simulation/replay; no physical Franka backend.

The canonical HandUMI identifier and asset directory are `franka_emika_panda`,
matching Menagerie. Model filenames retain `panda.xml`, `panda.urdf`, and
`panda_handumi.xml`; use `--robot franka_emika_panda` in commands.

## Source and license

Source: [Google DeepMind MuJoCo Menagerie, `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda),
commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`, Apache-2.0
(`LICENSE.menagerie`). `panda.xml` and `assets/` are unmodified copies.
Menagerie derived the model from Franka's `franka_ros/franka_description`;
see the upstream README for its mesh-processing and modeling history.

## Derivation

Run `.venv/bin/python assets/franka_emika_panda/generate.py` to regenerate `panda.urdf`
and `panda_handumi.xml` from `panda.xml`. No meshes are regenerated.

- URDF joints/links are namespaced `right_`, anchored to a fixed `world`
  link. Joint position limits, body transforms, axes, inertias, visual
  materials and collision geometry follow Menagerie.
- URDF velocity limits are conservative simulation values (2 rad/s arm,
  0.2 m/s fingers). Menagerie supplies no joint velocity limits; these
  additions are not a hardware controller specification.
- `right_finger_joint2` mimics `right_finger_joint1`: eight independent
  joints, seven revolute and one finger slide.
- `right_tcp` is the midpoint of the opposing main fingertip pads,
  0.1029 m along hand +Z (0.0584 m mount + 0.0445 m pad center), with the
  hand's orientation. The adapted MJCF has the same TCP site.
- Adapted MJCF arm actuators are named after their joints. The gripper
  retains the equality constraint and split tendon, with actuator name
  `finger_joint1`. Control is finger displacement (0–0.04 m), replacing
  Menagerie's 0–255 scale without changing the position-servo gain.
- Home is Menagerie's `home` keyframe with one independent finger value.
  Full aperture is 0.08 m. Replay preserves physical aperture, clipped to
  that range.

Canonical datasets contain `right_joint1.pos` through `right_joint7.pos`
(radians), then `right_gripper.width_m` (meters). No left-arm columns.
The portable table transform is identity: +X forward, +Y left, +Z up from
the base. Physical installations require measured deployment calibration.

Tests compare URDF FK against upstream MuJoCo at multiple poses, load all
meshes, verify canonical/gripper round trips, replay right-only trajectories,
and exercise the adapted MuJoCo actuator. See
[the Panda workflow](../../docs/source/workflows/franka_emika_panda.md).
