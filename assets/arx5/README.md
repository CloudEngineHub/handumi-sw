# ARX X5 bimanual assets

This directory contains a simulation/replay model for two ARX X5 6-DoF arms
with their stock parallel gripper.

Source: `real-stanford/arx5-sdk` commit
`c160db7834530a3d80fec524547c74e1d4fa8035`, licensed under MIT (see
`LICENSE.arx5-sdk`, copied from that commit). The kinematics come from
`models/X5.urdf` and the meshes from `models/meshes/X5/`. The joint origins
and axes are identical to the vendor's `x5.urdf` in `ARXroboticsX/X5`.

`arx5_bimanual.urdf` makes only the adaptations required by HandUMI:

- two copies are namespaced as `left_` and `right_`;
- both bases are placed 0.60 m apart in a +X-right, +Y-forward, +Z-up world,
  yawed +90 degrees so the vendor +X (forward) axis points into the shared
  workspace, and shifted 1 mm along +X (mounts at X=-0.299/+0.301) because
  the vendor wrist is offset 1 mm sideways (`joint5` y=-0.001) and both arms
  are identical rather than mirrored; this centres the TCP pair at X=0;
- vendor joint origins and axes are preserved verbatim; the URDF's
  placeholder `+/-10 rad` limits and `1000 rad/s` velocities are replaced by
  the limits the SDK controller enforces for `"X5"` in `include/app/config.h`
  (position min/max and `joint_vel_max`);
- `link6.STL` is split into `link6_body.stl` and the two finger meshes
  `finger_pos_y.stl`/`finger_neg_y.stl` by partitioning its face-connected
  components (pad, slider carriage and carriage screws per side); no triangle
  is changed. The CAD fingers sit 0.0301 m off the jaw centreline, so each
  finger visual is offset by that amount and two prismatic joints span 0
  (closed) to 0.044 m, matching the SDK's 0.088 m `gripper_width`; `joint7`
  drives the +Y finger and `joint8` mirrors it through a URDF mimic tag, the
  same structure `assets/piper/piper.urdf` uses;
- `link3.STL` is split the same way into `link3_arm.stl` (the two shell
  halves spanning the whole link) and `link3_motor.stl` (the joint4 motor
  housing), so the forearm can carry the production arm's brushed aluminium
  while the motor stays black;
- the SolidWorks export's uniform grey is replaced by the production X5
  finish: matte black links, motors, gripper and fingers, the aluminium
  forearm shell, and a white base like ARX's bimanual base station;
- vendor inertial and collision blocks are dropped (kinematic model,
  matching `assets/yam/yam_bimanual.urdf`);
- one fixed TCP link per side (`left_tcp`/`right_tcp`) sits at the
  fingertip plane, 0.1466 m along link6 +X where the finger meshes end, with
  +Z along the approach axis and Y across the jaw. The SDK's own `eef_link`
  is at 0.145 m.

The finger split was produced with this face partition (trimesh): components
that start at link6 X >= 0.06 m and lie entirely beyond |Y| >= 0.03 m are
fingers, everything else stays on link6.

This is currently a kinematic model. It does not register an ARX hardware
backend or a bimanual MuJoCo contact model.
