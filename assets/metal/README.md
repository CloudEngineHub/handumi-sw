# MakerMods Metal bimanual assets

This directory contains a simulation/replay model for two MakerMods Metal
6-DoF arms with their stock parallel-jaw gripper.

Source: `makermods-robotics/metal-python-ros` branch `humble`, commit
`ef4181f1305cbcfc63431d3bcfb96f5fb7f72763`, licensed under MIT (see
`LICENSE.makermods`, copied from `metal_sdk/LICENSE` in that commit). The
kinematics and the ten visual meshes come from
`metal_ros2/src/metal_description/` (`urdf/metal_with_gripper.urdf` and
`meshes/*.STL`).

`metal_bimanual.urdf` makes only the adaptations required by HandUMI:

- two copies are namespaced as `left_` and `right_`;
- both bases are placed 0.60 m apart in a +X-right, +Y-forward, +Z-up world,
  yawed +90 degrees so the vendor zero pose reaches into the shared workspace;
- vendor joint origins, axes, limits, efforts, and the driven/mimic finger
  pair (`joint7` drives, `joint8` mirrors it) are preserved verbatim -- the
  same gripper structure `assets/piper/piper.urdf` uses;
- the vendor's `velocity="0"` finger limits are raised to `0.1` m/s so the
  prismatic joints remain usable;
- `package://metal_description/...` mesh paths are rewritten as relative
  `meshes/...` paths shared by both sides;
- per-visual inline colors are hoisted into named materials, replacing the
  CAD's silver/blue engineering colors with the production arm's matte black
  finish (finger carriages slightly lighter so the jaw reads in the viewer);
- vendor inertial blocks are dropped (kinematic model, matching
  `assets/yam/yam_bimanual.urdf`), but the vendor collision meshes are kept:
  pyroki capsulizes them at load time for the opt-in IK collision penalties
  configured in `configs/robots/metal.yaml`;
- one fixed TCP link per side (`left_tcp`/`right_tcp`) marks the grasp point
  between the finger pads, 0.085 m from `gripper_base` along the approach
  axis (fingertips end at 0.0914 m per `link7.STL`).

This is currently a kinematic model. It does not register a MakerMods
hardware backend or a bimanual MuJoCo contact model.
