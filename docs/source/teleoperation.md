# Teleoperation

Ultima modificacion: 2026-07-15 11:26:49 -05 -0500

HandUMI produces robot-agnostic live tool poses and gripper commands. A selected
robot embodiment maps those commands to its kinematics; an optional hardware
backend sends them to physical arms. Start in simulation and connect hardware
only after tracking, calibration, and motion mapping behave correctly.

## Live Simulation

Select any supported robot model through `--robot`:

```bash
handumi-teleop-sim --device meta --robot <robot_id> \
  --workspace-camera --space-start
```

For example, using the currently supported Piper embodiment:

```bash
TARGET_ROBOT=piper
handumi-teleop-sim --device meta --robot "$TARGET_ROBOT" \
  --workspace-camera --space-start
```

OpenArm v1 uses the same command and defaults to its official arms-down pose:

```bash
handumi-teleop-sim --device meta --robot openarmv1 \
  --home-pose down --space-start
```

This opens Viser with the live robot model and Rerun with tracking, TCP trails,
gripper widths, and the left wrist, workspace, and right wrist cameras. Nothing
is recorded. Use `--device pico` for PICO.

Add a task scene with:

```bash
handumi-teleop-sim --device meta --robot "$TARGET_ROBOT" --scene cube_in_box
```

`--context-camera` is an alias for `--workspace-camera`. The devices come from
`cameras.left_wrist`, `cameras.workspace`, and `cameras.right_wrist` in
`configs/rig.yaml`. When using `--cam-ids`, provide three IDs in that order.

Viser shows the robot and Rerun shows tracking and camera trails. Use `--no-rerun` or `--no-viser` when a viewer is not needed.

### Start and Reset

- Double clap starts the enabled, tracked arms from home.
- Another double clap clears anchors and returns them home.
- With `--space-start`, Space starts any idle enabled arm.
- Tracking loss cancels pending motion and holds the latest command.

## Real Robot Teleoperation

The HandUMI tracking and control flow remains the same for every robot. Physical
teleoperation additionally requires a backend for the selected manufacturer and
model; simulation or replay support alone does not imply hardware support.

The general interface is:

```bash
handumi-setup-hardware --robot <robot_id> --device meta
handumi-teleop-real --robot <robot_id> --device meta
```

| Robot | Live simulation | Real teleoperation |
| --- | --- | --- |
| Piper | Supported | Supported |
| OpenArm v1 | Supported (kinematic) | Supported through optional `openarm` backend |
| Axol | Supported | Not yet supported |
| Other robots | Add an embodiment | Add a hardware backend |

See [Add a New Robot Embodiment](development/new_embodiment.md) for the common
interface used to add future manufacturers and models without changing the
HandUMI capture workflow.

:::{dropdown} Example: physical Piper arms

First complete the robot-independent
[HandUMI Setup and Calibration](setup.md), then install the Piper backend and
map its CAN adapters:

```bash
uv sync --extra piper
handumi-setup-hardware --robot piper --device meta \
  --skip-feetech-map --skip-feetech-calibration
handumi-teleop-real --device meta --robot piper
```

The CAN wizard maps the right Piper adapter first and the left adapter second,
then stores that machine-local mapping under `robots.piper.can` in
`configs/rig.yaml`. Use `--skip-can-map` only after verifying an existing
mapping.

Start with one arm:

```bash
handumi-teleop-real --device meta --robot piper --side right
```
:::

:::{dropdown} Example: physical OpenArm v1

Install the optional backend, then run the guided setup:

```bash
uv sync --extra openarm
handumi-setup-hardware --robot openarmv1 --device meta
```

The wizard maps the right adapter first and the left adapter second, configures
CAN-FD at 1/5 Mbps, verifies J1-J8 without enabling motion, reuses complete
Feetech calibration, and refuses real teleoperation until the OpenArm tool has
an explicit Controller-to-TCP calibration.

Mechanical-zero calibration moves the robot and is never automatic:

```bash
handumi-setup-hardware --robot openarmv1 --device meta \
  --skip-can-map --skip-feetech-map --skip-feetech-calibration \
  --calibrate-openarm-zero
```

Start with the right arm, reduced translation, and the down pose:

```bash
handumi-teleop-real --device meta --robot openarmv1 --side right \
  --home-pose down --translation-scale 0.25 --space-start
```

Validate right, then left, before using `--side both`. Keep the emergency stop
reachable during every moving step.
:::

### Safety

Keep the workspace clear and an emergency stop accessible. Enforce joint, velocity, acceleration, workspace, and collision limits. Run `handumi-teleop-real --help` for backend-specific options.

To inspect an existing recording rather than live motion, continue with
[Quality Assurance](workflows/datasets.md).
