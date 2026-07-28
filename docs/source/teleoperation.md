# Teleoperation

HandUMI produces robot-agnostic live tool poses and gripper commands. A selected
robot embodiment maps those commands to its kinematics; an optional hardware
backend sends them to physical arms. Start in simulation and connect hardware
only after tracking, calibration, and motion mapping behave correctly.

## Live Simulation

Select any supported robot model through `--robot`:

```bash
handumi teleop --device meta --robot <robot_id>
```

For example, using the currently supported Piper embodiment:

```bash
TARGET_ROBOT=piper
handumi teleop --device meta --robot "$TARGET_ROBOT"
```

OpenArm v1 uses the same command and starts from its configured `home_q`:

```bash
handumi teleop --device meta --robot openarmv1
```

This opens Viser with the live robot model and Rerun with tracking, TCP trails,
gripper widths, and both wrist cameras. Nothing is recorded. Use `--device pico`
for PICO.

Add a task scene with:

```bash
handumi teleop --device meta --robot "$TARGET_ROBOT" --scene cube_in_box
```

Teleoperation shows both wrist cameras by default, so `--cameras` is only
needed to change that selection — for instance to add the overhead view:

```bash
handumi teleop --device meta --robot "$TARGET_ROBOT" \
  --cameras left_wrist,right_wrist,workspace
```

It accepts the logical names `left_wrist`, `right_wrist`, and `workspace`;
their physical device IDs come only from the corresponding entries in
`configs/rig.yaml`, which is where each camera is declared once. Use
`--skip-cameras` to run without any camera view.

Viser shows the robot and Rerun shows tracking and camera trails. Use `--no-rerun` or `--no-viser` when a viewer is not needed.

### Start and Reset

Arms sit idle at home until they are started, and the same gesture stops them
again:

- **Double-squeeze a gripper**: start the enabled, tracked arms from home.
- **Double-squeeze again**: clear the anchors and return them home. This is the
  stop.
- Tracking loss cancels pending motion and holds the latest command.

Two optional ways to start exist for when squeezing a gripper is impractical —
neither replaces the double-squeeze, which stays active in every mode:

- `--space-start`: also start idle arms by pressing Space in the terminal.
  Space only *starts*; it is not a stop or pause key.
- `--auto-start`: start on their own once controller tracking has been valid
  for `--auto-start-delay-s` (default 5), with no gesture at all.

## Real Robot Teleoperation

The HandUMI tracking and control flow remains the same for every robot. Physical
teleoperation additionally requires a backend for the selected manufacturer and
model; simulation or replay support alone does not imply hardware support.

The general interface is:

```bash
handumi setup --robot <robot_id> --device meta
handumi teleop-real --robot <robot_id> --device meta
```

| Robot | Live simulation | Real teleoperation |
| --- | --- | --- |
| Piper | Supported | Supported |
| OpenArm v1 | Supported (kinematic) | Supported through optional `openarm` backend |
| TRLC-DK1 | Supported (kinematic) | Not yet supported |
| Axol | Supported | Not yet supported |
| Other robots | Add an embodiment | Add a hardware backend |

See [Add a New Robot Embodiment](development/new_embodiment.md) for the common
interface used to add future manufacturers and models without changing the
HandUMI capture workflow.

Complete the hardware-specific preparation before commanding a physical robot:

- [Piper Hardware Setup](physical_robots/piper_setup.md)
- [OpenArm v1 Hardware Setup](physical_robots/openarm_v1_setup.md)

Both guides start with single-arm validation before enabling both arms.

## Real Robot Recording

Use `handumi teleop-record` when the real robot should be driven live and saved
as a joint-level dataset:

```bash
handumi teleop-record --robot <robot_id> --device meta \
  --output-dir outputs/my-dataset
```

This command has its own parser and operational defaults. It does not use
`--record` on `handumi teleop`; the plain `handumi teleop` command is reserved
for live simulation. `--resume` with the same `--output-dir` verifies and
appends to that finalized local dataset.

To stream the context and wrist cameras into a PICO headset independently of
the selected robot, see [PICO Remote Vision](workflows/pico_remote_vision.md).

### Safety

Keep the workspace clear and an emergency stop accessible. Enforce joint, velocity, acceleration, workspace, and collision limits. Run `handumi teleop-real --help` for backend-specific options.

To inspect an existing recording rather than live motion, continue with
[Replay a Local Recording in Simulation](workflows/replay_in_sim.md), then run
the checks in [Quality Assurance](workflows/datasets.md).

```{toctree}
:hidden:
:maxdepth: 1

workflows/pico_remote_vision
```
