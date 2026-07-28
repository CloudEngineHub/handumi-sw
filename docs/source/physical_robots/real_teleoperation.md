# Physical Robot Teleoperation

Drive real arms live with HandUMI, and optionally record what they do as a
joint-level dataset. This is the entry point for everything that commands
hardware; the robot-free capture flow is
[Record Demonstrations](../record.md).

Validate the same robot in
[Simulated Teleoperation](../teleoperation.md) first. Tracking, calibration,
and motion mapping must already behave correctly there — the control flow is
identical and only the destination changes.

## Support by Robot

The HandUMI tracking and control flow remains the same for every robot.
Physical teleoperation additionally requires a backend for the selected
manufacturer and model; simulation or replay support alone does not imply
hardware support.

| Robot | Live simulation | Real teleoperation |
| --- | --- | --- |
| Piper | Supported | Supported |
| OpenArm v1 | Supported (kinematic) | Supported through optional `openarm` backend |
| TRLC-DK1 | Supported (kinematic) | Not yet supported |
| Axol | Supported | Not yet supported |
| Other robots | Add an embodiment | Add a hardware backend |

See [Add a New Robot Embodiment](../development/new_embodiment.md) for the
common interface used to add future manufacturers and models without changing
the HandUMI capture workflow.

## Hardware Preparation

Complete the hardware-specific preparation before commanding a physical robot.
Both guides start with single-arm validation before enabling both arms:

- [Piper Hardware Setup](piper_setup.md)
- [OpenArm v1 Hardware Setup](openarm_v1_setup.md)

## Teleoperate

The general interface is:

```bash
handumi setup --robot <robot_id> --device meta
handumi teleop-real --robot <robot_id> --device meta
```

## Record a Real-Robot Dataset

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

Unlike a robot-free HandUMI capture, this dataset is bound to the robot that
produced it. To collect demonstrations that can be retargeted to any supported
embodiment later, record with HandUMI alone.

## See the Cameras in the Headset

While teleoperating, the context and wrist camera feeds can be streamed into a
PICO headset so the operator does not have to look at the workstation screen.
It works independently of the selected robot and does not change the
teleoperation commands above.

- [PICO Remote Vision](pico_remote_vision.md)

## Safety

Keep the workspace clear and an emergency stop accessible. Enforce joint,
velocity, acceleration, workspace, and collision limits. Run
`handumi teleop-real --help` for backend-specific options.

To inspect an existing recording rather than live motion, continue with
[Replay a Local Recording in Simulation](../workflows/replay_in_sim.md), then
run the checks in [Quality Assurance](../workflows/datasets.md).

```{toctree}
:hidden:
:maxdepth: 1

pico_remote_vision
```
