# Piper Hardware Setup

This procedure prepares two physical AgileX Piper arms for HandUMI real
teleoperation. Complete the robot-independent
[HandUMI Setup and Calibration](../setup.md) first.

## Safety and prerequisites

Before connecting or commanding the arms:

- Clear the complete arm workspace and keep the emergency stop reachable.
- Power both arms, but stop every other process that may use their CAN buses.
- Install the Piper backend with `uv sync --extra piper`.
- Connect one USB-to-CAN adapter per arm.
- Verify tracking and motion mapping in simulation before enabling hardware.

## Install and map the CAN adapters

Run the guided hardware setup:

```bash
uv sync --extra piper
handumi setup --robot piper --device meta \
  --skip-feetech-map --skip-feetech-calibration
```

The wizard maps the right Piper adapter first and the left adapter second. It
stores the machine-local result under `robots.piper.can` in
`configs/rig.yaml`. Follow the prompts to disconnect and reconnect adapters so
that each physical side is identified correctly.

Use `--skip-can-map` only after verifying an existing mapping. Rerun the wizard
whenever adapters, USB ports, or arm assignments change.

## Verify CAN and troubleshoot the mapping

Check that both arms are powered, both adapters are present, and no other
process owns the CAN interfaces. If an interface is down or bus-off, stop
teleoperation, inspect power and wiring, and rerun the guided setup from the
previous section.

Do not continue to real motion until the wizard identifies both physical sides
and communication is stable.

## Calibrate the Piper gripper closed positions

Calibrate each Piper gripper before the first real teleoperation, after
replacing a gripper, or whenever a physically closed gripper reports a
non-zero or negative position. From the repository, run:

```bash
uv run handumi calibrate piper-grippers
```

The command connects both Piper controllers but does not send arm-joint
targets. For each side, it disables only that gripper motor and continuously
prints its raw feedback. Close the gripper gently and consistently against its
physical stop, then press Enter. The value visible at that instant becomes the
logical `opening=0.0` for that side; it is measured during the procedure and is
not hardcoded.

To calibrate only one side:

```bash
uv run handumi calibrate piper-grippers --side left
uv run handumi calibrate piper-grippers --side right
```

The per-machine result is stored outside the repository at
`~/.cache/handumi/piper_grippers.yaml` (or under `$XDG_CACHE_HOME` when set).
Teleoperation uses the captured value both to normalize physical feedback and
as the closed command target. For example, if the captured raw value is
`-3000 um`, subsequent diagnostics report it as calibrated `0.000 mm` and a
normalized opening of `0.000000`, while retaining `-3000 um` as the raw value.

You can verify both grippers without running arm teleoperation:

```bash
uv run python tests/real/test_piper_gripper_diagnostic.py
```

With each gripper physically closed, expect `estado=CERRADO`,
`Piper=0.000000`, and `calibrado=0.000 mm`. Press Ctrl+C to stop the monitor.

## First real teleoperation

Start with simulation and the same robot profile:

```bash
handumi teleop --device meta --robot piper
```

After tracking, calibration, and simulated motion behave correctly, validate
one physical arm first:

```bash
handumi teleop-real --device meta --robot piper --side right
```

Keep the emergency stop reachable and confirm that the right controller moves
only the right arm. Stop and correct the CAN mapping if the wrong side moves.
Validate the left side separately before enabling both arms:

```bash
handumi teleop-real --device meta --robot piper --side left
handumi teleop-real --device meta --robot piper --side both
```

For shared controls, safety behavior, and tracking semantics, continue with
[Physical Robot Teleoperation](real_teleoperation.md). For common failures,
see [Troubleshooting](../troubleshooting.md).
