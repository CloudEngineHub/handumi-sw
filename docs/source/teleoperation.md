# Deploy to a Robot

HandUMI data collection is complete before this stage. Deployment selects a
target embodiment, retargets the HandUMI tool-tip trajectories, and optionally
connects a hardware backend. Adding UR, YAM, or another brand should not change
the recording workflow or raw dataset schema.

## Choose a Target Model

Set the target explicitly. Piper is one available example:

```bash
TARGET_ROBOT=piper
handumi-teleop-sim --device meta --robot "$TARGET_ROBOT"
```

Use `--device pico` for PICO. Add a physical task scene with:

```bash
handumi-teleop-sim --device meta --robot "$TARGET_ROBOT" --scene cube_in_box
```

Viser shows the robot and Rerun shows tracking and camera trails. Use `--no-rerun` or `--no-viser` when a viewer is not needed.

For an offline dataset, use `handumi-replay-in-sim` as described in the
[Dataset Pipeline](workflows/datasets.md#optional-retarget-to-a-robot). Always
validate the same target model and calibration in simulation before moving
physical hardware.

## Real Robot Backends

Real hardware support is separate from model/replay support. A robot can be
used for conversion and simulation without having a real-time hardware backend.

| Robot | Conversion / simulation | Real-time hardware |
| --- | --- | --- |
| Piper | Supported | Supported |
| Axol | Supported | Not yet supported |
| Other robots | Add an embodiment | Add a vendor backend |

See [Add a New Robot Embodiment](development/new_embodiment.md) for the stable
interface expected from future UR, YAM, and other integrations.

### Piper Real Arm Setup

This section is needed only when commanding physical Piper arms. First complete
the robot-independent [HandUMI Setup and Calibration](setup.md), then install
the Piper backend and map its CAN adapters:

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

### Controls and Safety

- Double clap starts tracked arms from home.
- Another double clap clears anchors and returns the arms home.
- `--space-start` enables keyboard start.
- Tracking loss cancels pending motion and holds the latest command.

Keep the workspace clear and an emergency stop accessible. Enforce joint, velocity, acceleration, workspace, and collision limits. Run `handumi-teleop-real --help` for backend-specific options.

Offline replay directly on real arms is not currently exposed. Do not treat
`handumi-teleop-real` as a dataset replay command.
