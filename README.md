# HandUMI Software

Software stack for recording HandUMI bimanual raw demonstrations as
LeRobot-compatible datasets.

HandUMI records data without a robot in the collection loop:

```text
left/right wrist cameras
+ left/right Feetech gripper encoder widths
+ optional left/right VR tracking poses
-> HandUMI raw LeRobot dataset
```

Robot-specific datasets for Piper, Axol, and other embodiments are derived
later through offline retargeting / IK.

## Requirements

- Linux workstation with USB access.
- `uv` installed.
- Two USB wrist cameras.
- Two Feetech servos used as gripper encoders.
- Optional: PICO / Meta Quest tracking for later capture stages.

## Installation

From a fresh clone:

```bash
git clone <repo-url> handumi-sw
cd handumi-sw
uv sync
source .venv/bin/activate
uv pip install -e .
```

If you are already inside the repository, start at `uv sync`.

This installs LeRobot Feetech support, Rerun visualization, OpenCV camera
support, and the `handumi-*` CLI commands.

## Checkpoint 1: Cameras + Feetech Width

This is the first hardware target:

```text
left USB wrist camera
right USB wrist camera
left Feetech servo encoder  -> left gripper opening
right Feetech servo encoder -> right gripper opening
LeRobotDataset output
```

PICO / Meta Quest tracking is optional and disabled by default for this
checkpoint.

### 1. Feetech Ports And IDs

Connect the Feetech USB adapters and scan all serial ports:

```bash
handumi-find-servos --all-ports --start-id 0 --end-id 20
```

Expected convention:

```text
left gripper  -> servo ID 0
right gripper -> servo ID 1
```

If a servo still has another ID, connect one servo at a time and write the
target ID:

```bash
handumi-setup-servos \
  --write-id left \
  --current-id <current_left_id> \
  --left-id 0 \
  --left-port /dev/ttyUSB_LEFT \
  --config configs/feetech.yaml

handumi-setup-servos \
  --write-id right \
  --current-id <current_right_id> \
  --right-id 1 \
  --right-port /dev/ttyUSB_RIGHT \
  --config configs/feetech.yaml
```

Then save the final left/right port mapping:

```bash
handumi-setup-servos \
  --left-id 0 \
  --right-id 1 \
  --left-port /dev/ttyUSB_LEFT \
  --right-port /dev/ttyUSB_RIGHT \
  --config configs/feetech.yaml
```

If both servos share one serial bus, use:

```bash
handumi-setup-servos \
  --left-id 0 \
  --right-id 1 \
  --port /dev/ttyUSB0 \
  --config configs/feetech.yaml
```

### 2. Calibrate Gripper Opening

Measure the physical maximum opening of the grippers in millimeters. Then run:

```bash
handumi-calibrate-grippers \
  --config configs/feetech.yaml \
  --max-width-mm 80
```

The command asks you to close both grippers, then open both grippers. It stores:

```text
closed_ticks
open_ticks
max_width_mm
```

Per frame, HandUMI records raw ticks, normalized width, width in mm, and state
width in meters.

### 3. Cameras

Connect both USB wrist cameras and run:

```bash
handumi-find-cameras
```

Use the detected indices as:

```text
first --cam-ids value  -> observation.images.left_wrist
second --cam-ids value -> observation.images.right_wrist
```

### 4. Live Monitor

Before recording, run the live Rerun monitor:

```bash
handumi-teleoperate \
  --cam-ids 0 2 \
  --feetech-config configs/feetech.yaml \
  --fps 30
```

This does not save data. It streams:

```text
left/right wrist camera images
left/right raw Feetech ticks
left/right normalized gripper opening
left/right gripper opening in mm
```

Use it to verify that camera assignment, servo IDs, ports, and calibration are
correct before recording.

### 5. Record Dataset

```bash
handumi-record \
  --cam-ids 0 2 \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

Equivalent launcher:

```bash
bash bin/record.sh \
  --cam-ids 0 2 \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20
```

The raw dataset stores:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.state                  # float32[16]
action                             # float32[16]
observation.feetech.left_ticks
observation.feetech.right_ticks
observation.feetech.left_width_mm
observation.feetech.right_width_mm
observation.feetech.left_normalized
observation.feetech.right_normalized
```

`observation.state[14]` and `observation.state[15]` are the calibrated left/right
gripper widths in meters.

### 6. Inspect With LeRobot

```bash
lerobot-dataset-viz \
  --repo-id local/handumi_width_test \
  --root outputs/datasets/handumi_width_test \
  --episode-index 0
```

## Record With Tracking

After the hardware checkpoint works, enable PICO streams with:

```bash
handumi-record \
  --use-pico \
  --pico-mandos \
  --cam-ids 0 2 \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_pico_test \
  --output-dir outputs/datasets/handumi_pico_test
```

## Retarget / Replay

Convert a HandUMI source dataset to a robot-specific dataset:

```bash
bash bin/process_handumi_to_lerobot.sh \
  --embodiment piper \
  --output-name handumi-dataset-v2-piper \
  --output-root outputs/datasets/handumi-dataset-v2-piper
```

Inspect retargeting:

```bash
python scripts/replay_pico_ik.py --embodiment piper --episode 0 --visualize
python scripts/compare_axis.py --embodiment axol --episode 0
```

Replay a Piper robot-specific dataset:

```bash
bash bin/piper/replay_from_dataset.sh --episode 0 --dry-run
```

## Project Layout

```text
.
├── assets/                  # Robot URDFs and meshes
├── bin/                     # Shell launchers
├── configs/                 # Camera, Feetech, tracking configs
├── docs/                    # Architecture and embodiment guide
├── scripts/                 # Pipeline CLI wrappers
├── src/handumi/             # Core package
├── tests/                   # Automated tests
└── utils/                   # Upload helpers
```

```text
src/handumi/
├── capture/                 # HandUMI raw recorder
├── cameras/                 # USB wrist cameras
├── cli/                     # Hardware setup commands
├── dataset/                 # Raw schema, LeRobot IO, conversion
├── feetech/                 # Feetech encoder bus/calibration/gripper widths
├── replay/                  # PICO IK replay and robot replay
├── retargeting/             # Raw/PICO poses to robot targets
├── robots/                  # Piper/Axol embodiment registry and IK specs
└── tracking/                # PICO / tracker backends
```

## Docs

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture, raw schema, configs, and entrypoints |
| [docs/add-new-embodiment.md](docs/add-new-embodiment.md) | How to add a new robot embodiment |
