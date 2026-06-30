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
- Python 3.12.
- `uv` installed.
- Two USB wrist cameras.
- Two Feetech servos used as gripper encoders.
- Optional: PICO / Meta Quest tracking for later capture stages.

## Installation

```bash
git clone <repo-url> handumi-sw
cd handumi-sw
uv sync --python "$(command -v python3.12)"
source .venv/bin/activate
```

Verify:

```bash
python --version
PYTHONPATH=src python scripts/record_handumi.py --help
```

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

### 1. Ports

```bash
PYTHONPATH=src python scripts/setup/setup_ports.py
```

Follow the prompts: disconnect all devices, then connect left Feetech, left
camera, right Feetech, and right camera. This assigns Feetech IDs, saves
`configs/feetech.yaml`, and saves camera indices in `configs/cameras.yaml`.

Check encoder ticks:

```bash
PYTHONPATH=src python scripts/setup/calibrate_grippers.py monitor
```

### 2. Gripper Calibration

Run calibration and follow the terminal prompts:

```bash
PYTHONPATH=src python scripts/setup/calibrate_grippers.py calibrate
```

It asks for the max gripper opening in mm, then records:

```text
left max_width_mm, open_ticks, closed_ticks
right max_width_mm, open_ticks, closed_ticks
```

Per frame, HandUMI records raw ticks, normalized width, width in mm, and state
width in meters.

### 3. Live Monitor

Before recording, run the live Rerun monitor:

```bash
PYTHONPATH=src python -m handumi.capture.teleoperate_handumi \
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

### 4. Record Dataset

```bash
PYTHONPATH=src python scripts/record_handumi.py \
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

### 5. Inspect With LeRobot

```bash
lerobot-dataset-viz \
  --repo-id local/handumi_width_test \
  --root outputs/datasets/handumi_width_test \
  --episode-index 0
```

## Record With Tracking

After the hardware checkpoint works, enable PICO streams with:

```bash
PYTHONPATH=src python scripts/record_handumi.py \
  --use-pico \
  --pico-mandos \
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
├── scripts/                 # Manual hardware and pipeline scripts
├── src/handumi/             # Core package
├── tests/                   # Automated tests
└── utils/                   # Upload helpers
```

```text
src/handumi/
├── capture/                 # HandUMI raw recorder
├── cameras/                 # USB wrist cameras
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
| [docs/architecture.md](docs/architecture.md) | System architecture, raw schema, configs, and manual scripts |
| [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) | Meta Quest/WebXR tracking and live Viser plan |
| [docs/add-new-embodiment.md](docs/add-new-embodiment.md) | How to add a new robot embodiment |
