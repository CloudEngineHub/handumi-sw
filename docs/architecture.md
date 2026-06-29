# HandUMI Software Architecture

Ultima modificacion: 2026-06-29

`handumi-sw` contiene el codigo real para captura raw, configuracion de
hardware, datasets, retargeting, IK y replay. La regla principal es simple:

```text
HandUMI raw data = source of truth
robot-specific datasets = derived artifacts
```

La captura no debe depender de Piper, Axol, Viser, pyroki ni nombres de joints
de robots. Los robots aparecen despues, en conversion offline, replay o
deployment.

## Data Flow

```text
HandUMI hardware
  wrist cameras + Feetech encoders + optional tracking
        |
        v
HandUMI raw LeRobot dataset
  robot-agnostic
        |
        v
Offline retargeting / IK
        |
        v
Robot-specific LeRobot dataset
  Piper / Axol / other embodiments
        |
        v
Replay, training, deployment
```

The implemented paths are:

```text
HandUMI raw record:
wrist cameras + Feetech widths -> raw LeRobot dataset

Optional tracking record:
wrist cameras + Feetech widths + PICO/Meta Quest poses -> raw LeRobot dataset

Robot-specific conversion:
PICO body pose -> retargeting -> IK -> Piper/Axol dataset -> replay/sim
```

## Project Layout

```text
.
|-- assets/                  # Robot URDFs and meshes
|-- bin/                     # Shell launchers
|-- configs/                 # Hardware/configuration defaults
|-- docs/                    # Architecture and embodiment guide
|-- scripts/                 # Pipeline CLI wrappers
|-- src/handumi/             # Core package
|-- tests/                   # Automated tests
`-- utils/                   # Upload/helper scripts
```

```text
src/handumi/
|-- capture/                 # Recording loop and capture features
|-- cameras/                 # USB cameras and preview helpers
|-- cli/                     # Hardware setup commands
|-- tracking/                # PICO / tracking backends
|-- feetech/                 # Feetech servo encoders and calibration
|-- dataset/                 # LeRobot schemas, readers, writers, conversion
|-- retargeting/             # Human/wearable/PICO poses -> robot targets
|-- robots/                  # Embodiment registry, IK specs, sim wiring
|-- replay/                  # PICO IK replay and robot hardware replay
`-- utils/
```

## Module Boundaries

| Module | Owns | Must not own |
|--------|------|--------------|
| `capture/` | episode timing, frame reads, raw recording | robot IK, robot joint names |
| `cameras/` | camera discovery/read/preview | dataset conversion |
| `cli/` | hardware setup commands | reusable hardware logic |
| `tracking/` | PICO/tracker pose reads | robot-specific transforms |
| `feetech/` | servo IDs, encoder reads, width calibration | robot action layout |
| `dataset/` | LeRobot IO, schema, metadata, conversion | hardware polling |
| `retargeting/` | pose-to-target logic, axis maps | robot hardware replay |
| `robots/` | URDF names, IK specs, command layout, registry | capture logic |
| `replay/` | visualization/replay/deployment tools | raw recording |

## Raw Dataset Contract

The canonical HandUMI raw state is `float32[16]`:

```text
0   left_x
1   left_y
2   left_z
3   left_qx
4   left_qy
5   left_qz
6   left_qw
7   right_x
8   right_y
9   right_z
10  right_qx
11  right_qy
12  right_qz
13  right_qw
14  left_gripper_width
15  right_gripper_width
```

Gripper widths in the raw state are calibrated widths in meters. Feetech
auxiliary features also store raw ticks, normalized width, and width in mm for
hardware validation.

Core raw features:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.state
action
observation.feetech.left_ticks
observation.feetech.right_ticks
observation.feetech.left_width_mm
observation.feetech.right_width_mm
timestamp
frame_index
episode_index
task_index
```

`src/handumi/dataset/raw.py` is the code contract for the 16D layout. Use
`HANDUMI_RAW_STATE_NAMES` and `HANDUMI_RAW_STATE_SIZE` instead of repeating the
schema in scripts.

PICO/body/Feetech auxiliary signals can be additive features, but they should not
replace the compact raw state contract.

Feetech is used only as an encoder source for gripper aperture. It is not a
robot-arm dependency.

## Configs

```text
configs/handumi.yaml              # top-level recording config
configs/cameras.yaml              # left_wrist/right_wrist assignment
configs/feetech.yaml              # left/right servo IDs and calibration ticks
configs/tracking_pico.yaml        # PICO backend settings
configs/tracking_meta_quest.yaml  # Meta Quest backend settings
```

Initial Feetech convention:

```text
servo ID 0 -> left HandUMI gripper
servo ID 1 -> right HandUMI gripper
```

Each gripper can have its own USB serial port, or both can share one Feetech
bus. `configs/feetech.yaml` stores the left/right port mapping, closed/open
encoder ticks, and `max_width_mm`.

## Robot Embodiments

Robot-specific behavior is loaded through:

```python
from handumi.robots.registry import load_embodiment

runtime = load_embodiment("piper")
solver = runtime.solver_cls(config=runtime.config_cls())
sim = runtime.make_sim()
```

Each robot package contributes configuration, not algorithms:

```text
src/handumi/robots/<name>/
|-- shared.py       # URDF names, command layout, unit conversion
|-- solver.py       # RobotKinematicsSpec + KinematicsSolver binding
`-- retargeting.py  # RetargetingSpec binding
```

Shared robot logic lives once:

```text
robots/kinematics.py     # BimanualPyrokiSolver
robots/sim.py            # ViserSim
robots/registry.py       # load_embodiment("piper" | "axol")
retargeting/pico_to_robot.py
retargeting/handumi_to_robot.py
```

Per-arm command vectors are `(8,)`:

```text
Piper: [j1, j2, j3, j4, j5, j6, unused, gripper]
Axol : [j1, j2, j3, j4, j5, j6, j7, gripper]
```

## Entry Points

Pipeline scripts:

```text
scripts/record_handumi.py              -> handumi.capture.record_handumi
scripts/process_handumi_to_lerobot.py  -> handumi.dataset.conversion
scripts/replay_pico_ik.py              -> handumi.replay.pico_ik
scripts/compare_axis.py                -> handumi.retargeting.compare_axis
scripts/piper/replay_from_dataset.py   -> handumi.replay.piper
```

Hardware setup CLIs:

```text
handumi-find-servos
handumi-find-cameras
handumi-setup-servos
handumi-calibrate-grippers
handumi-teleoperate
handumi-record
```

`handumi-teleoperate` is the LeRobot-style live inspection loop. It does not
write a dataset; it streams cameras and Feetech aperture signals to Rerun so the
operator can validate hardware before `handumi-record`.

Shell launchers:

```text
bin/record.sh
bin/process_handumi_to_lerobot.sh
bin/piper/replay_from_dataset.sh
```

Automated tests live only under `tests/`.

## Invariants

- Raw recording remains robot-agnostic.
- Robot datasets are reproducible from raw/source data plus config.
- Scripts stay thin; reusable logic lives in `src/handumi`.
- New robots are added through `robots/<name>/` and the registry.
