# Adding a New Embodiment

Ultima modificacion: 2026-06-26 16:59:21 -05 -0500

This guide describes what you need to add a new bimanual robot (e.g. `"franka"`) to handumi so it works with PICO retargeting, IK, and Viser visualization.

See [architecture.md](architecture.md) for how the pieces fit together.

---

## Overview

Adding an embodiment means supplying **configuration**, not reimplementing algorithms. You will create:

```
src/handumi/robots/<name>/
├── shared.py       # URDF naming, command layout, command_to_arm_q
├── solver.py       # RobotKinematicsSpec + collision builder
└── retargeting.py  # RetargetingSpec + PicoTo<Name>ArmRetargeter

assets/<name>/      # URDF + meshes
```

Then register the embodiment in `src/handumi/robots/registry.py`.

---

## Checklist

- [ ] URDF with bimanual left/right naming convention
- [ ] `shared.py` with joint enum, URDF helpers, `command_to_arm_q`
- [ ] `solver.py` with `RobotKinematicsSpec` and collision builder
- [ ] `retargeting.py` with rest poses and `RetargetingSpec`
- [ ] Entry in `registry.py` (`EMBODIMENT_NAMES`, `load_embodiment`, axis-map defaults)
- [ ] Smoke test: import, IK solve, optional Viser replay

---

## Step 1 — URDF assets

Place the robot description under `assets/<name>/`:

```
assets/myrobot/
├── myrobot.urdf
└── meshes/          # if referenced by the URDF
```

Requirements:

- **Bimanual structure** with distinguishable left and right arms (consistent prefix, e.g. `left_`, `right_`).
- **Actuated joints** named explicitly in the URDF (revolute and/or prismatic).
- **End-effector link** identifiable for IK (tool center point or gripper link).
- Meshes resolve relative to the URDF directory (same pattern as Piper/Axol).

Verify the URDF loads:

```python
import yourdfpy
urdf = yourdfpy.URDF.load("assets/myrobot/myrobot.urdf", mesh_dir="assets/myrobot")
print([j.name for j in urdf.robot.joints if j.joint_type != "fixed"])
```

---

## Step 2 — `shared.py`

Create `src/handumi/robots/myrobot/shared.py`. This is the **single source of truth** for URDF strings.

### Required contents

1. **`Joint` enum** — logical joints on one arm, in control order.
2. **`ARM_JOINTS`** — joints solved by IK (revolute only).
3. **`URDF_PATH`** — resolved via `_resolve_urdf_path()` (follow Piper/Axol pattern).
4. **Command layout constants:**
   - `ARM_JOINT_COUNT`
   - `COMMAND_SIZE` (typically 8: arm joints + optional padding + gripper slot)
   - `GRIPPER_INDEX` (if applicable)
5. **Naming helpers** (at minimum):
   - `urdf_joint_name(joint, *, is_left: bool) -> str`
   - `urdf_body_name(joint, *, is_left: bool) -> str`
   - `urdf_arm_joint_names(*, is_left: bool) -> list[str]` — all actuated URDF joints per arm, in Viser order
   - `urdf_revolute_joint_names(*, is_left: bool) -> list[str]` — IK joints only (if different from full list)
6. **`command_to_arm_q(command: np.ndarray) -> np.ndarray`**

   Maps one per-arm command vector to the URDF actuated-joint sub-vector for that arm.

   Examples from existing embodiments:

   - **Piper:** 6 revolute values + convert gripper scalar to 2 finger positions.
   - **Axol:** 7 revolute values; ignore gripper index (no actuated gripper in URDF).

### Rules

- Do **not** hard-code URDF strings outside `shared.py`.
- If the gripper is normalized to `[0, 1]`, document the mapping to physical joint values here.

---

## Step 3 — `solver.py`

Create `src/handumi/robots/myrobot/solver.py`.

### Required contents

1. **`_build_robot_collision(urdf, robot, config)`** (recommended)

   Restrict pyroki self-collision pairs to contacts that matter for your robot geometry. Start from Piper or Axol as a template and adjust link-name predicates.

2. **`MYROBOT_KINEMATICS_SPEC = RobotKinematicsSpec(...)`**

   | Field | What to set |
   |-------|-------------|
   | `name` | Human-readable name |
   | `urdf_path` | From `shared.URDF_PATH` |
   | `left_ee_link`, `right_ee_link` | URDF link names at the tool/gripper |
   | `left_shoulder_link`, `right_shoulder_link` | Base of each arm chain |
   | `left_arm_joint_names`, `right_arm_joint_names` | Joints solved by IK (tuple of URDF joint names) |
   | `left_control_joint_names`, `right_control_joint_names` | Optional; all joints in output `q` if different from IK joints (see Piper fingers) |
   | `left_elbow_link`, `right_elbow_link` | Optional; for diagnostics/visualization |
   | `collision_builder` | Your `_build_robot_collision` function |

3. **`KinematicsSolver = make_kinematics_solver(MYROBOT_KINEMATICS_SPEC)`**

### Sanity check

```python
from handumi.robots.myrobot.solver import KinematicsSolver
from handumi.robots.kinematics import KinematicsConfig

solver = KinematicsSolver(config=KinematicsConfig())
q = solver.ik(
    q_current=solver.robot.joints.zeros(),
    left_pose=(solver._left_shoulder_pos + [0.3, 0, 0.2], eye(3)),
    right_pose=(solver._right_shoulder_pos + [0.3, 0, 0.2], eye(3)),
)
print(q.shape, solver.num_joints)
```

---

## Step 4 — `retargeting.py`

Create `src/handumi/robots/myrobot/retargeting.py`.

### Required contents

1. **`REST_LEFT_ARM`, `REST_RIGHT_ARM`** — numpy arrays, one element per IK arm joint. Used as posture prior and for first-frame calibration.

   Choose a comfortable, collision-free pose. Piper uses zeros; Axol uses a slight elbow bend.

2. **`_left_front_wrist(forward, lateral, height)`** and **`_right_front_wrist(...)`**

   Map workspace offsets (meters) to robot-base-frame wrist positions for the `--workspace front` mode. Mirror left/right lateral sign on the right arm (see existing implementations).

3. **`MYROBOT_RETARGETING_SPEC = RetargetingSpec(...)`**

   ```python
   RetargetingSpec(
       name="myrobot",
       rest_left_arm=REST_LEFT_ARM,
       rest_right_arm=REST_RIGHT_ARM,
       command_size=8,           # must match shared.COMMAND_SIZE
       gripper_index=7,          # must match shared.GRIPPER_INDEX
       left_front_wrist=_left_front_wrist,
       right_front_wrist=_right_front_wrist,
   )
   ```

4. **`PicoToMyrobotArmRetargeter(PicoToRobotArmRetargeter)`** — thin subclass passing `spec=MYROBOT_RETARGETING_SPEC` to `super().__init__(...)`.

5. **Re-export** (optional but conventional):

   ```python
   from handumi.retargeting.pico_to_robot import (
       move_retargeter_to_front_workspace,
       settle_first_frame,
       robot_link_positions,
   )
   ```

---

## Step 5 — Register in `registry.py`

Edit `src/handumi/robots/registry.py`:

1. Add `"myrobot"` to `EMBODIMENT_NAMES`.
2. Add axis-map candidates to `DEFAULT_COMPARE_AXIS_MAPS` (start with 8 sign permutations of `"x,z,y"` or copy from a similar robot and tune with `scripts/compare_axis.py`).
3. Add a branch in `load_embodiment()`:

```python
if name == "myrobot":
    from handumi.robots.myrobot.retargeting import (
        PicoToMyrobotArmRetargeter,
        move_retargeter_to_front_workspace,
        settle_first_frame,
    )
    from handumi.robots.myrobot.shared import (
        COMMAND_SIZE,
        URDF_PATH,
        command_to_arm_q,
        urdf_arm_joint_names,
    )
    from handumi.robots.myrobot.solver import KinematicsSolver

    return EmbodimentRuntime(
        name="myrobot",
        config_cls=KinematicsConfig,
        solver_cls=KinematicsSolver,
        retargeter_cls=PicoToMyrobotArmRetargeter,
        move_to_front_workspace=move_retargeter_to_front_workspace,
        settle_first_frame=settle_first_frame,
        urdf_path=URDF_PATH,
        urdf_arm_joint_names=urdf_arm_joint_names,
        command_size=COMMAND_SIZE,
        command_to_arm_q=command_to_arm_q,
        default_port=8004,              # pick an unused port
        default_axis_map="x,z,y",       # tune with compare_axis
        default_compare_axis_maps=DEFAULT_COMPARE_AXIS_MAPS["myrobot"],
        default_workspace="rest",
        wrist_forward=0.30,               # tune for your reach
        wrist_height=0.25,
        wrist_lateral=0.20,
    )
```

### Optional: `__init__.py`

For backward-compatible imports:

```python
# src/handumi/robots/myrobot/__init__.py
from handumi.robots.kinematics import KinematicsConfig
from .solver import KinematicsSolver

def Sim(**kwargs):
    from handumi.robots.registry import load_embodiment
    return load_embodiment("myrobot").make_sim(**kwargs)

__all__ = ["KinematicsConfig", "KinematicsSolver", "Sim"]
```

---

## Step 6 — Test

### Import and sim factory

```python
from handumi.robots.registry import load_embodiment
import numpy as np

runtime = load_embodiment("myrobot")
sim = runtime.make_sim()
left = np.zeros(runtime.command_size, dtype=np.float32)
q_arm = runtime.command_to_arm_q(left)
print("arm q length:", len(q_arm))
```

### PICO replay (requires a dataset)

```bash
uv run python scripts/replay_pico_ik.py --embodiment myrobot --episode 0 --visualize
uv run python scripts/compare_axis.py --embodiment myrobot --episode 0
```

Update `--embodiment` choices in test argument parsers if they use a hard-coded `choices=(...)` tuple.

---

## Tuning guide

After the embodiment loads, you will likely need to tune:

| Parameter | Where | Purpose |
|-----------|-------|---------|
| `REST_LEFT_ARM` / `REST_RIGHT_ARM` | `retargeting.py` | Stable rest posture, IK prior |
| `default_axis_map` | `registry.py` | PICO → robot frame alignment |
| `default_compare_axis_maps` | `registry.py` | Candidates for `compare_axis.py` |
| `wrist_forward/height/lateral` | `registry.py` | Front workspace placement |
| `KinematicsConfig` weights | passed at runtime | IK tracking vs smoothness |
| `_build_robot_collision` | `solver.py` | Self-collision behavior |

Use `scripts/compare_axis.py` to find a good axis map before running full replay.

---

## What you do **not** need to copy

Do **not** duplicate these modules per robot:

| Module | Reason |
|--------|--------|
| `kinematics.py` | IK algorithm is shared; only `RobotKinematicsSpec` changes |
| `pico_to_robot.py` | Retargeting algorithm is shared; only `RetargetingSpec` changes |
| `sim.py` | Viser server is shared; only `command_to_arm_q` changes |
| `registry.py` logic | Only add a new entry; `make_sim()` is generic |

---

## Minimal file template

Use Piper as the reference implementation when in doubt:

- `src/handumi/robots/piper/shared.py`
- `src/handumi/robots/piper/solver.py`
- `src/handumi/robots/piper/retargeting.py`
- `src/handumi/robots/registry.py` (piper branch)

Copy the Piper package, rename symbols, adjust joint counts and URDF mappings, then iterate with Viser and axis-map comparison until motion looks correct.
