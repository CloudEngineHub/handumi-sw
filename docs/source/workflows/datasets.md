# Dataset Pipeline

Validation, storage, and training operate on the robot-agnostic HandUMI
recording. Selecting a robot is an optional downstream retargeting step.

## Validate

```bash
handumi-validate \
  --repo-id your-name/handumi-demo \
  --root outputs/datasets/handumi-demo
```

The report is written to `meta/handumi_quality.json`. Rejected episodes are excluded automatically during conversion.

## Upload

```bash
hf auth login
huggingface-cli upload your-name/handumi-demo \
  outputs/datasets/handumi-demo --repo-type dataset
```

## Dataset Contents

Raw datasets preserve the information needed to validate, recalibrate, or
retarget a capture:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.images.workspace
observation.state                  # controller poses + gripper widths
observation.feetech.*              # ticks, width, time, health
observation.tracking.*             # device poses, validity, aligned time
observation.sync.*                 # shared target and record times
observation.camera.<name>.*        # sample time and health
```

`observation.state[14:16]` stores left/right gripper widths in meters. Tool,
controller mount, calibration hashes, source enablement, and coordinate layout
are stored in metadata. Raw controller poses remain unchanged so the same
capture can be converted again for a different robot.

## Train

HandUMI produces LeRobot-compatible datasets:

```bash
lerobot-train \
  --dataset.repo_id=your-name/handumi-demo \
  --policy.type=act
```

## Optional: Retarget to a Robot

Choose the target only when robot joint trajectories or simulation are needed.
Piper is used below as a currently available example; future robot backends use
the same boundary.

### Convert to Robot Joints

```bash
TARGET_ROBOT=piper
handumi-convert \
  --repo-id your-name/handumi-demo \
  --embodiment "$TARGET_ROBOT"
```

Add `--push-to-hub` to upload the converted dataset.

### Replay in Simulation

```bash
handumi-replay-in-sim \
  --repo-id your-name/handumi-demo \
  --robot "$TARGET_ROBOT"
```

Table-calibrated datasets preserve the recorded bimanual geometry automatically. Use `--headless` in automated checks and `--strict-ik` to fail on excessive IK error.

:::{dropdown} Absolute-table replay and calibration precedence
For an explicit geometry-preserving replay:

```bash
handumi-replay-in-sim --repo-id your-name/handumi-demo \
  --retarget-mode absolute-table \
  --deployment-calibration configs/calibration/<robot>_table.yaml
```

`absolute-table` applies `robot_from_table` to both TCP trajectories, preserving
their bimanual separation. By default, replay aligns each tool orientation on
the first frame and preserves subsequent wrist rotations. Use
`--absolute-orientation table-absolute` only when the HandUMI and robot TCP
frames were externally calibrated.

Controller-to-TCP calibration is selected in this order:

1. Explicit `--controller-tcp-calibration`.
2. Identity-bound snapshot stored in the dataset.
3. Robot/device calibration from `configs/robots/*.yaml`.
4. Device fallback for legacy data.

Replay prints the calibration source and hash, TCP distances, minimum height,
bimanual separation, deployment transform, and IK errors. These transformations
produce a target-specific result without changing the original HandUMI data.
:::

Physical deployment is separate. Continue with [Deploy to a Robot](../teleoperation.md)
only when simulation or a real arm is required.
