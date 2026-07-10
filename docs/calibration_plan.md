# HandUMI calibration runbook

Status: **pending physical calibration**. The committed Meta TCP translation
has an 8.4 cm norm, while the current gripper is expected to place the tip
roughly 24 cm from the controller. Treat 24 cm only as a sanity check: the
pivot solve estimates the offset from the Quest controller tracking origin.

Keep this file as the calibration record. After calibration, add the date,
mount revision, results, and dataset paths.

## 1. Setup

- Rigidly mount the HMD on the stationary rig. Do not leave it hanging.
- Confirm both controllers remain `tracked=1` and `valid=1`.
- Fix a dimple or cradle to the table so the gripper tip cannot slide.
- Keep the mount unchanged throughout all trials.

## 2. Record pivot trials

Record three 25-second episodes per side. Keep the tip fixed while rotating
through roll, pitch, and yaw. Feetech is not required; cameras remain enabled
because the recorder currently has no `--skip-cameras` option.

```bash
handumi-record --device meta --skip-feetech \
  --repo-id local/tcp_pivot_left --output-dir outputs/datasets/tcp_pivot_left \
  --task "tcp pivot left" --num-episodes 3 --episode-time-s 25
```

Repeat for the right side. Validate both datasets before solving:

```bash
handumi-validate --repo-id local/tcp_pivot_left \
  --root outputs/datasets/tcp_pivot_left --fail-on-reject
```

## 3. Solve without touching production

Run each accepted episode into a temporary output and compare the three
translations. Example for left episode 0:

```bash
handumi-calibrate-tcp-offset pivot --device meta --side left \
  --parquet outputs/datasets/tcp_pivot_left/data/chunk-000/file-000.parquet \
  -e 0 --output outputs/calibration/meta_left_trial_0.yaml
```

Acceptance criteria per side:

- RMS residual **<= 2 mm** and maximum residual **<= 5 mm**.
- No weak-rotation-diversity warning.
- Translation estimates across the three trials agree within **2 mm**.
- Offset direction and scale are physically plausible; left/right symmetry is
  a sanity check, not a hard requirement.

Re-run the selected left and right trials into the same candidate file, then
inspect it. Do not overwrite the committed calibration until verification.

```bash
handumi-calibrate-tcp-offset inspect \
  outputs/calibration/meta_controller_tcp_candidate.yaml
```

## 4. Rotation and table frame

Do not estimate rotation from an orientation described only relative to the
table: the current recording workspace is initialized from the HMD. Keep the
existing rotation only if the controller mount orientation is unchanged.

Before production data collection, implement the YUBI-style Quest-to-table
extrinsic calibration:

- Place a ChArUco board at a repeatable table pose.
- Observe it with the rigid wrist cameras and fixed workspace camera.
- Estimate and save `T_table_quest` with board geometry, reprojection error,
  timestamp, and rig identifier.
- Express recorded controller/TCP poses in that table frame.

This software path is **not implemented yet**. Rotation calibration against
the table must wait for it or use a rigid orientation fixture whose pose is
known in the current Quest workspace.

## 5. Verify and release

```bash
handumi-live --device meta --anchor-z 0.0
```

- Hold the tip in one pivot and rotate the wrist: simulated TCP motion should
  remain within 2 mm.
- After one anchor, touch several table points using different wrist
  orientations; simulated height should remain within 2 mm of the table.
- Record and replay a pick-and-place episode and confirm the same behavior.

After passing, replace `configs/calibration/meta_controller_tcp.yaml`, run
`handumi-calibrate-tcp-offset inspect --device meta`, commit the calibration,
and record the measured RMS/max/condition values below.

## Calibration record

- Date:
- Mount revision:
- Left offset / RMS / max / condition:
- Right offset / RMS / max / condition:
- Quest-to-table calibration:
- Source datasets:
