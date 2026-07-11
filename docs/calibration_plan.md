# HandUMI portable calibration runbook

Status: **not ready for production collection**. TCP pivot calibration and the
Quest-to-table ChArUco path must pass the checks below first.

The first collection mode is portable: secure the Quest rigidly to the chest.
Do not leave it hanging freely from the neck. Moving the HMD is valid because
Quest tracks the HMD and controllers in one world frame; the table origin must
come from ChArUco, not from the current per-episode HMD recenter.

## 1. Print the table reference

Use one canonical board for every station and session:

- ChArUco: **7 x 5 squares**.
- Dictionary: **OpenCV `DICT_5X5_1000`**.
- Square length: **30.00 mm**.
- ArUco marker length: **22.00 mm**.
- Page: A4 landscape, black and white, with at least 20 mm white margin.
- Print at **100% / Actual size**. Disable Fit, Shrink and borderless scaling.

Do not use an arbitrary ChArUco image from the internet. Generate and keep a
versioned PDF with these exact parameters. After printing, measure at least
three squares in X and Y with calipers. Their mean must be 30.00 mm within
0.20 mm. Reject and reprint if the scale differs or X/Y scale is unequal.

Glue the sheet flat to a rigid matte plate; avoid wrinkles, glossy lamination
and bent foam board. Mark the operator-facing edge. The HandUMI table frame is:

- origin: center of the printed board on its top surface;
- +X: operator's right;
- +Y: away from the operator;
- +Z: upward from the table.

Place the board flat at a repeatable location near the center of the workspace.
It may be removed after session calibration without moving the table or Quest
tracking environment.

## 2. Calibrate controller to TCP once per mount

Rigidly attach each Quest controller and wrist camera to its HandUMI. Fix a
dimple or cradle to the table so the gripper tip cannot slide. Record three
25-second pivot episodes per side while rotating through roll, pitch and yaw:

```bash
handumi-record --device meta --skip-feetech \
  --repo-id local/tcp_pivot_left --output-dir outputs/datasets/tcp_pivot_left \
  --task "tcp pivot left" --num-episodes 3 --episode-time-s 25
```

Repeat for the right side, validate, and solve each accepted episode:

```bash
handumi-validate --repo-id local/tcp_pivot_left \
  --root outputs/datasets/tcp_pivot_left --fail-on-reject

handumi-calibrate-tcp-offset pivot --device meta --side left \
  --parquet outputs/datasets/tcp_pivot_left/data/chunk-000/file-000.parquet \
  -e 0 --output outputs/calibration/meta_left_trial_0.yaml
```

Acceptance per side:

- RMS residual <= 2 mm and maximum residual <= 5 mm.
- No weak-rotation-diversity warning.
- The three translations agree within 2 mm.
- Offset direction and scale are physically plausible.

Only after verification replace
`configs/calibration/meta_controller_tcp.yaml`.

## 3. Calibrate controller to wrist camera once per mount

This extrinsic is required because the wrist camera observes ChArUco while
Quest reports the controller pose. It is separate from controller-to-TCP.

For each side, hold the board fixed and capture at least 20 synchronized views
covering different distances and roll/pitch/yaw angles. Each view must contain
at least 12 detected ChArUco corners. Solve the rigid
`T_controller_wrist_camera` transform from controller poses and board poses.

Save one calibration per side with board parameters, camera intrinsics,
reprojection error, timestamp and mount revision. Recalibrate after moving a
controller or wrist camera mount.

Acceptance:

- Mean reprojection error <= 0.5 px; maximum <= 1.5 px.
- Reconstructed fixed-board positions agree within 2 mm RMS.
- Left- and right-camera estimates of the same board agree within 3 mm and 1 deg.

**This solver and its persisted calibration are not implemented yet.**

## 4. Set the table origin at the start of each session

1. Wear the Quest rigidly on the chest and start YubiQuestApp.
2. Place the canonical board flat in its marked table location.
3. Observe it from 8-12 varied poses with either wrist camera; using both is a
   stronger cross-check.
4. For each view compute:

   `T_quest_board = T_quest_controller * T_controller_camera * T_camera_board`

5. Reject outliers and average the accepted transforms.
6. Convert the board frame to the defined table frame and save
   `T_table_quest` with residuals, board ID, rig/session ID and timestamp.
7. Touch three known points and the table plane with both TCPs. Require <= 2 mm
   height error and <= 3 mm 3D disagreement before recording.

Quest may move with the operator after this calibration. Recalibrate the
session if Quest loses/relocalizes tracking, the chest mount slips, the table
moves, or validation fails.

The recorder must use the saved `T_table_quest` for every episode in the
session and must not replace it with the current HMD-based reset when the start
gesture is detected. Raw controller and HMD poses remain stored unchanged;
table-frame TCP poses are derived reproducibly from the recorded transforms.

**This session-calibration and recorder path are not implemented yet.**

## 5. Pilot before production

After Sections 2-4 are implemented and accepted:

1. Record 10 bimanual episodes using double-clap start/stop.
2. Run `handumi-validate --fail-on-reject`.
3. Replay and convert with the recorded calibration metadata.
4. Verify both wrist videos, Feetech widths, tracking health and table-frame
   TCP trajectories.
5. Start production only if all 10 episodes pass without manual correction.

## Calibration record

- Date / operator:
- Quest and mount revision:
- Printed board ID / measured square size X/Y:
- Camera intrinsics:
- Left TCP offset / RMS / max:
- Right TCP offset / RMS / max:
- Left/right controller-to-camera residuals:
- Quest-to-table residual / verification points:
- Source datasets and software commit:
