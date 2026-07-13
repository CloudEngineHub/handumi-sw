# Calibrate and record

## 1. Place the ChArUco board

Fix the 5 x 7 board flat and vertical as printed, with IDs 15 and 16 closest
to the operator. Its center is the table origin: +X right, +Y away, +Z up.

## 2. Calibrate cameras once

The tool automatically accepts a valid, distinct view every two seconds.
Move the board across the image and vary distance and inclination.
No assistant or keyboard input is required; `Q` only cancels the process.

```bash
handumi-calibrate-spatial intrinsics --camera left_wrist
handumi-calibrate-spatial intrinsics --camera right_wrist
handumi-calibrate-spatial intrinsics --camera workspace
```

Repeat after changing camera, resolution or focus.

## 3. Calibrate controller-camera mounts

Keep the board fixed. Move each complete HandUMI using distinct roll, pitch and
yaw orientations. This finds the rigid controller-to-wrist-camera transform.
Hold it still briefly before each automatic capture and keep its tracking ring
visible to the Quest cameras.

```bash
handumi-calibrate-spatial mount --side left
handumi-calibrate-spatial mount --side right
```

Repeat after moving a controller or wrist-camera mount.

## 4. Calibrate each session

Fix the Quest to the chest and keep the board at its marked position. Use any
side with a saved mount; this calibrates the table position, orientation and
height relative to Quest. Verify with the other calibrated side when available.

```bash
handumi-calibrate-spatial session --side left
handumi-calibrate-spatial visualize
```

If only the workspace stage fails, retry it without repeating the HandUMI:

```bash
handumi-calibrate-spatial workspace
```

Inspect the final alignment in Rerun. Then remove the board without moving the
table, cameras or Quest. Repeat after relocalization or tracking reset.

## 5. Record a pilot

Authenticate once and confirm which Hugging Face account is active:

```bash
hf auth login
hf auth whoami
```

```bash
handumi-record \
  --device meta \
  --robot piper \
  --task "pick and place the blue cube in the white box" \
  --repo-id NONHUMAN-RESEARCH/test_handumi_pickandplace \
  --controller-tcp-calibration configs/calibration/meta_controller_tcp.yaml \
  --session-calibration outputs/calibration/session.yaml \
  --wrist-cameras --workspace-camera \
  --clap-control \
  --num-episodes 20 \
  --episode-time-s 60 \
  --push-to-hub
```

`--repo-id` is `account-or-organization/dataset-name`. The authenticated token
must have write access to that namespace. Without `--push-to-hub`, the validated
dataset remains local under `outputs/`.

A right double clap starts or stops/saves the episode. A left double clap while
recording discards the current attempt and immediately restarts the same episode.
`--episode-time-s` is the maximum duration and saves automatically if reached.
`Esc` or `Ctrl+C` discards any active partial episode before stopping. `Esc` is
available with `--clap-control` in an interactive terminal.

Finalization validates LeRobot v3 Parquet, episode metadata, frame counts and
videos, then writes a local Hugging Face `README.md` dataset card. Use
`--dataset-license <id>` to set its data license (`other` by default), and
`--push-to-hub` only uploads after validation passes.

## 6. Validate

```bash
handumi-validate --root outputs/datasets/handumi-demo --fail-on-reject
```

After the pilot passes, increase `--num-episodes` for production.
