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

```bash
handumi-record \
  --device meta \
  --robot piper \
  --controller-tcp-calibration configs/calibration/meta_controller_tcp.yaml \
  --session-calibration outputs/calibration/session.yaml \
  --wrist-cameras --workspace-camera \
  --clap-control \
  --num-episodes 20 \
  --episode-time-s 60
```

A double clap starts. Another double clap stops and saves the episode.
`--episode-time-s` is the maximum duration and saves automatically if reached.

## 6. Validate

```bash
handumi-validate --root outputs/datasets/handumi-demo --fail-on-reject
```

After the pilot passes, increase `--num-episodes` for production.
