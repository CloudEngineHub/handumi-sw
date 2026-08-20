# HandUMI Setup and Calibration

Complete this page before recording. No robot arm is required: these steps
configure HandUMI, its tracking device, cameras, grippers, and workspace.
Some calibrations are permanent for one physical assembly; the table/session
alignment must be checked each session.

| Calibration | Repeat when |
| --- | --- |
| Servo homing and opening width | Servo, linkage, or gripper geometry changes |
| Camera intrinsics | Camera, resolution, or focus changes |
| Controller-to-camera mount | A controller or wrist camera mount moves |
| Controller-to-TCP | The controller/gripper mount or physical tool changes |
| Table/session frame | Each session, relocalization, or tracking reset |

## 1. Map HandUMI Hardware

`install.sh` creates the ignored machine-local `configs/rig.yaml`. Inspect the
connected cameras and Feetech adapters:

```bash
handumi setup ports
```

Reconnect one physical device at a time and assign its port under `cameras`
or `feetech` in `configs/rig.yaml`. Robot-arm buses do not belong in this
recording setup; configure them only for real-robot teleoperation.

Set new Feetech IDs only when required:

```bash
handumi servo set-id --port /dev/ttyUSB0 --new-id 0
handumi servo set-id --port /dev/ttyUSB0 --new-id 1
```

:::{dropdown} Hardware mapping details
Two grippers may share one serial `port` only when they use different
`servo_id` values. With separate USB adapters, each side normally has its own
port.

A USB camera commonly exposes two `/dev/video*` nodes. Start with the first
node reported for each physical camera and confirm the stream. Map
`left_wrist`, `right_wrist`, and `workspace` explicitly in `configs/rig.yaml`.

Keep these machine-local paths in `configs/rig.yaml`; do not commit them as
portable project configuration.
:::

### Camera types and resolutions

Declare capture settings for the three logical views: `left_wrist`,
`right_wrist`, and `workspace`. Each view can use the normal `opencv` backend
or `zedmini`, which expects the ZED Mini side-by-side UVC mode and exposes only
its left image:

```yaml
cameras:
  workspace:
    type: zedmini
    index_or_path: 4
    width: 1344
    height: 376
    fps: 30
```

For `zedmini`, `width` and `height` describe the captured stereo frame.
HandUMI keeps `frame[:, :672]`, so previews and datasets contain one
`672×376` RGB image. The supported rates for this mode are 15, 30, 60, and
100 FPS. `index_or_path` accepts either an OpenCV integer such as `4` or an
explicit Linux path such as `/dev/video4`. Per-camera values in `cameras:`
take precedence over the global camera fallbacks used by older rig files.

## 2. Calibrate the Grippers

First confirm that both encoders change smoothly while opening and closing:

```bash
handumi calibrate grippers monitor
```

Home each servo with the gripper held at **mid-travel**. This centers the
encoder range and avoids crossing the 0/4095 wrap point:

```bash
handumi servo home
handumi servo home --side right  # one side only
```

Then calibrate the physical opening width:

```bash
handumi calibrate grippers calibrate
handumi calibrate grippers calibrate --side right
```

For each side, enter the maximum opening in millimeters, place the gripper fully
open and press Enter, then fully close it and press Enter. The result is stored
in `~/.cache/handumi/calibration.yaml`. Open and close each gripper again with
`monitor` and confirm that width increases toward fully open without flipping
or saturating.

## 3. Connect Tracking

### Meta Quest

Enable Developer Mode, connect the headset over USB, authorize `adb`, and
install [HandUMI Quest App](https://github.com/murobotics-ai/handumi-quest-app/releases):

```bash
wget https://github.com/murobotics-ai/handumi-quest-app/releases/download/v0.2.1/handumi-quest-app-v0.2.1.apk
adb install -r handumi-quest-app-v0.2.1.apk
adb shell ip route  # find the address after "src"
```

Set that address as `meta_quest.connection.quest_ip` in `configs/rig.yaml`.
Launch the app from Library → Unknown Sources and keep it in the foreground.

```bash
python -m handumi.tracking.meta_quest --config configs/rig.yaml
```

A healthy stream reports steady FPS and both controllers tracked.

### PICO

Install the [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)
and follow the current [XR Robotics headset instructions](https://github.com/XR-Robotics).
Start the PC service, then launch streaming:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

Use `127.0.0.1:63901` for USB or the workstation IP with `--pico-wifi`.

Smoke-test a short capture before calibration:

```bash
handumi record --output-dir outputs/pico-smoke \
  --device pico --skip-feetech --no-voice-control \
  --task "pico smoke" --episodes 1 --episode-time-s 10
```

Healthy output reports `xrobotoolkit_sdk initialised` without repeated
`still waiting for PICO data` messages.

## 4. Calibrate Cameras and Workspace

Fix the 5 × 7 ChArUco board flat at its marked table position, with IDs 15 and
16 nearest the operator. Its center defines the table origin: +X right, +Y away,
and +Z up.

### Camera Intrinsics

```bash
handumi calibrate spatial intrinsics --camera left_wrist
handumi calibrate spatial intrinsics --camera right_wrist
handumi calibrate spatial intrinsics --camera workspace
```

Move the board throughout each image and vary distance and inclination. The
tool automatically accepts a distinct valid view every two seconds. Repeat
after changing camera, resolution, or focus.

### Controller-to-Camera Mounts

Keep the board fixed. Move the complete HandUMI through varied roll, pitch, and
yaw poses, pausing briefly for each automatic capture. Keep the controller
tracking ring visible to the headset.

Choose the tracking device explicitly. Global options such as `--device`,
`--pico-wifi`, and `--quest-ip` come before the subcommand.

Meta Quest:

```bash
handumi calibrate spatial --device meta mount --side left
handumi calibrate spatial --device meta mount --side right
```

PICO:

```bash
handumi calibrate spatial --device pico --pico-mode mandos mount --side left
handumi calibrate spatial --device pico --pico-mode mandos mount --side right
```

PICO calibration relies on live XRoboToolkit snapshots, so hold the HandUMI
steady while each view is accepted. Use `--pico-wifi` for a wireless PICO setup.

Repeat only if a controller or wrist-camera mount moves.

### Session/Table Frame

With the board still at its marked position and the headset fixed as it will be
during recording, solve the table frame for the same tracking device.

```bash
handumi calibrate spatial --device meta session --side left
handumi calibrate spatial --device meta visualize
```

For PICO:

```bash
handumi calibrate spatial --device pico --pico-mode mandos session --side left
handumi calibrate spatial --device pico --pico-mode mandos visualize
```

By default, `session` uses only the current capture. To retry with accumulated
views, opt in to pools under `outputs/calibration/accumulation_N/`.

:::{dropdown} Session view accumulation
- **`--start-accumulation N`:** start or reset lot `N`.
- **`--accumulation N`:** continue lot `N`; use this to switch between lots
  (for example different lighting) without resetting them.
:::

Inspect all cameras and both TCP trails in Rerun. The table surface must align
with `z=0`. If only the workspace-camera stage fails, retry it with:

```bash
handumi calibrate spatial workspace
```

Remove the board without moving the table, cameras, or headset. Repeat the
session calibration after relocalization or a tracking reset. The saved
`outputs/calibration/session.yaml` records `tracking_device` and
`table_from_device`; use it only with the same `--device`.

## 5. Calibrate the HandUMI Tool Tip

Controller-to-TCP reconstructs the physical tool-tip pose from each tracked
controller. It belongs to the **tool assembly** -- the gripper tip screwed onto
the HandUMI shells, plus the controller mount -- and not to the robot the data
is later retargeted to. A different tip needs its own calibration even on the
same robot, and one tip serves every robot it is used with.

Each side is captured the same way: **wedge the tip into a firm indentation so
it cannot slide, then rotate the rest of the assembly around it for 25 seconds,
through as many different orientations as the mount allows.** The tip staying
put is what makes the fit correct; the variety of orientations is what makes it
well-conditioned.

### Step 1. Capture and fit the left side

```bash
LEFT=outputs/tcp_pivot_left
handumi record --output-dir $LEFT --skip-feetech --no-voice-control \
  --cameras left_wrist --task "tcp pivot left" \
  --episodes 1 --episode-time-s 25 --tracking-loss-timeout-s 3 --no-sounds

handumi calibrate tcp pivot --side left --dataset $LEFT
```

`--dataset` resolves the recording's parquet and episode, and the fit is
written to `outputs/calibration/controller_tcp_candidate.yaml`. The tracking
device comes from `recording.device` in `configs/rig.yaml`; pass `--device` to
override it.

Your hands are busy holding the tool during a pivot capture, so
`--no-voice-control` keeps the episode on the plain ENTER-then-timer flow
instead of waiting to be spoken to.

### Step 2. Capture and fit the right side

The same two commands with `right` in place of `left`:

```bash
RIGHT=outputs/tcp_pivot_right
handumi record --output-dir $RIGHT --skip-feetech --no-voice-control \
  --cameras right_wrist --task "tcp pivot right" \
  --episodes 1 --episode-time-s 25 --tracking-loss-timeout-s 3 --no-sounds

handumi calibrate tcp pivot --side right --dataset $RIGHT
```

Both sides write into the same candidate file; nothing is applied to the
project yet.

### Step 3. Check the fit

```bash
handumi calibrate tcp inspect
```

| Metric | Accept | If it fails |
|---|---|---|
| RMS | below 0.50 cm | The tip slipped. Find a deeper indentation and recapture. |
| Maximum error | below 1.00 cm | As above; check for a moment of lost tracking. |
| Condition | below 500 | The capture lacked rotational variety. Recapture covering more orientations. |

Recapture that side until it passes. Do not promote a fit that does not.

Then compare the two sides. The mounts are mirror twins, so `x` and `z` should
agree between them and only `y` flips sign. A mismatch of several millimeters
means one of the captures drifted, not that the tool is asymmetric.

### Step 4. Promote it into the project

Pivot fitting solves translation only, so keep the official quaternions and
symmetrize just the measured positions:

```text
x = (left.x + right.x) / 2
y = (left.y - right.y) / 2
z = (left.z + right.z) / 2
left.position  = [x,  y, z]
right.position = [x, -y, z]
```

Update only those two `position` values in the calibration file for this tool
assembly. Those files live in `configs/calibration/controller_tcp/` as
`{device}_{tool}.yaml`, and each robot points at its own under
`controller_tcp_calibrations` in `configs/robots/<robot>.yaml`:

```yaml
handumi_tool:
  gripper: ARX5_beta          # the tip physically screwed onto HandUMI
  controller_mount: handumi_v1
controller_tcp_calibrations:
  meta: configs/calibration/controller_tcp/meta_ARX5_beta.yaml
```

Fitting a **new** tip means writing a new file and pointing the robot at it,
rather than overwriting the previous tip's calibration. Existing datasets keep
their own recorded assembly identity, so old recordings stay reproducible.

### Step 5. Verify

```bash
uv run pytest -q tests/tracking/test_transforms.py \
  tests/scripts/test_replay_in_sim.py
```

These check the mirror invariant and that the file is selected for the robot.
One of them also bounds the tip-to-controller distance; that bound is a
property of the tips in use, so a genuinely new tip legitimately widens it.

Then confirm it physically: touch one point with both tips and check that their
calibrated positions coincide. After session calibration, touching the table
should place both tips near `z=0`.

Next: [Record Demonstrations](record.md).
