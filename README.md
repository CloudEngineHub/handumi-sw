# HandUMI Software

Record HandUMI bimanual raw demonstrations as LeRobot-compatible datasets.

```text
left/right wrist cameras
+ left/right Feetech gripper encoder widths
+ optional VR tracking poses
-> HandUMI raw LeRobot dataset
```

## Install

```bash
git clone <repo-url> handumi-sw
cd handumi-sw
uv sync --python "$(command -v python3.12)"
source .venv/bin/activate
```

Check:

```bash
python --version
PYTHONPATH=src python scripts/record_handumi_pico.py --help
```

## Hardware Setup

### 1. Identify Ports

```bash
PYTHONPATH=src python scripts/setup/setup_ports.py
```

Connect/disconnect one device at a time and note the changed port.
Use `Ctrl+C` to stop.

The Feetech section shows each serial port and detected servo IDs:

```text
/dev/ttyACM0: ids=[0]
/dev/ttyACM1: ids=[1]
```

Edit `configs/feetech.yaml`:

```yaml
left:
  servo_id: 0
  port: /dev/ttyACM0
right:
  servo_id: 1
  port: /dev/ttyACM1
```

Edit `configs/cameras.yaml`:

```yaml
left_wrist:
  index_or_path: 0
right_wrist:
  index_or_path: 2
```

### 2. Check Feetech Ticks

```bash
PYTHONPATH=src python scripts/setup/calibrate_grippers.py monitor
```

Open/close each gripper and confirm `ticks` changes.

### 3. Home Servos (centre the encoder range)

The Feetech encoder reports position modulo 4096 and wraps at the 0/4095 seam.
If a gripper's travel crosses that seam, the width readout flips or saturates.
Homing stores a correction so the current shaft angle reads 2048 (centre):

```bash
PYTHONPATH=src python scripts/setup/home_servos.py              # both sides
PYTHONPATH=src python scripts/setup/home_servos.py --side right # one side
```

Hold the gripper at **mid-travel** (half open, ~2040 ticks) and press ENTER so
the full range sits clear of the seam. The script reads the position back and
reports `OK` / `CHECK`. Always re-calibrate afterwards (closed/open shift).

A software unwrap in `handumi.feetech.gripper` also tracks wraps continuously,
so even an un-homed range is fine as long as recording **starts with the
grippers roughly closed** (away from the seam).

### 4. Calibrate Gripper Width

```bash
PYTHONPATH=src python scripts/setup/calibrate_grippers.py calibrate
PYTHONPATH=src python scripts/setup/calibrate_grippers.py calibrate --side right
```

For each side:

```text
enter max opening in mm
open gripper fully while watching live ticks, press ENTER
close gripper fully while watching live ticks, press ENTER
```

Use `--side left|right` to recalibrate one gripper without disturbing the other.
This updates `configs/feetech.yaml`.

### 5. Live Monitor

```bash
PYTHONPATH=src python -m handumi.capture.teleoperate_handumi \
  --feetech-config configs/feetech.yaml \
  --fps 30
```

Streams cameras and gripper widths to Rerun without saving data. Start with the
grippers closed so the encoder unwrap anchors correctly.

### 6. Record

There are two recorders, one per tracking source:

- `scripts/record_handumi_pico.py` — PICO / XRoboToolkit tracking.
- `scripts/record_handumi_quest.py` — Meta Quest tracking (Phase 2; see
  [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md)).

Both write the same 16D HandUMI raw state + Feetech width.

```bash
PYTHONPATH=src python scripts/record_handumi_pico.py \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

Or use the launcher (wraps the PICO recorder):

```bash
bash bin/record.sh \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20
```

For Meta Quest recording (`record_handumi_quest.py`), see
[Motion Tracking (Phase 2)](#motion-tracking-phase-2) below — it covers the
one-time Quest app install and the run commands.

## Inspect Dataset

```bash
lerobot-dataset-viz \
  --repo-id local/handumi_width_test \
  --root outputs/datasets/handumi_width_test \
  --episode-index 0
```

## Dataset Fields

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

`observation.state[14]` and `observation.state[15]` are left/right gripper width
in meters.

## Motion Tracking (Phase 2)

Phase 2 adds Meta Quest controller tracking. The Quest is **body-worn on the
neck as a tracking base** (no headset UI) with a controller mounted on each
gripper; the two gripper wrist cameras are the only cameras. Poses arrive over
**TCP/JSON + a UDP time-sync**, not WebXR. The workstation calibrates them with
unit-tested transforms, merges Feetech width into the 16D raw state, and renders
a **live 3D controller trajectory in Rerun**. The Viser robot follow-along is
deferred to Phase 2B. Details:
[docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md).

### The Quest app

We do **not** build our own headset app — we reuse the prebuilt **YubiQuestApp**
from [yubi-sw](https://github.com/airoa-org/yubi-sw), which streams OVR
controller/HMD poses in the exact TCP/JSON format this repo's receiver parses.
(Building a dedicated HandUMI app is a possible future step.)

### 1. Try it now without a Quest (mock)

The whole Python pipeline runs against a built-in mock that emulates the Quest
app. Two terminals:

```bash
# terminal 1 — fake Quest (TCP pose stream + UDP time-sync on localhost)
PYTHONPATH=src python -m handumi.tracking.mock_quest_sender

# terminal 2 — live tracking to Rerun (no cameras/Feetech needed)
PYTHONPATH=src python scripts/live_tracking.py \
  --quest-ip 127.0.0.1 --skip-cameras --skip-feetech
```

You should see fps, a non-zero clock offset, and the left/right controller
trajectories drawing in the Rerun 3D view.

### 2. Enable Developer Mode on the Quest (one-time)

Sideloading needs Developer Mode, which needs a (free) Meta developer
organization:

1. Create/join an org at <https://developers.meta.com/> (one-time, on the web).
2. In the **Meta Horizon** mobile app (paired with the headset):
   *Menu → Devices → your headset → Headset Settings → Developer Mode → on*.
3. Connect the headset to the laptop with a **USB-C** cable and, **inside the
   headset**, accept the *Allow USB debugging* prompt (check "Always allow").

### 3. Install the YubiQuestApp (one-time)

The Quest is Android; the app is sideloaded over USB with `adb` (the same `adb`
the PICO setup uses — install with `sudo apt install adb` if missing).

```bash
# a) download the prebuilt APK (or download it in a browser)
wget https://releases.dev.airoa.io/yubi/quest-app/yubi-quest-app-v0.1.0.apk

# b) confirm the headset is visible over USB
adb devices          # the Quest should appear as "<serial>   device"

# c) install (-r upgrades over an existing install, keeping its data)
adb install -r yubi-quest-app-v0.1.0.apk
```

The app appears in the Quest library under **Unknown Sources** as
**YubiQuestApp**. (GUI alternative: [SideQuest](https://sidequestvr.com/),
drag-and-drop the APK.)

### 4. Find the Quest IP and set it once

The USB cable is only for installing — streaming is over **Wi-Fi/LAN**, and the
laptop dials the Quest. Put both on the **same network**, then find the Quest IP:

- On the headset: *Settings → Wi-Fi → (your network) → details*, or
- `adb shell ip route` (prints the headset IP while USB is connected).

Set it once in `configs/tracking_meta_quest.yaml` so you don't pass `--quest-ip`
every time:

```yaml
connection:
  quest_ip: "192.168.1.42"   # ← your Quest IP
  tcp_port: 65432            # YubiQuestApp defaults — leave as-is
  sync_port: 42000
```

### 5. Smoke-test the connection

Put on the headset and **launch YubiQuestApp** (Library → Unknown Sources). Then,
on the laptop, run the receiver in print mode — the simplest "is data flowing?"
check (no Rerun, no cameras):

```bash
PYTHONPATH=src python -m handumi.tracking.meta_quest \
  --config configs/tracking_meta_quest.yaml
```

You should see a live line with rising `fps`, a non-zero `off=` (clock offset),
and left/right positions that change as you move the controllers. `Ctrl+C` to
stop. Once this works, the headset can go on the neck mount.

### 6. Live tracking and recording

```bash
# live visualization — Rerun 3D trajectory (uses quest_ip from the config)
PYTHONPATH=src python scripts/live_tracking.py

# record a dataset (16D state + observation.quest.* poses/clocks)
PYTHONPATH=src python scripts/record_handumi_quest.py \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_quest_test \
  --output-dir outputs/datasets/handumi_quest_test \
  --task "quest tracking test" --num-episodes 1 --episode-time-s 20
```

Move the controllers and their trajectories draw in the Rerun 3D view. Add
`--skip-cameras` / `--skip-feetech` to run without that hardware.

Controls (no headset UI — feedback is on the workstation): **left X** resets the
workspace on the current HMD pose (also auto-set on the first tracked frame);
**right A** starts/stops an episode when recording with `--button-control`.

### Troubleshooting

- **`adb` doesn't list the Quest** — replug the USB-C cable, re-accept *Allow USB
  debugging* in the headset, try `adb kill-server && adb devices`.
- **Receiver says connecting but no frames** — YubiQuestApp must be **running and
  in the foreground** on the headset; check `quest_ip` is the headset's current
  Wi-Fi IP (DHCP can change it) and that laptop + Quest share the network.
- **Connects but poses are frozen/zero** — the controllers must be on and visible
  to the headset cameras; `tracked`/`valid` go false when occluded.
- **Frames arrive but fields look wrong** — the APK is the source of truth for the
  wire format. Dump one raw sample and compare it against
  [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) →
  *TCP/JSON Payload*:

  ```bash
  PYTHONPATH=src python -m handumi.tracking.meta_quest \
    --config configs/tracking_meta_quest.yaml --print-raw
  ```

  If key names differ, adjust `handumi.tracking.meta_quest.parse_frame`.

## Docs

- [docs/architecture.md](docs/architecture.md)
- [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) — Meta Quest
  motion tracking (body-worn, no-UI), Rerun trajectory rendering, yubi-sw/axol-vr
  references
- [docs/add-new-embodiment.md](docs/add-new-embodiment.md)
