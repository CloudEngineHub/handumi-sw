# Gripper Setup (Feetech + Cameras)

One-time **per-laptop** hardware setup before teleoperating or recording: serial
ports, servo homing, gripper-width calibration. Run commands live in
[README.md](README.md).

> **Where things are stored.** Two different concerns, two different homes:
> - **Ports** (`servo_id`/`port`, like camera `index_or_path`) are wiring, so
>   they're committed in `configs/feetech.yaml` — edit that file directly, same
>   as `configs/cameras.yaml`.
> - **Calibration** (open/closed ticks, max width mm) is a measured property of
>   the physical gripper, so it lives in a per-user cache — never committed —
>   at `~/.cache/handumi/calibration.yaml` (or `$XDG_CACHE_HOME/...`). Homing
>   itself is stored in the servo's EEPROM (persists across power cycles and
>   laptops).
>
> Every setup tool prints the path(s) it's using as its first line(s) of output.

## 1. Identify Ports

```bash
handumi-setup-ports
```

Connect/disconnect one device at a time and note the changed port. `Ctrl+C` to
stop. The Feetech section shows each serial port and detected servo IDs:

```text
/dev/ttyACM0: ids=[0]
/dev/ttyACM1: ids=[1]
```

Edit `configs/feetech.yaml` (committed, machine-specific — same idea as
`configs/cameras.yaml`):

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

## 2. Check Feetech Ticks

```bash
handumi-calibrate-grippers monitor
```

Open/close each gripper and confirm `ticks` changes.

## 3. Home Servos (centre the encoder range)

The encoder wraps at the 0/4095 seam; travel crossing it makes the width readout
flip or saturate. Homing stores a correction so the current shaft angle reads
2048 (centre), clearing the range of the seam:

```bash
handumi-home-servos              # both sides
handumi-home-servos --side right # one side
```

Hold the gripper at **mid-travel** (~2040 ticks), press ENTER; the script reports
`OK` / `CHECK`. Re-calibrate afterwards.

A software unwrap in `handumi.devices.feetech.gripper` also tracks wraps continuously, so
an un-homed range is fine as long as recording **starts with the grippers roughly
closed**.

## 4. Calibrate Gripper Width

```bash
handumi-calibrate-grippers calibrate
handumi-calibrate-grippers calibrate --side right
```

For each side:

```text
enter max opening in mm
open gripper fully while watching live ticks, press ENTER
close gripper fully while watching live ticks, press ENTER
```

Use `--side left|right` to recalibrate one gripper without disturbing the other.
This writes to the per-user calibration cache (`~/.cache/handumi/calibration.yaml`),
not to `configs/feetech.yaml`.

Setup done — head back to [README.md](README.md) to teleoperate and record.
