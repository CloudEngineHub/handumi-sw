# Gripper Setup (Feetech + Cameras)

One-time **per-laptop** hardware setup before teleoperating or recording:
serial ports, camera indices, servo homing. Width calibration lives in
[README_tcp_offset.md](README_tcp_offset.md).

Ports (`servo_id`/`port`, camera `index_or_path`) are wiring — committed in
`configs/feetech.yaml` / `configs/cameras.yaml`; edit them directly. Homing
is stored in the servo's EEPROM (persists across power cycles and laptops).

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

### Troubleshooting: Feetech serial ports shows `none`

`Feetech serial ports -> none` means Linux did not create any `/dev/ttyACM*`
or `/dev/ttyUSB*` serial device. This is different from camera devices, which
show up as `/dev/video*`.

First check whether the serial device node exists:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

If no devices exist, check whether the USB adapter itself is visible:

```bash
lsusb
journalctl -k -f
```

Common Feetech USB serial adapters show up as QinHeng/CH34x devices such as
`1a86:55d3` or `1a86:7523`. If `lsusb` shows the adapter but there is still no
`/dev/ttyUSB*`, the USB cable and hub are probably fine, but the serial driver
did not bind.

On Arch Linux, this often happens right after a system update: the running
kernel and installed module tree no longer match. Check:

```bash
uname -r
modinfo ch341
ls /usr/lib/modules/$(uname -r)
```

If `modinfo ch341` fails or `/usr/lib/modules/$(uname -r)` is missing, reboot:

```bash
sudo reboot
```

After rebooting, reconnect the Feetech adapter and check again:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
handumi-setup-ports
```

If `/dev/ttyUSB0` exists but `handumi-setup-ports` reports it as unavailable
or permission denied, add your user to the serial device group shown by the
script. On Arch this is usually `uucp`; on Debian/Ubuntu it is usually
`dialout`:

```bash
sudo usermod -aG uucp $USER
```

Then log out and back in.

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

A software unwrap in `handumi.feetech.gripper` also tracks wraps continuously, so
an un-homed range is fine as long as recording **starts with the grippers roughly
closed**.

## 4. Calibrate Gripper Width

```bash
handumi-calibrate-grippers calibrate            # both sides
handumi-calibrate-grippers calibrate --side right
```

For each side: enter the max opening in mm, open fully (ENTER), close fully
(ENTER). This writes to the per-user cache
(`~/.cache/handumi/calibration.yaml`), never to the repo.

Setup done — head back to [README.md](README.md) to record. The other
calibration (controller → gripper TCP, once per mount design) is in
[README_tcp_offset.md](README_tcp_offset.md).
