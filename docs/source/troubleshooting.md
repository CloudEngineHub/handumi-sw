# Troubleshooting

## Installation Stops During a Build

`install.sh` aborts on the first failed build. Both common failures are missing
system packages from
[System Prerequisites](getting_started/installation.md#system-prerequisites):

| Message | Install |
| --- | --- |
| `build.sh: line 25: cmake: command not found` | `build-essential cmake` |
| `fatal error: Python.h: No such file or directory` (building `evdev`) | `python3-dev` |

Install the package and rerun the same `install.sh` command. Completed steps
are detected and skipped, so the XRoboToolkit clone and native build are not
repeated.

## Device or Port Missing

```bash
handumi setup ports
lsusb
ls /dev/ttyACM* /dev/ttyUSB*
```

Reconnect one device at a time. Log out and back in after serial-group changes.

:::{dropdown} Adapter visible in lsusb but no serial port exists
Common Feetech adapters use the CH341 driver. Check the running kernel and
installed module tree:

```bash
uname -r
modinfo ch341
ls /usr/lib/modules/$(uname -r)
```

If the module tree does not match the running kernel, reboot after the system
update, reconnect the adapter, and rerun `handumi setup ports`.
:::

## Quest Does Not Stream

Keep HandUMI Quest App in the foreground, confirm both devices share a network, update `quest_ip` in `configs/rig.yaml`, and wake both controllers.

## PICO Does Not Stream

Confirm the PC service and headset stream are running. For USB, check:

```bash
adb devices
adb reverse --list
```

## Gripper Width Is Wrong

Confirm side/port mappings, then run the unified calibration. It guides you
through mid-travel homing, full opening, and full closure:

```bash
handumi calibrate grippers
```

If left/right motion is swapped, correct the mapping in `configs/rig.yaml`;
do not compensate by reversing calibration values. If a camera appears twice,
test the first `/dev/video*` node associated with that physical device.

## Voice Control Does Not Respond

Check what the recorder is listening to:

```bash
handumi doctor
```

If the microphone is the wrong one, name it explicitly with
`--voice-device <name-or-index>`; the default follows the system input, so a
headset only takes over once the OS has switched to it. If commands are heard
but ignored, the phrase must be exactly "start recording", "stop recording", or
"restart" — nothing else is in the recognizer's vocabulary. Lower
`--voice-confidence` if a correct phrase is still being dropped, and raise it if
the room is noisy enough to trigger commands on its own.

To record without voice, use `--no-voice-control`, and add `--clap-control` to
keep a hands-free control through gripper squeezes.

## Recording Is Rejected

Inspect `meta/handumi_quality.json`. The common causes are tracking loss, stale cameras, synchronization errors, frozen poses, large motion jumps, or an episode that is too short.

## Replay Prints a CUPTI Traceback

If JAX reports `Unable to load cuPTI` but replay continues, force the supported
CPU path for the command:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0
```

This warning concerns optional CUDA profiling libraries, not the dataset or
robot IK.

## Dataset Video Loading Fails in TorchCodec

A traceback ending in `Could not load libtorchcodec`, `libavutil.so.*`, or
`libavdevice.so.*` occurs while LeRobot opens the dataset's MP4 features. It
happens while a command decodes camera video and does not by itself indicate a
damaged dataset. Current trajectory replay reads the Parquet state columns
without opening MP4 files, so a plain `handumi replay` should not require
TorchCodec. Report a replay traceback if it still enters `decode_video_frames`.

On Ubuntu/Debian, install the system FFmpeg package:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Confirm that the executables and TorchCodec loader work:

```bash
ffmpeg -version
ffprobe -version
uv run python -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec OK')"
```

Errors for FFmpeg major versions that are not installed are expected in the
expanded TorchCodec traceback. Inspect the block for the installed version; on
Ubuntu 24.04 the repository package is FFmpeg 6 and provides, among the other
runtime libraries, `libavdevice.so.60`.

This video-runtime failure is unrelated to a preceding message such as
`Uninstalled xrobotoolkit-sdk`. A standalone `uv sync --extra sim` may remove
that locally installed PICO package because it is not part of the portable
lockfile. Replay does not use XRoboToolkit. Run `bash install.sh --sim` later if
the same workstation must retain or restore PICO capture support.

## Viser Shows Trajectories but No Robot

Messages such as `Can't find meshes/visual/base_link.glb` mean the URDF loaded
but its visual asset paths did not resolve. Restart replay after updating the
checkout. TRLC-DK1 meshes must exist under:

```text
assets/trlc-dk1/meshes/visual/
assets/trlc-dk1/meshes/collision/
```

Run this check from the repository root:

```bash
JAX_PLATFORMS=cpu uv run python -c \
  "from handumi.robots.registry import load_embodiment; u=load_embodiment('trlc_dk1').load_urdf(load_meshes=True); print(len(u.scene.geometry))"
```

The current GLB assets expand to hundreds of internal submeshes; a nonzero
count without `Can't find` messages confirms that the visuals loaded.

## Piper CAN Is Down or BUS-OFF

This applies only to physical Piper teleoperation. Check robot power and wiring,
then follow the CAN checks in
[Piper Hardware Setup](physical_robots/piper_setup.md#verify-can-and-troubleshoot-the-mapping).

## Piper Real Arms Do Not Start

Test simulation first, verify both controllers are tracked, confirm CAN is up,
and use `--side right` for the first hardware check. See
[First real teleoperation](physical_robots/piper_setup.md#first-real-teleoperation)
for the complete startup sequence.
