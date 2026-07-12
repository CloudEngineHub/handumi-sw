# Guided Hardware Setup

`handumi-setup-hardware` is the interactive setup command for machine-local
hardware wiring. It creates `configs/rig.yaml` if needed and writes the
detected CAN and Feetech mappings into that ignored local config file.

Current scope:

- Piper CAN mapping and CAN interface repair.
- Feetech gripper serial-port and servo-ID mapping.
- Guided Feetech servo homing and gripper-width calibration.
- PICO USB/ADB session preparation.

It does not configure cameras or compute the controller-to-TCP offset. Those
steps remain in the dedicated calibration READMEs.

## Typical First-Time Flow

1. Install HandUMI and activate the virtual environment.
2. If preparing new Feetech servos, set their IDs first:
   [README_gripper_width.md](README_gripper_width.md#servo-ids).
3. Run the guided hardware setup:

   ```bash
   handumi-setup-hardware --robot piper --device pico
   ```

4. Configure cameras and review the gripper calibration details:
   [README_gripper_width.md](README_gripper_width.md).
5. Finish tracking setup:
   [README_pico.md](README_pico.md) or [README_quest.md](README_quest.md).
6. Calibrate the controller-to-gripper TCP offset:
   [README_tcp_offset.md](README_tcp_offset.md).

## What the Wizard Does

For Piper, the wizard maps CAN by reconnecting one adapter at a time:

1. Right Piper CAN adapter.
2. Left Piper CAN adapter.

It writes:

```yaml
robots:
  piper:
    can:
      bitrate: 1000000
      restart_ms: 100
      left_port: can0
      right_port: can1
```

Then it checks the configured CAN interfaces. If they are down, at the wrong
bitrate, missing link, or `BUS-OFF`, it asks for `sudo` and runs the required
`ip link` repair commands.

For Feetech, the wizard maps gripper adapters by reconnecting:

1. Right Feetech adapter.
2. Left Feetech adapter.

It writes:

```yaml
feetech:
  baudrate: 1000000
  protocol_version: 0
  left:
    servo_id: 0
    port: /dev/ttyACM0
  right:
    servo_id: 1
    port: /dev/ttyACM1
```

If `--device pico` is used, the wizard also prepares the PICO USB/ADB session.

## Commands

Default Piper + PICO setup:

```bash
handumi-setup-hardware --robot piper --device pico
```

Use Meta Quest tracking instead of PICO preparation:

```bash
handumi-setup-hardware --robot piper --device meta
```

Keep the existing CAN mapping and only repair/check CAN plus run the remaining
steps:

```bash
handumi-setup-hardware --robot piper --device pico --skip-can-map
```

Keep the existing Feetech mapping:

```bash
handumi-setup-hardware --robot piper --device pico --skip-feetech-map
```

Skip PICO ADB preparation:

```bash
handumi-setup-hardware --robot piper --device pico --skip-pico
```

Set CAN parameters explicitly:

```bash
handumi-setup-hardware --robot piper --device pico --bitrate 1000000 --restart-ms 100
```

Use a different local rig config:

```bash
handumi-setup-hardware --robot piper --device pico --rig-config configs/rig.local.yaml
```

## Options

- `--robot piper` selects the real robot hardware backend.
- `--device pico|meta` selects the tracking device setup path.
- `--rig-config <path>` selects the local YAML config to create/update.
- `--bitrate <int>` sets the CAN bitrate written to the rig config.
- `--restart-ms <int>` sets the CAN automatic restart time.
- `--skip-can-map` keeps the existing `robots.piper.can` mapping.
- `--skip-can-repair` skips `sudo ip link` CAN repair.
- `--skip-feetech-map` keeps the existing `feetech` mapping.
- `--feetech-start-id <int>` and `--feetech-end-id <int>` set the servo ID
  scan range.
- `--skip-pico` skips PICO ADB preparation.
- `--skip-adb-check` skips the `adb devices` check during PICO setup.

## Troubleshooting

### Serial permissions were updated

Close the terminal session and log in again, or restart the machine. Group
membership changes do not apply to the current session.

### CAN repair asks for sudo

CAN interfaces require elevated permissions for `ip link set`. The wizard asks
for `sudo` only when it needs to bring CAN up or repair the bitrate/state.

### CAN remains down or `BUS-OFF`

Check robot power, CAN wiring, adapter connection, and that the correct
left/right adapters were mapped. Then re-run:

```bash
handumi-setup-hardware --robot piper --device pico --skip-feetech-map --skip-pico
```

### Feetech mapping cannot find a servo

Connect one adapter/gripper at a time as requested by the prompt. If the servo
ID is outside the default scan range, pass a wider range:

```bash
handumi-setup-hardware --robot piper --device pico --feetech-start-id 0 --feetech-end-id 30
```

### PICO ADB is not detected

Enable USB debugging on the headset, authorize the workstation, and check:

```bash
adb devices
```

Use `--pico-wifi` later in teleop/record commands when you want to stream over
Wi-Fi instead of USB/ADB.
