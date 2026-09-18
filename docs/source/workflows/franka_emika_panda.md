# Franka Panda with the right HandUMI

The `franka_emika_panda` embodiment uses **one arm and the right HandUMI only**. It supports
robot-free recording, QA, conversion, Viser replay, and simulated teleoperation
with the Google DeepMind Menagerie model. Physical Franka teleoperation and
`replay-real` are not implemented.

## Configure the wearable

Use the normal machine-local `configs/rig.yaml`, with a right wrist camera,
right encoder port/ID, and your tracking connection. No left camera, encoder,
or tracked controller is required. For example:

```yaml
recording:
  robot: franka_emika_panda
  device: pico
  cameras: [right_wrist, workspace]
  fps: 30
feetech:
  baudrate: 1000000
  protocol_version: 0
  right:
    port: /dev/ttyACM0  # replace with the actual local port
    servo_id: 1
```

Declare selected cameras under `cameras` as usual. The workspace camera is
optional for capture, but needed for the dataset direction check. PICO and
Meta Quest remain supported; use the device and calibration for your rig.

```bash
handumi calibrate grippers --side right
```

Controller-to-TCP calibration describes the **wearable tool**, not the Panda.
The profile references the existing PICO/Piper-tip and Meta/ARX5-tip
calibrations. Reuse them only for that physical assembly; otherwise pass its
measured calibration with `--controller-tcp-calibration`. Follow
[setup](../setup.md) for camera and session/table calibration.

## Record

```bash
handumi record --robot franka_emika_panda --device pico \
  --cameras right_wrist,workspace \
  --output-dir outputs/panda-demo --task "pick and place" \
  --session-calibration outputs/calibration/session.yaml --dry-run
```

Use your actual session calibration path, then remove `--dry-run` to record.
Panda selects the right side automatically; `--side right` is equivalent.
Implicit camera defaults omit the left wrist. Only the right encoder is
opened, and only right tracking gates recording. Start/save using the right
double-squeeze or voice; use voice `restart` to discard/repeat without a left
gripper.

Raw captures retain the standard 16D schema. The absent side has a neutral
pose, zero width and false tracking flags; `handumi.active_sides` records
`["right"]`. QA ignores that absent side but still checks right tracking,
pose freezes and enabled sensors. Resume preserves the selection; merge
refuses different active sides. Legacy datasets remain bilateral.

## Review, convert and replay

```bash
handumi dataset qa outputs/panda-demo --robot franka_emika_panda --deployment-profile sim
handumi convert outputs/panda-demo --robot franka_emika_panda --output outputs/panda-joints
handumi replay-joints outputs/panda-joints --robot franka_emika_panda
```

For a pilot without workspace video or enough episodes for the direction
check, add `--skip-direction` to QA. Review remaining warnings before
conversion. The converted vector has **eight columns**: seven joint angles
in radians and one logical opening in meters. A bilateral recording can also
be retargeted to Panda; its left trajectory is ignored. The normal reviewed
trajectory cache and provenance checks still apply.

If QA rejects episodes, remove them before conversion and use the curated
dataset as input. Curation requires at least one exclusion; skip it when
every episode is accepted:

```bash
handumi dataset curate outputs/panda-demo --output outputs/panda-clean --exclude-rejected
handumi convert outputs/panda-clean --robot franka_emika_panda --output outputs/panda-joints
```

Preview the raw recording directly:

```bash
handumi replay outputs/panda-demo --robot franka_emika_panda --episode 0 \
  --deployment-profile sim --strict-ik
```

Table-calibrated captures automatically use `absolute-table`. The shipped
simulation placement uses the base as table origin (+X forward, +Y left,
+Z up). Demonstrations must lie within reach; use an appropriate deployment
calibration for a different placement. `--retarget-mode local-relative`
provides a motion preview anchored at home when absolute scene placement is
not required.

## Live simulation

```bash
handumi teleop --robot franka_emika_panda --device pico --side right
```

Viser shows one Panda. Its Menagerie contact model is available with a
supported `--scene`. Maximum opening is 80 mm; replay/conversion preserve
physical aperture, so a 40 mm HandUMI opening becomes 40 mm on Panda.

The pinned model revision, license and derivations are documented in
`assets/franka_emika_panda/README.md`. Synthetic tests validate software and model geometry;
they do not establish physical calibration or hardware tracking accuracy.
