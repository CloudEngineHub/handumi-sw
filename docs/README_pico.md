# PICO Tracking Setup (XRoboToolkit)

Streams body/controller poses via **XRoboToolkit**: a PC service
(workstation) + headset app + Python SDK. `bash install.sh` (without
`--skip-xrt`) builds the SDK. The PICO is worn on the neck as a tracking
base, one controller mounted on each gripper.

## Install (one-time)

1. PC service (Ubuntu x86_64; other platforms in the
   [releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)):

   ```bash
   wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
   sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
   ```

2. Headset app: in the PICO browser, download and install
   [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-PICO/releases)
   (needs Developer Mode on; app lands in Library → Unknown).
3. USB mode (default transport): enable USB debugging
   (*Settings → Developer*), connect USB-C, authorize via `adb devices`.

## Connect (per session)

1. Start the PC service: `bash /opt/apps/roboticsservice/runService.sh`
   (HandUMI also starts it automatically when recording).
2. Launch XRoboToolkit on the PICO and set the PC-service IP+port:
   `127.0.0.1:63901` for USB (default), or the workstation LAN IP for
   `--pico-wifi`.
3. Start streaming in the app (body tracking / controllers as needed).

## Smoke-test

```bash
handumi-record --device pico --skip-feetech \
  --repo-id local/pico_smoke --output-dir outputs/datasets/pico_smoke \
  --task "pico smoke" --num-episodes 1 --episode-time-s 10
```

Good = `xrobotoolkit_sdk initialised.` and no repeated
`still waiting for PICO data`. `--pico-mode mandos|object|whole-body`
selects what streams; `--manual-control` maps episodes to PICO buttons
(**A** start/stop, **B** repeat, **Y** finish).

## Troubleshooting

- **`xrt.init()` core dump** — PC service not running.
- **All zeros** — app not streaming, wrong IP/port, or (USB) missing
  `adb reverse --list` → `tcp:63901`.
- **SDK import error** — re-run `bash install.sh` without `--skip-xrt`.

---

Next: hardware + width calibration ([README_gripper_width.md](README_gripper_width.md)),
mount calibration ([README_tcp_offset.md](README_tcp_offset.md)), then
record per [README.md](../README.md).
