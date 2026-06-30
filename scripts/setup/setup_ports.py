from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import yaml

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import FeetechConfig, GripperCalibration, load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively assign HandUMI gripper and camera ports.")
    parser.add_argument("--feetech-config", type=Path, default=Path("configs/feetech.yaml"))
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--left-servo-id", type=int, default=0)
    parser.add_argument("--right-servo-id", type=int, default=1)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=20)
    args = parser.parse_args()

    print("Disconnect both Feetech servo adapters and both wrist cameras.")
    input("Press ENTER when disconnected...")
    serial_baseline = _serial_ports()
    video_baseline = _video_devices()

    left_port, serial_baseline = _detect_serial("left Feetech servo", serial_baseline)
    left_camera, video_baseline = _detect_camera("left wrist camera", video_baseline)
    right_port, serial_baseline = _detect_serial("right Feetech servo", serial_baseline)
    right_camera, video_baseline = _detect_camera("right wrist camera", video_baseline)

    feetech_config = load_config(args.feetech_config)
    scan_ids = range(args.start_id, args.end_id + 1)
    left_current_id = _single_servo_id(left_port, feetech_config, scan_ids, "left")
    right_current_id = _single_servo_id(right_port, feetech_config, scan_ids, "right")

    _ensure_servo_id(left_port, left_current_id, args.left_servo_id, feetech_config)
    _ensure_servo_id(right_port, right_current_id, args.right_servo_id, feetech_config)

    feetech_out = FeetechConfig(
        port=None,
        baudrate=feetech_config.baudrate,
        protocol_version=feetech_config.protocol_version,
        left=GripperCalibration(
            servo_id=args.left_servo_id,
            port=left_port,
            closed_ticks=feetech_config.left.closed_ticks,
            open_ticks=feetech_config.left.open_ticks,
            max_width_mm=feetech_config.left.max_width_mm,
        ),
        right=GripperCalibration(
            servo_id=args.right_servo_id,
            port=right_port,
            closed_ticks=feetech_config.right.closed_ticks,
            open_ticks=feetech_config.right.open_ticks,
            max_width_mm=feetech_config.right.max_width_mm,
        ),
    )
    save_config(feetech_out, args.feetech_config)
    _save_camera_config(args.camera_config, left_camera, right_camera)

    print(f"Saved {args.feetech_config}")
    print(f"  left servo : port={left_port}, id={args.left_servo_id}")
    print(f"  right servo: port={right_port}, id={args.right_servo_id}")
    print(f"Saved {args.camera_config}")
    print(f"  left camera : index={left_camera}")
    print(f"  right camera: index={right_camera}")


def _detect_serial(label: str, before: set[str]) -> tuple[str, set[str]]:
    print(f"\nConnect the {label} cable only.")
    input("Press ENTER after it appears...")
    after = _serial_ports()
    new_ports = sorted(after - before)
    port = _choose_one(new_ports, f"{label} serial port", sorted(after))
    return port, after


def _detect_camera(label: str, before: set[int]) -> tuple[int, set[int]]:
    print(f"\nConnect the {label} only.")
    input("Press ENTER after it appears...")
    after = _video_devices()
    new_indices = sorted(after - before)
    readable = [index for index in new_indices if _camera_reads(index)]
    candidates = readable or new_indices
    camera = int(_choose_one([str(index) for index in candidates], f"{label} OpenCV index", [str(i) for i in sorted(after)]))
    return camera, after


def _serial_ports() -> set[str]:
    return set(glob.glob("/dev/serial/by-id/*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def _video_devices() -> set[int]:
    indices: set[int] = set()
    for path in glob.glob("/dev/video*"):
        suffix = Path(path).name.removeprefix("video")
        if suffix.isdigit():
            indices.add(int(suffix))
    return indices


def _choose_one(candidates: list[str], label: str, fallback: list[str]) -> str:
    if len(candidates) == 1:
        print(f"{label}: {candidates[0]}")
        return candidates[0]
    if candidates:
        print(f"Detected multiple candidates for {label}: {', '.join(candidates)}")
    else:
        print(f"No new device detected for {label}. Current devices: {', '.join(fallback) or 'none'}")
    value = input(f"Enter {label}: ").strip()
    if not value:
        raise SystemExit(f"{label} is required.")
    return value


def _camera_reads(index: int) -> bool:
    try:
        import cv2
    except ImportError:
        return True
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def _single_servo_id(port: str, config: FeetechConfig, ids: range, side: str) -> int:
    try:
        with FeetechBus(port=port, baudrate=config.baudrate, protocol_version=config.protocol_version) as bus:
            found = bus.scan(ids)
    except Exception as exc:
        raise SystemExit(f"{port}: could not scan {side} servo: {exc}") from exc
    if len(found) != 1:
        raise SystemExit(f"Expected exactly one {side} servo on {port}, found {found}.")
    return found[0]


def _ensure_servo_id(port: str, current_id: int, target_id: int, config: FeetechConfig) -> None:
    if current_id == target_id:
        print(f"{port}: ID already {target_id}")
        return
    with FeetechBus(port=port, baudrate=config.baudrate, protocol_version=config.protocol_version) as bus:
        try:
            bus.write_servo_id(current_id, target_id)
        except RuntimeError as exc:
            if bus.ping(target_id):
                print(f"{port}: warning: write returned an error, but servo responds as ID {target_id}.")
            else:
                raise SystemExit(f"{port}: could not write ID {current_id} -> {target_id}: {exc}") from exc
    print(f"{port}: ID {current_id} -> {target_id}")


def _save_camera_config(path: Path, left_index: int, right_index: int) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
    else:
        data = {}
    data.setdefault("left_wrist", {})
    data.setdefault("right_wrist", {})
    data["left_wrist"]["index_or_path"] = left_index
    data["right_wrist"]["index_or_path"] = right_index
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


if __name__ == "__main__":
    main()
