"""Find Feetech servos on one or more serial ports."""

from __future__ import annotations

import argparse
from pathlib import Path

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/feetech.yaml"))
    parser.add_argument("--port", default=None, help="Serial port to scan.")
    parser.add_argument(
        "--all-ports",
        action="store_true",
        help="Scan all detected /dev/ttyUSB*, /dev/ttyACM*, and tty.usb* ports.",
    )
    parser.add_argument("--baudrate", type=int, default=None, help="Override baudrate from config.")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baudrate = args.baudrate or config.baudrate
    ids = range(args.start_id, args.end_id + 1)
    ports = _ports_to_scan(args.port, args.all_ports, config.port)

    any_found = False
    for port in ports:
        try:
            with FeetechBus(
                port=port,
                baudrate=baudrate,
                protocol_version=config.protocol_version,
            ) as bus:
                found = bus.scan(ids)
        except Exception as exc:
            print(f"{port}: ERROR {exc}")
            continue

        if not found:
            print(f"{port}: no servos found")
            continue
        any_found = True
        print(f"{port}: found Feetech servo IDs")
        for servo_id in found:
            print(f"  {servo_id}")

    if not any_found:
        print("No Feetech servos found.")


def _ports_to_scan(port: str | None, all_ports: bool, config_port: str | None) -> list[str]:
    if port:
        return [port]
    if not all_ports and config_port:
        return [config_port]

    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise SystemExit("pyserial is required for --all-ports.") from exc

    ports = [
        item.device
        for item in list_ports.comports()
        if "ttyUSB" in item.device or "ttyACM" in item.device or "tty.usb" in item.device
    ]
    if not ports:
        raise SystemExit("No serial USB ports detected.")
    return sorted(ports)


if __name__ == "__main__":
    main()
