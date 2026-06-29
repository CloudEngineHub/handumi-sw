"""Calibrate HandUMI gripper closed/open Feetech encoder positions."""

from __future__ import annotations

import argparse
from pathlib import Path

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    load_config,
    save_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/feetech.yaml"))
    parser.add_argument("--port", default=None, help="Override shared serial port from config.")
    parser.add_argument("--left-port", default=None, help="Override left serial port.")
    parser.add_argument("--right-port", default=None, help="Override right serial port.")
    parser.add_argument("--baudrate", type=int, default=None, help="Override baudrate from config.")
    parser.add_argument("--max-width-mm", type=float, default=None)
    parser.add_argument("--left-max-width-mm", type=float, default=None)
    parser.add_argument("--right-max-width-mm", type=float, default=None)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Do not read hardware; require tick values through CLI arguments.",
    )
    parser.add_argument("--left-closed", type=int, default=None)
    parser.add_argument("--left-open", type=int, default=None)
    parser.add_argument("--right-closed", type=int, default=None)
    parser.add_argument("--right-open", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = load_config(args.config)
    baudrate = args.baudrate or current.baudrate
    left_port = args.left_port or args.port or current.left.port or current.port
    right_port = args.right_port or args.port or current.right.port or current.port
    shared_port = args.port if args.port is not None else current.port
    if not left_port or not right_port:
        raise SystemExit("Configure left/right Feetech ports before calibration.")

    if args.manual:
        left_closed = _required(args.left_closed, "--left-closed")
        left_open = _required(args.left_open, "--left-open")
        right_closed = _required(args.right_closed, "--right-closed")
        right_open = _required(args.right_open, "--right-open")
    else:
        left_bus = FeetechBus(
            port=left_port,
            baudrate=baudrate,
            protocol_version=current.protocol_version,
        )
        right_bus = (
            left_bus
            if right_port == left_port
            else FeetechBus(
                port=right_port,
                baudrate=baudrate,
                protocol_version=current.protocol_version,
            )
        )
        left_bus.open()
        if right_bus is not left_bus:
            right_bus.open()
        try:
            input("Close both HandUMI grippers, then press ENTER...")
            left_closed = left_bus.read_position(current.left.servo_id)
            right_closed = right_bus.read_position(current.right.servo_id)
            input("Open both HandUMI grippers, then press ENTER...")
            left_open = left_bus.read_position(current.left.servo_id)
            right_open = right_bus.read_position(current.right.servo_id)
        finally:
            if right_bus is not left_bus:
                right_bus.close()
            left_bus.close()

    left_max_width_mm = args.left_max_width_mm or args.max_width_mm or current.left.max_width_mm
    right_max_width_mm = args.right_max_width_mm or args.max_width_mm or current.right.max_width_mm
    if left_max_width_mm is None or right_max_width_mm is None:
        raise SystemExit(
            "Set --max-width-mm, or --left-max-width-mm and --right-max-width-mm."
        )
    config = FeetechConfig(
        port=shared_port,
        baudrate=baudrate,
        protocol_version=current.protocol_version,
        left=GripperCalibration(
            servo_id=current.left.servo_id,
            port=args.left_port if args.left_port is not None else (None if args.port else current.left.port),
            closed_ticks=left_closed,
            open_ticks=left_open,
            max_width_mm=left_max_width_mm,
        ),
        right=GripperCalibration(
            servo_id=current.right.servo_id,
            port=(
                args.right_port if args.right_port is not None else (None if args.port else current.right.port)
            ),
            closed_ticks=right_closed,
            open_ticks=right_open,
            max_width_mm=right_max_width_mm,
        ),
    )
    save_config(config, args.config)
    print(f"Saved Feetech calibration to {args.config}")
    print(f"  left : id={config.left.servo_id}, port={left_port}, closed={left_closed}, open={left_open}, max_width_mm={left_max_width_mm}")
    print(f"  right: id={config.right.servo_id}, port={right_port}, closed={right_closed}, open={right_open}, max_width_mm={right_max_width_mm}")


def _required(value: int | None, flag: str) -> int:
    if value is None:
        raise SystemExit(f"{flag} is required with --manual.")
    return value


if __name__ == "__main__":
    main()
