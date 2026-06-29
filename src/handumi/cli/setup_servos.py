"""Assign HandUMI left/right Feetech servo IDs and ports."""

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
    parser.add_argument("--port", default=None, help="Shared serial port saved to config.")
    parser.add_argument("--left-port", default=None, help="Serial port for the left gripper.")
    parser.add_argument("--right-port", default=None, help="Serial port for the right gripper.")
    parser.add_argument("--baudrate", type=int, default=None, help="Override baudrate saved to config.")
    parser.add_argument("--left-id", type=int, default=0)
    parser.add_argument("--right-id", type=int, default=1)
    parser.add_argument(
        "--write-id",
        choices=("left", "right"),
        default=None,
        help="Physically change one connected servo ID before saving config.",
    )
    parser.add_argument(
        "--current-id",
        type=int,
        default=None,
        help="Current ID of the connected servo when using --write-id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = load_config(args.config)
    baudrate = args.baudrate or current.baudrate
    shared_port = args.port if args.port is not None else current.port
    left_port = args.left_port if args.left_port is not None else (None if args.port else current.left.port)
    right_port = (
        args.right_port if args.right_port is not None else (None if args.port else current.right.port)
    )

    if args.write_id:
        target_id = args.left_id if args.write_id == "left" else args.right_id
        current_id = args.current_id
        if current_id is None:
            raise SystemExit("--current-id is required with --write-id.")
        port = _side_port(args, args.write_id, current)
        with FeetechBus(
            port=port,
            baudrate=baudrate,
            protocol_version=current.protocol_version,
        ) as bus:
            bus.write_servo_id(current_id, target_id)
        print(f"Wrote {args.write_id} servo ID: {current_id} -> {target_id} on {port}")

    config = FeetechConfig(
        port=shared_port,
        baudrate=baudrate,
        protocol_version=current.protocol_version,
        left=GripperCalibration(
            servo_id=args.left_id,
            port=left_port,
            closed_ticks=current.left.closed_ticks,
            open_ticks=current.left.open_ticks,
            max_width_mm=current.left.max_width_mm,
        ),
        right=GripperCalibration(
            servo_id=args.right_id,
            port=right_port,
            closed_ticks=current.right.closed_ticks,
            open_ticks=current.right.open_ticks,
            max_width_mm=current.right.max_width_mm,
        ),
    )
    save_config(config, args.config)
    print(f"Saved Feetech config to {args.config}")
    print(f"  shared port   : {config.port}")
    print(f"  left servo    : id={config.left.servo_id}, port={config.left.port or config.port}")
    print(f"  right servo   : id={config.right.servo_id}, port={config.right.port or config.port}")


def _side_port(args: argparse.Namespace, side: str, current: FeetechConfig) -> str:
    if side == "left":
        port = args.left_port or args.port or current.left.port or current.port
    else:
        port = args.right_port or args.port or current.right.port or current.port
    if not port:
        raise SystemExit(f"No serial port configured for {side}.")
    return port


if __name__ == "__main__":
    main()
