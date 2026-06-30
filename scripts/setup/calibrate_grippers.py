from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import FeetechConfig, GripperCalibration, load_config, save_config


@dataclass
class Monitor:
    port: str
    servo_id: int
    bus: FeetechBus
    initial: int
    last: int
    peak_delta: int = 0

    def update(self) -> None:
        self.last = self.bus.read_position(self.servo_id)
        self.peak_delta = max(self.peak_delta, abs(self.last - self.initial))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and calibrate HandUMI Feetech gripper encoders.")
    parser.add_argument("--config", type=Path, default=Path("configs/feetech.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor = subparsers.add_parser("monitor", help="Watch encoder ticks for configured grippers.")
    monitor.add_argument("--duration-s", type=float, default=20.0)
    monitor.add_argument("--interval-s", type=float, default=0.2)
    monitor.add_argument("--keep-torque", action="store_true")
    monitor.set_defaults(func=cmd_monitor)

    calibrate = subparsers.add_parser("calibrate", help="Record left/right open/closed ticks and max width.")
    calibrate.add_argument("--max-width-mm", type=float, default=None)
    calibrate.add_argument("--left-max-width-mm", type=float, default=None)
    calibrate.add_argument("--right-max-width-mm", type=float, default=None)
    calibrate.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


def cmd_monitor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    left_port = _side_port(config, config.left, "left")
    right_port = _side_port(config, config.right, "right")
    monitors: list[Monitor] = []
    buses: list[FeetechBus] = []
    try:
        for port, calibration in ((left_port, config.left), (right_port, config.right)):
            bus = FeetechBus(port=port, baudrate=config.baudrate, protocol_version=config.protocol_version)
            bus.open()
            buses.append(bus)
            if not args.keep_torque:
                bus.disable_torque(calibration.servo_id)
            ticks = bus.read_position(calibration.servo_id)
            monitors.append(Monitor(port, calibration.servo_id, bus, ticks, ticks))

        print("Open/close each gripper and check that ticks or peak_delta changes.")
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            for monitor in monitors:
                monitor.update()
            _print_monitor(monitors)
            time.sleep(args.interval_s)
    finally:
        for bus in buses:
            bus.close()


def cmd_calibrate(args: argparse.Namespace) -> None:
    current = load_config(args.config)
    left_port = _side_port(current, current.left, "left")
    right_port = _side_port(current, current.right, "right")
    left_width_mm = args.left_max_width_mm or args.max_width_mm
    right_width_mm = args.right_max_width_mm or args.max_width_mm

    left_closed, left_open, left_width_mm = _calibrate_side(
        side="left",
        port=left_port,
        calibration=current.left,
        baudrate=current.baudrate,
        protocol_version=current.protocol_version,
        max_width_mm=left_width_mm,
    )
    right_closed, right_open, right_width_mm = _calibrate_side(
        side="right",
        port=right_port,
        calibration=current.right,
        baudrate=current.baudrate,
        protocol_version=current.protocol_version,
        max_width_mm=right_width_mm,
    )

    config = FeetechConfig(
        port=current.port,
        baudrate=current.baudrate,
        protocol_version=current.protocol_version,
        left=GripperCalibration(current.left.servo_id, left_closed, left_open, left_width_mm, current.left.port),
        right=GripperCalibration(current.right.servo_id, right_closed, right_open, right_width_mm, current.right.port),
    )
    save_config(config, args.config)
    print(f"Saved {args.config}")
    print(f"left : closed={left_closed}, open={left_open}, max_width_mm={left_width_mm}")
    print(f"right: closed={right_closed}, open={right_open}, max_width_mm={right_width_mm}")


def _calibrate_side(
    *,
    side: str,
    port: str,
    calibration: GripperCalibration,
    baudrate: int,
    protocol_version: int,
    max_width_mm: float | None,
) -> tuple[int, int, float]:
    width_mm = max_width_mm or _prompt_positive_float(f"{side} max gripper opening in mm")
    with FeetechBus(port=port, baudrate=baudrate, protocol_version=protocol_version) as bus:
        input(f"Open {side} gripper to maximum width, then press ENTER...")
        open_ticks = bus.read_position(calibration.servo_id)
        input(f"Close {side} gripper fully, then press ENTER...")
        closed_ticks = bus.read_position(calibration.servo_id)
    return closed_ticks, open_ticks, width_mm


def _prompt_positive_float(label: str) -> float:
    while True:
        value = input(f"{label}: ").strip()
        try:
            parsed = float(value)
        except ValueError:
            print("Enter a numeric value.")
            continue
        if parsed <= 0:
            print("Value must be positive.")
            continue
        return parsed


def _side_port(config: FeetechConfig, calibration: GripperCalibration, side: str) -> str:
    port = calibration.port or config.port
    if not port:
        raise SystemExit(f"{side} Feetech port is not configured.")
    return port


def _print_monitor(monitors: list[Monitor]) -> None:
    print("port          id  ticks  delta  peak_delta")
    for monitor in monitors:
        delta = monitor.last - monitor.initial
        print(f"{monitor.port:<12} {monitor.servo_id:>2}  {monitor.last:>5}  {delta:>5}  {monitor.peak_delta:>10}")
    print()


if __name__ == "__main__":
    main()
