"""Monitor and calibrate HandUMI Feetech gripper encoder widths.

``monitor`` streams live encoder ticks so you can confirm each gripper responds
when opened/closed. The default guided flow records the physical aperture,
homes the encoder at mid-travel, and then records open/closed ticks, writing the
result to the per-user cache at
``~/.cache/handumi/calibration.yaml``.

Usage
-----
::

    handumi calibrate grippers
    handumi calibrate grippers --side right
    handumi calibrate grippers monitor
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    load_config,
    save_calibration,
    user_calibration_path,
)
from handumi.scripts.setup import home_servos


@dataclass
class Monitor:
    port: str
    servo_id: int
    bus: FeetechBus
    initial: int
    last: int
    peak_delta: int = 0
    failed_reads: int = 0
    last_error: str | None = None

    def update(self) -> None:
        try:
            self.last = self.bus.read_position(self.servo_id)
        except RuntimeError as exc:
            self.failed_reads += 1
            self.last_error = str(exc)
            return
        self.last_error = None
        self.peak_delta = max(self.peak_delta, abs(self.last - self.initial))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate HandUMI Feetech grippers in one guided flow: physical "
            "width, mid-travel homing, full opening, and full closure."
        )
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Rig file containing Feetech servo_id/port values.",
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=None,
        help="Override the calibration cache path (default: per-user cache).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("monitor",),
        help="Use 'monitor' to watch encoder ticks instead of calibrating.",
    )
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--max-width-mm", type=float, default=None)
    parser.add_argument("--left-max-width-mm", type=float, default=None)
    parser.add_argument("--right-max-width-mm", type=float, default=None)
    parser.add_argument("--interval-s", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--keep-torque", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.calibration_config = args.calibration_config or user_calibration_path()
    if args.interval_s is None:
        args.interval_s = 0.2 if args.command == "monitor" else 0.1
    print(f"Using rig: {args.rig_config}")
    print(f"Using calibration cache: {args.calibration_config}")
    if args.command == "monitor":
        cmd_monitor(args)
    else:
        cmd_calibrate(args)


def cmd_monitor(args: argparse.Namespace) -> None:
    config = load_config(args.rig_config, args.calibration_config)
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
                try:
                    bus.disable_torque(calibration.servo_id)
                except RuntimeError as exc:
                    print(
                        f"Warning: could not disable torque on servo "
                        f"{calibration.servo_id} at {port}: {exc}"
                    )
            try:
                ticks = bus.read_position(calibration.servo_id)
            except RuntimeError as exc:
                raise SystemExit(
                    f"Could not read initial position from servo "
                    f"{calibration.servo_id} at {port}: {exc}"
                ) from exc
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
    current = load_config(args.rig_config, args.calibration_config)
    sides = ["left", "right"] if args.side == "both" else [args.side]
    side_width = {"left": args.left_max_width_mm, "right": args.right_max_width_mm}

    results = {"left": current.left, "right": current.right}
    for side in sides:
        calibration = getattr(current, side)
        port = _side_port(current, calibration, side)
        width_mm = side_width[side] or args.max_width_mm
        if width_mm is None:
            width_mm = _prompt_positive_float(f"{side} max gripper opening in mm")

        print(f"\n{side} calibration steps: mid-travel, fully open, fully closed.")
        home_servos._home_side(
            side=side,
            port=port,
            calibration=calibration,
            baudrate=current.baudrate,
            protocol_version=current.protocol_version,
            interval_s=args.interval_s,
        )
        closed, open_, width_mm = _calibrate_side(
            side=side,
            port=port,
            calibration=calibration,
            baudrate=current.baudrate,
            protocol_version=current.protocol_version,
            max_width_mm=width_mm,
            interval_s=args.interval_s,
        )
        results[side] = GripperCalibration(
            calibration.servo_id, closed, open_, width_mm, calibration.port
        )

    config = FeetechConfig(
        port=current.port,
        baudrate=current.baudrate,
        protocol_version=current.protocol_version,
        left=results["left"],
        right=results["right"],
    )
    saved_path = save_calibration(config, args.calibration_config)
    print(f"Saved calibration to {saved_path}")
    for side in sides:
        c = results[side]
        print(f"{side}: closed={c.closed_ticks}, open={c.open_ticks}, max_width_mm={c.max_width_mm}")


def _calibrate_side(
    *,
    side: str,
    port: str,
    calibration: GripperCalibration,
    baudrate: int,
    protocol_version: int,
    max_width_mm: float | None,
    interval_s: float,
) -> tuple[int, int, float]:
    print(f"\nCalibrating {side} gripper: port={port}, servo_id={calibration.servo_id}")
    width_mm = max_width_mm or _prompt_positive_float(f"{side} max gripper opening in mm")
    with FeetechBus(port=port, baudrate=baudrate, protocol_version=protocol_version) as bus:
        try:
            bus.disable_torque(calibration.servo_id)
        except RuntimeError as exc:
            print(f"Warning: could not disable torque on {side} servo: {exc}")
        open_ticks = _capture_live_ticks(
            bus,
            calibration.servo_id,
            f"Open {side} gripper to maximum width",
            interval_s,
        )
        closed_ticks = _capture_live_ticks(
            bus,
            calibration.servo_id,
            f"Close {side} gripper fully",
            interval_s,
        )
    return closed_ticks, open_ticks, width_mm


def _capture_live_ticks(bus: FeetechBus, servo_id: int, prompt: str, interval_s: float) -> int:
    print(f"{prompt}. Press ENTER to capture the current encoder value.")
    initial: int | None = None
    latest: int | None = None
    while True:
        try:
            latest = bus.read_position(servo_id)
        except RuntimeError as exc:
            if latest is None:
                sys.stdout.write(f"\r  read failed: {exc}")
            else:
                sys.stdout.write(
                    f"\r  ticks={latest:5d}  last read failed: {exc}"
                )
        else:
            if initial is None:
                initial = latest
            delta = latest - initial
            sys.stdout.write(f"\r  ticks={latest:5d}  delta={delta:+6d}")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], interval_s)
        if ready:
            sys.stdin.readline()
            sys.stdout.write("\n")
            if latest is None:
                raise RuntimeError(
                    f"Cannot capture servo {servo_id}: no valid encoder value was read."
                )
            return latest


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
    print("port          id  ticks  delta  peak_delta  status")
    for monitor in monitors:
        delta = monitor.last - monitor.initial
        status = "ok"
        if monitor.last_error is not None:
            status = f"read failed x{monitor.failed_reads}: {monitor.last_error}"
        print(
            f"{monitor.port:<12} {monitor.servo_id:>2}  {monitor.last:>5}  "
            f"{delta:>5}  {monitor.peak_delta:>10}  {status}"
        )
    print()


if __name__ == "__main__":
    main()
