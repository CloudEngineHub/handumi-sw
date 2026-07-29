#!/usr/bin/env python3
"""Print raw Piper gripper feedback values for both arms.

This is read-only: it does not enable the arms, home, or send GripperCtrl.
Open the grippers however you want, then run this script to see the raw
position feedback reported by each Piper.

Examples::

    # Print feedback from both grippers for 10 s (default).
    .venv/bin/python tests/real/piper/print_piper_gripper_feedback.py

    # Left gripper only, longer sample window.
    .venv/bin/python tests/real/piper/print_piper_gripper_feedback.py \\
        --side left --duration-s 30 --rate-hz 10

    # Repair CAN interfaces if they are down, then print.
    .venv/bin/python tests/real/piper/print_piper_gripper_feedback.py --repair-can
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import load_piper_can_settings


SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--repair-can", action="store_true")
    return parser.parse_args()


def _selected_sides(side: str) -> tuple[str, ...]:
    return SIDES if side == "both" else (side,)


def _port_for_side(settings, side: str) -> str:
    return settings.left_port if side == "left" else settings.right_port


def _connect(port: str):
    try:
        from piper_sdk import C_PiperInterface_V2
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing piper_sdk. Install real Piper support.") from exc

    arm = C_PiperInterface_V2(port)
    arm.ConnectPort()
    time.sleep(0.3)
    return arm


def _status_bits(status) -> str:
    names = [
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "sensor_status",
        "driver_error_status",
        "driver_enable_status",
        "homing_status",
    ]
    active = [name for name in names if bool(getattr(status, name, False))]
    return ",".join(active) if active else "none"


def _read_gripper_feedback(arm) -> tuple[int, int, str]:
    feedback = arm.GetArmGripperMsgs().gripper_state
    return (
        int(feedback.grippers_angle),
        int(feedback.grippers_effort),
        _status_bits(feedback.status_code),
    )


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0 or args.rate_hz <= 0.0:
        raise SystemExit("--duration-s and --rate-hz must be > 0.")

    settings = load_piper_can_settings(args.rig_config)
    sides = _selected_sides(args.side)
    ports = {side: _port_for_side(settings, side) for side in sides}
    ensure_can_interfaces_ready(
        list(ports.values()),
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=args.repair_can,
    )
    arms = {side: _connect(port) for side, port in ports.items()}

    print("elapsed_s side  port  grippers_angle  angle_mm?  grippers_effort  status")
    period_s = 1.0 / args.rate_hz
    started = time.perf_counter()
    next_tick = started
    try:
        while True:
            now = time.perf_counter()
            elapsed_s = now - started
            if elapsed_s > args.duration_s:
                break
            for side, arm in arms.items():
                angle, effort, status = _read_gripper_feedback(arm)
                print(
                    f"{elapsed_s:8.3f} {side:5s} {ports[side]:5s} "
                    f"{angle:15d} {angle / 1000.0:9.3f} "
                    f"{effort:15d}  {status}",
                    flush=True,
                )
            next_tick += period_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        for arm in arms.values():
            disconnect = getattr(arm, "DisconnectPort", None)
            if disconnect is not None:
                disconnect()


if __name__ == "__main__":
    main()
