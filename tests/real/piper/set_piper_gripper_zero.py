#!/usr/bin/env python3
"""Set the Piper SDK's persistent gripper zero from the current position.

This is a hardware calibration command.  Before confirming it, manually close
each selected gripper to the physical position that should report 0 mm.  The
script does not move the grippers; it asks the Piper controller to store their
current encoder positions as zero.

By default it only prints the current feedback.  To write the zero for both
grippers, run::

    .venv/bin/python tests/real/piper/set_piper_gripper_zero.py \\
        --confirm "SET PIPER GRIPPER ZEROS"

Afterward, use ``print_piper_gripper_feedback.py`` and manually move each
gripper through its range to record the resulting feedback range.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import load_piper_can_settings


CONFIRMATION = "SET PIPER GRIPPER ZEROS"
SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Gripper(s) whose current physical position becomes zero.",
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--settle-s",
        type=float,
        default=1.5,
        help="Delay before and after storing zero, in seconds (default: 1.5).",
    )
    parser.add_argument("--repair-can", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=f'Required to write zero: --confirm "{CONFIRMATION}".',
    )
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


def _read_feedback(arm) -> tuple[int, int]:
    feedback = arm.GetArmGripperMsgs().gripper_state
    return int(feedback.grippers_angle), int(feedback.grippers_effort)


def _print_feedback(label: str, arms: dict[str, object], ports: dict[str, str]) -> None:
    print(label)
    print("side  port  grippers_angle  reported_mm  grippers_effort")
    for side, arm in arms.items():
        angle, effort = _read_feedback(arm)
        print(
            f"{side:5s} {ports[side]:5s} {angle:15d} {angle / 1000.0:11.3f} "
            f"{effort:15d}",
            flush=True,
        )


def _store_current_position_as_zero(arm, effort: int, settle_s: float) -> None:
    """Use the Piper SDK's documented zero-setting sequence without motion."""
    arm.GripperCtrl(0, int(effort), 0x00, 0)
    if settle_s:
        time.sleep(settle_s)
    arm.GripperCtrl(0, int(effort), 0x00, 0xAE)


def main() -> None:
    args = parse_args()
    if args.settle_s < 0.0:
        raise SystemExit("--settle-s must be >= 0.")

    settings = load_piper_can_settings(args.rig_config)
    sides = _selected_sides(args.side)
    ports = {side: _port_for_side(settings, side) for side in sides}
    ensure_can_interfaces_ready(
        list(ports.values()),
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=args.repair_can,
    )

    arms: dict[str, object] = {}
    try:
        for side, port in ports.items():
            arms[side] = _connect(port)
        _print_feedback("Current feedback (no calibration written yet):", arms, ports)

        if args.confirm != CONFIRMATION:
            print(
                "\nNo zero was written. Manually close the selected gripper(s), then "
                f're-run with --confirm "{CONFIRMATION}".',
            )
            return

        print(
            "\nWriting current physical position as persistent zero for: "
            + ", ".join(sides)
            + ".",
            flush=True,
        )
        for side in sides:
            _store_current_position_as_zero(
                arms[side], settings.gripper_effort, args.settle_s
            )
            print(f"Stored zero for {side} ({ports[side]}).", flush=True)
        if args.settle_s:
            time.sleep(args.settle_s)
        _print_feedback("Feedback after storing zero:", arms, ports)
    finally:
        for arm in arms.values():
            disconnect = getattr(arm, "DisconnectPort", None)
            if disconnect is not None:
                disconnect()


if __name__ == "__main__":
    main()
