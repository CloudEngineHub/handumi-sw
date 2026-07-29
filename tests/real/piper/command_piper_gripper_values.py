#!/usr/bin/env python3
"""Send direct Piper gripper commands and print feedback.

This bypasses HandUMI gripper conversion and teleop.  It only sends Piper SDK
``GripperCtrl`` targets to the selected gripper(s), then prints the raw
feedback so the physical opening can be compared side by side.

Examples::

    # Dry run: show configured ports and current feedback, but do not move.
    .venv/bin/python tests/real/piper/command_piper_gripper_values.py --targets-mm 0,66

    # Send 0 mm then 66 mm to both grippers.
    .venv/bin/python tests/real/piper/command_piper_gripper_values.py \\
        --targets-mm 0,66 --confirm "COMMAND PIPER GRIPPERS"

    # Send normalized fractions of piper.yaml gripper_max_width_m.
    .venv/bin/python tests/real/piper/command_piper_gripper_values.py \\
        --targets-normalized 0,0.25,0.5,0.75,1 \\
        --confirm "COMMAND PIPER GRIPPERS"

    # If the left gripper really needs its larger observed range, test it alone.
    .venv/bin/python tests/real/piper/command_piper_gripper_values.py \\
        --side left --targets-mm 0,66,80,96,103 \\
        --allow-over-configured-max --confirm "COMMAND PIPER GRIPPERS"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import load_piper_can_settings
from handumi.robots.registry import load_robot_config


CONFIRMATION = "COMMAND PIPER GRIPPERS"
SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--robot", default="piper")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--targets-um",
        default=None,
        help="Comma-separated Piper gripper targets in micrometers, e.g. 0,66000.",
    )
    target_group.add_argument(
        "--targets-mm",
        default=None,
        help="Comma-separated Piper gripper targets in millimeters, e.g. 0,66.",
    )
    target_group.add_argument(
        "--targets-normalized",
        default=None,
        help=(
            "Comma-separated fractions of piper.yaml gripper_max_width_m, "
            "e.g. 0,0.5,1."
        ),
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=2.0,
        help="Seconds to wait after each command before the after-feedback sample.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.3,
        help="Seconds to wait after connecting before the first feedback sample.",
    )
    parser.add_argument(
        "--effort",
        type=int,
        default=None,
        help="Override Piper gripper effort. Defaults to rig.yaml Piper setting.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Repeat command cadence while holding each target.",
    )
    parser.add_argument(
        "--allow-over-configured-max",
        action="store_true",
        help=(
            "Allow targets above piper.yaml gripper_max_width_m. Use this only "
            "for careful hardware diagnostics."
        ),
    )
    parser.add_argument("--repair-can", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=f'Required to move grippers: --confirm "{CONFIRMATION}".',
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
    return arm


def _enable(arm, side: str, speed_percent: int) -> None:
    deadline = time.monotonic() + 10.0
    while not arm.EnablePiper():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{side}: Piper did not enable within 10 seconds")
        time.sleep(0.02)
    arm.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)


def _read_feedback(arm) -> tuple[int, int]:
    feedback = arm.GetArmGripperMsgs().gripper_state
    return int(feedback.grippers_angle), int(feedback.grippers_effort)


def _parse_csv_floats(raw: str, flag_name: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise SystemExit(f"{flag_name} contains a non-numeric value: {item!r}") from exc
    if not values:
        raise SystemExit(f"{flag_name} must include at least one value.")
    return values


def _targets_microm(args: argparse.Namespace, configured_max_um: int) -> list[int]:
    if args.targets_um is not None:
        return [int(round(value)) for value in _parse_csv_floats(args.targets_um, "--targets-um")]
    if args.targets_mm is not None:
        return [
            int(round(value * 1000.0))
            for value in _parse_csv_floats(args.targets_mm, "--targets-mm")
        ]
    if args.targets_normalized is not None:
        return [
            int(round(value * configured_max_um))
            for value in _parse_csv_floats(
                args.targets_normalized, "--targets-normalized"
            )
        ]
    return [0, configured_max_um]


def _validate_args(args: argparse.Namespace, targets_um: list[int], configured_max_um: int) -> None:
    if args.hold_s < 0.0:
        raise SystemExit("--hold-s must be >= 0.")
    if args.settle_s < 0.0:
        raise SystemExit("--settle-s must be >= 0.")
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be > 0.")
    if any(target < 0 for target in targets_um):
        raise SystemExit("Negative gripper targets are not allowed.")
    if not args.allow_over_configured_max:
        oversized = [target for target in targets_um if target > configured_max_um]
        if oversized:
            max_mm = configured_max_um / 1000.0
            raise SystemExit(
                "Target exceeds configured piper.yaml gripper_max_width_m "
                f"({max_mm:.3f} mm). Re-run with --allow-over-configured-max "
                "for this diagnostic only."
            )


def _print_sample(
    *,
    label: str,
    side: str,
    port: str,
    target_um: int | None,
    angle: int,
    effort: int,
) -> None:
    target = "none" if target_um is None else f"{target_um:7d}"
    target_mm = "" if target_um is None else f"{target_um / 1000.0:8.3f}"
    print(
        f"{label:6s} {side:5s} {port:5s} {target:>7s} {target_mm:>8s} "
        f"{angle:15d} {angle / 1000.0:11.3f} {effort:15d}",
        flush=True,
    )


def _hold_target(arms: dict[str, object], target_um: int, effort: int, hold_s: float, rate_hz: float) -> None:
    period_s = 1.0 / rate_hz
    deadline = time.perf_counter() + hold_s
    while True:
        for arm in arms.values():
            arm.GripperCtrl(int(target_um), int(effort), 0x01, 0)
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        time.sleep(min(period_s, remaining))


def main() -> None:
    args = parse_args()
    settings = load_piper_can_settings(args.rig_config)
    robot_config = load_robot_config(args.robot)
    configured_max_um = int(round(robot_config.gripper_max_width_m * 1_000_000.0))
    targets_um = _targets_microm(args, configured_max_um)
    _validate_args(args, targets_um, configured_max_um)

    sides = _selected_sides(args.side)
    ports = {side: _port_for_side(settings, side) for side in sides}
    effort = settings.gripper_effort if args.effort is None else int(args.effort)

    print(
        f"Configured max: {configured_max_um} um "
        f"({configured_max_um / 1000.0:.3f} mm). Effort: {effort}."
    )
    print("Targets um:", ", ".join(str(value) for value in targets_um))

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
        if args.settle_s:
            time.sleep(args.settle_s)

        print(
            "phase  side  port  target_um target_mm  grippers_angle  reported_mm  grippers_effort"
        )
        for side, arm in arms.items():
            angle, observed_effort = _read_feedback(arm)
            _print_sample(
                label="before",
                side=side,
                port=ports[side],
                target_um=None,
                angle=angle,
                effort=observed_effort,
            )

        if args.confirm != CONFIRMATION:
            print(
                "\nDry run only. Re-run with "
                f'--confirm "{CONFIRMATION}" to send GripperCtrl commands.'
            )
            return

        for side, arm in arms.items():
            _enable(arm, side, settings.speed_percent)

        for target_um in targets_um:
            print(f"\nCommanding {target_um} um ({target_um / 1000.0:.3f} mm).")
            _hold_target(arms, target_um, effort, args.hold_s, args.rate_hz)
            for side, arm in arms.items():
                angle, observed_effort = _read_feedback(arm)
                _print_sample(
                    label="after",
                    side=side,
                    port=ports[side],
                    target_um=target_um,
                    angle=angle,
                    effort=observed_effort,
                )
    finally:
        for arm in arms.values():
            disconnect = getattr(arm, "DisconnectPort", None)
            if disconnect is not None:
                disconnect()


if __name__ == "__main__":
    main()
