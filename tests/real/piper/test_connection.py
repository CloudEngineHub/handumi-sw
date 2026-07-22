#!/usr/bin/env python3
"""Minimal hardware check for an AgileX Piper CAN connection.

This is deliberately an executable diagnostic, not a pytest test.  By default
it only opens the CAN connection and prints the arm status and six joint
positions.  Motion is opt-in and requires an exact confirmation string.

Examples::

    .venv/bin/python tests/real/piper/test_connection.py --side right
    .venv/bin/python tests/real/piper/test_connection.py --side right \
        --move-deg 2 --confirm "MOVE RIGHT J6"
    .venv/bin/python tests/real/piper/test_connection.py --side both \
        --stream-seconds 30 --confirm "STREAM BOTH"
    .venv/bin/python tests/real/piper/test_connection.py --side both \
        --exercise-seconds 8 --confirm "EXERCISE BOTH"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready, read_can_status
from handumi.real.piper.driver import ARM_JOINT_COUNT, load_piper_can_settings

PIPER_J6_LIMIT_MDEG = 120_000
PIPER_GRIPPER_MAX_MM = 66.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--rig-config", type=str, default=str(DEFAULT_RIG_CONFIG))
    parser.add_argument(
        "--repair-can",
        action="store_true",
        help="Repair a down/misconfigured CAN interface (may ask for sudo).",
    )
    parser.add_argument(
        "--move-deg",
        type=float,
        default=0.0,
        help="Optional relative move in degrees. Zero is read-only (default).",
    )
    parser.add_argument(
        "--stream-seconds",
        type=float,
        default=0.0,
        help=(
            "Opt-in fixed-position JointCtrl stream. Re-sends each arm's sampled "
            "current joint pose for this duration; 0 disables it (default)."
        ),
    )
    parser.add_argument(
        "--stream-hz",
        type=float,
        default=None,
        help="Command rate for streaming/exercise; default is the Piper teleop rate.",
    )
    parser.add_argument(
        "--exercise-seconds",
        type=float,
        default=0.0,
        help=(
            "Opt-in gentle hardware exercise: oscillate J6 and open/close the "
            "gripper. Zero disables it (default)."
        ),
    )
    parser.add_argument(
        "--exercise-joint-deg",
        type=float,
        default=0.75,
        help="J6 oscillation amplitude for --exercise-seconds (default: 0.75).",
    )
    parser.add_argument(
        "--gripper-closed-mm",
        type=float,
        default=15.0,
        help="Closed gripper width for --exercise-seconds (default: 15).",
    )
    parser.add_argument(
        "--gripper-open-mm",
        type=float,
        default=30.0,
        help="Open gripper width for --exercise-seconds (default: 30).",
    )
    parser.add_argument(
        "--joint",
        type=int,
        choices=range(1, ARM_JOINT_COUNT + 1),
        default=6,
        help="Joint for --move-deg; J6/wrist is the default.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help='Required for motion: e.g. --confirm "MOVE RIGHT J6".',
    )
    return parser.parse_args()


def _sides(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def _port_for_side(settings, side: str) -> str:
    return settings.left_port if side == "left" else settings.right_port


def _read_mdeg(piper) -> np.ndarray:
    joints = piper.GetArmJointMsgs().joint_state
    return np.array(
        [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
        ],
        dtype=np.int64,
    )


def _connect(port: str):
    try:
        from piper_sdk import C_PiperInterface_V2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing piper_sdk. Install it with: ./install.sh --robot piper"
        ) from exc

    piper = C_PiperInterface_V2(port)
    piper.ConnectPort()
    # The SDK receives feedback on a CAN thread; give it a short first sample window.
    time.sleep(0.3)
    return piper


def _enable_joint_control(piper, *, side: str, speed_percent: int) -> None:
    """Match the PiperSdkArm enable retry used by real teleop."""
    deadline = time.monotonic() + 10.0
    while not piper.EnablePiper():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{side}: Piper did not enable within 10 seconds")
        time.sleep(0.02)
    piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)


def _stream_current_positions(
    pipers, *, seconds: float, rate_hz: float, speed_percent: int
) -> None:
    """Match the teleop JointCtrl cadence while holding the sampled joint pose."""
    for side, piper in pipers.items():
        _enable_joint_control(piper, side=side, speed_percent=speed_percent)
    targets = {side: _read_mdeg(piper) for side, piper in pipers.items()}

    period = 1.0 / rate_hz
    deadline = time.perf_counter() + seconds
    next_tick = time.perf_counter()
    ticks = 0
    print(
        f"Streaming sampled joint positions at {rate_hz:g} Hz for {seconds:g} s "
        f"({len(pipers)} arm(s), no intentional motion)."
    )
    while time.perf_counter() < deadline:
        for side, piper in pipers.items():
            # JointCtrl emits J12, J34, and J56: exactly the Piper teleop path.
            piper.JointCtrl(*(int(value) for value in targets[side]))
        ticks += 1
        next_tick += period
        remaining = next_tick - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

    print(f"Completed {ticks} stream ticks ({ticks * len(pipers) * 3} JointCtrl CAN frames).")


def _exercise_gentle_motion(
    pipers,
    *,
    seconds: float,
    rate_hz: float,
    speed_percent: int,
    j6_amplitude_mdeg: int,
    gripper_closed_microm: int,
    gripper_open_microm: int,
    gripper_effort: int,
) -> None:
    """Run one gentle J6 and gripper cycle, then restore the J6 start pose."""
    for side, piper in pipers.items():
        _enable_joint_control(piper, side=side, speed_percent=speed_percent)
    starts = {side: _read_mdeg(piper) for side, piper in pipers.items()}
    for side, start in starts.items():
        if abs(int(start[5])) + j6_amplitude_mdeg > PIPER_J6_LIMIT_MDEG:
            raise RuntimeError(f"{side}: J6 is too close to its limit for this exercise")

    print(
        "Gentle exercise: J6 +/-"
        f"{j6_amplitude_mdeg / 1000.0:g} deg and gripper "
        f"{gripper_closed_microm / 1000.0:g}->{gripper_open_microm / 1000.0:g}"
        f"->{gripper_closed_microm / 1000.0:g} mm over {seconds:g} s."
    )
    period = 1.0 / rate_hz
    deadline = time.perf_counter() + seconds
    next_tick = time.perf_counter()
    ticks = 0
    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            phase = (now - (deadline - seconds)) / seconds
            j6_delta = int(round(j6_amplitude_mdeg * np.sin(2.0 * np.pi * phase)))
            gripper = int(
                round(
                    gripper_closed_microm
                    + (gripper_open_microm - gripper_closed_microm)
                    * 0.5
                    * (1.0 - np.cos(2.0 * np.pi * phase))
                )
            )
            for side, piper in pipers.items():
                target = starts[side].copy()
                target[5] += j6_delta
                piper.JointCtrl(*(int(value) for value in target))
                piper.GripperCtrl(gripper, gripper_effort, 0x01, 0)
            ticks += 1
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        for side, piper in pipers.items():
            piper.JointCtrl(*(int(value) for value in starts[side]))
            piper.GripperCtrl(gripper_closed_microm, gripper_effort, 0x01, 0)
    print(f"Completed {ticks} exercise ticks; J6 returned to its sampled start pose.")


def main() -> None:
    args = parse_args()
    settings = load_piper_can_settings(Path(args.rig_config))
    sides = _sides(args.side)
    ports = [_port_for_side(settings, side) for side in sides]
    ensure_can_interfaces_ready(
        ports,
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=args.repair_can,
    )

    if args.stream_seconds < 0.0 or args.exercise_seconds < 0.0:
        raise SystemExit("--stream-seconds and --exercise-seconds must be >= 0.")
    if args.stream_hz is not None and args.stream_hz <= 0.0:
        raise SystemExit("--stream-hz must be > 0.")
    selected_actions = sum(
        bool(value)
        for value in (args.move_deg, args.stream_seconds, args.exercise_seconds)
    )
    if selected_actions > 1:
        raise SystemExit("Use only one of --move-deg, --stream-seconds, or --exercise-seconds.")
    if not 0.0 < args.exercise_joint_deg <= 2.0:
        raise SystemExit("--exercise-joint-deg must be in (0, 2].")
    if not (
        0.0 <= args.gripper_closed_mm < args.gripper_open_mm <= PIPER_GRIPPER_MAX_MM
    ):
        raise SystemExit(
            f"Expected 0 <= --gripper-closed-mm < --gripper-open-mm <= {PIPER_GRIPPER_MAX_MM:g}."
        )

    if args.move_deg:
        if abs(args.move_deg) > 5.0:
            raise SystemExit("Refusing a move larger than 5 degrees in this diagnostic.")
        expected = f"MOVE {args.side.upper()} J{args.joint}"
        if args.confirm != expected:
            raise SystemExit(
                f"Motion cancelled. Pass --confirm {expected!r} to move the arm."
            )
    elif args.stream_seconds:
        expected = f"STREAM {args.side.upper()}"
        if args.confirm != expected:
            raise SystemExit(
                "Streaming cancelled. It enables joint control and repeatedly sends "
                f"the sampled pose. Pass --confirm {expected!r} to continue."
            )
    elif args.exercise_seconds:
        expected = f"EXERCISE {args.side.upper()}"
        if args.confirm != expected:
            raise SystemExit(
                "Exercise cancelled. It moves J6 and cycles the gripper. "
                f"Pass --confirm {expected!r} to continue."
            )

    pipers = {}
    try:
        for side in sides:
            port = _port_for_side(settings, side)
            piper = _connect(port)
            pipers[side] = piper
            status = piper.GetArmStatus().arm_status
            joints = _read_mdeg(piper)
            print(f"{side}: CONNECTED on {port}")
            print(f"  status: ctrl_mode={status.ctrl_mode}, motion_status={status.motion_status}")
            print(f"  joints (deg): {np.round(joints / 1000.0, 3).tolist()}")

        if args.stream_seconds:
            rate_hz = (
                settings.command_rate_hz
                if args.stream_hz is None
                else args.stream_hz
            )
            _stream_current_positions(
                pipers,
                seconds=args.stream_seconds,
                rate_hz=rate_hz,
                speed_percent=settings.speed_percent,
            )
            for port in ports:
                status = read_can_status(port)
                print(f"{port}: CAN state after stream={status.state or 'unknown'}")
            return

        if args.exercise_seconds:
            rate_hz = (
                settings.command_rate_hz
                if args.stream_hz is None
                else args.stream_hz
            )
            _exercise_gentle_motion(
                pipers,
                seconds=args.exercise_seconds,
                rate_hz=rate_hz,
                speed_percent=settings.speed_percent,
                j6_amplitude_mdeg=int(round(args.exercise_joint_deg * 1000.0)),
                gripper_closed_microm=int(round(args.gripper_closed_mm * 1000.0)),
                gripper_open_microm=int(round(args.gripper_open_mm * 1000.0)),
                gripper_effort=settings.gripper_effort,
            )
            for port in ports:
                status = read_can_status(port)
                print(f"{port}: CAN state after exercise={status.state or 'unknown'}")
            return

        if not args.move_deg:
            print("Read-only connection test passed; no motion command was sent.")
            return

        for side, piper in pipers.items():
            _enable_joint_control(piper, side=side, speed_percent=10)
            start = _read_mdeg(piper)
            target = start.copy()
            target[args.joint - 1] += int(round(args.move_deg * 1000.0))
            print(f"{side}: moving J{args.joint} by {args.move_deg:+.2f} deg at 10% speed")
            piper.JointCtrl(*(int(value) for value in target))
            time.sleep(1.0)
            print(f"  joints after (deg): {np.round(_read_mdeg(piper) / 1000.0, 3).tolist()}")
    finally:
        for piper in pipers.values():
            disconnect = getattr(piper, "DisconnectPort", None)
            if disconnect is not None:
                disconnect()


if __name__ == "__main__":
    main()
