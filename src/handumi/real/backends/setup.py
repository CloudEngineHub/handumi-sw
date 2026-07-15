"""Robot-specific setup steps behind one wizard-facing registry."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from handumi.real.can_setup import (
    ensure_can_fd_interfaces_ready,
    ensure_can_interfaces_ready,
    run_openarm_can_wizard,
    run_piper_can_wizard,
)
from handumi.real.openarm_can import load_openarm_settings, require_openarm_can
from handumi.real.piper_can import load_piper_can_settings
from handumi.robots.registry import load_robot_config


@dataclass(frozen=True)
class RobotSetupOptions:
    robot: str
    rig_config: Path
    bitrate: int
    dbitrate: int
    restart_ms: int
    skip_can_map: bool
    skip_can_repair: bool
    skip_motor_check: bool
    calibrate_openarm_zero: bool


def run_robot_setup(options: RobotSetupOptions) -> None:
    handlers = {
        "piper": _setup_piper,
        "openarmv1": _setup_openarm,
    }
    try:
        handler = handlers[options.robot]
    except KeyError as exc:
        raise SystemExit(f"No setup provider for {options.robot!r}.") from exc
    handler(options)


def _setup_piper(options: RobotSetupOptions) -> None:
    if not options.skip_can_map:
        run_piper_can_wizard(
            rig_config=options.rig_config,
            bitrate=options.bitrate,
            restart_ms=options.restart_ms,
        )
    settings = load_piper_can_settings(options.rig_config)
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=not options.skip_can_repair,
    )


def _setup_openarm(options: RobotSetupOptions) -> None:
    if shutil.which("openarm-can-cli") is None:
        raise SystemExit(
            "Missing openarm-can-cli. Install libopenarm-can-dev and "
            "openarm-can-utils from the official OpenArm repository."
        )
    try:
        require_openarm_can()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not options.skip_can_map:
        run_openarm_can_wizard(
            rig_config=options.rig_config,
            bitrate=options.bitrate,
            dbitrate=options.dbitrate,
        )
    settings = load_openarm_settings(
        options.rig_config, load_robot_config("openarmv1").real_options
    )
    ensure_can_fd_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        dbitrate=settings.dbitrate,
        repair=not options.skip_can_repair,
    )
    if not options.skip_motor_check:
        for side, port in (
            ("right", settings.right_port),
            ("left", settings.left_port),
        ):
            _check_openarm_motors(side, port)
    if options.calibrate_openarm_zero:
        answer = input(
            "OpenArm zero calibration moves joints to mechanical stops. "
            "Workspace clear and emergency stop ready? Type CALIBRATE: "
        ).strip()
        if answer != "CALIBRATE":
            raise SystemExit(
                "OpenArm zero calibration cancelled; no motors were moved."
            )
        for side, port in (
            ("right_arm", settings.right_port),
            ("left_arm", settings.left_port),
        ):
            subprocess.run(
                [
                    "openarm-can-zero-position-calibration",
                    "--canport",
                    port,
                    "--arm-side",
                    side,
                    "--robot-version",
                    "v1",
                ],
                check=True,
            )


def _check_openarm_motors(side: str, port: str) -> None:
    """Read motor parameters without enabling motor output."""
    result = subprocess.run(
        [
            "openarm-can-cli",
            "-i",
            port,
            "show_param",
            "--id",
            "1,2,3,4,5,6,7,8",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    responded = output.count("MOTOR ID:")
    if result.returncode != 0 or responded != 8 or "NO RESPONSE FROM MOTOR" in output:
        raise SystemExit(
            f"OpenArm {side} motor diagnostic failed on {port}:\n{output.strip()}"
        )
    print(f"OpenArm {side}: J1-J8 responded on {port}.")


__all__ = ["RobotSetupOptions", "run_robot_setup"]
