"""Verify the complete HandUMI calibration chain before recording.

The static preflight checks the local gripper calibration, camera/spatial and
session artifacts, robot-to-table configuration, and the controller-to-TCP
file selected for the requested robot and tracking device.  Unless
``--static-only`` is used, a short guided tracking check then compares both
physical tips at one point and verifies that the table is close to ``z=0``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_path_for_robot_device,
    load_controller_tcp_calibration,
)
from handumi.calibration.deployment import (
    load_deployment_calibration,
    local_calibration_path,
)
from handumi.calibration.spatial import (
    CharucoBoardSpec,
    calibration_hash,
    load_yaml,
    pose7_from_dict,
)
from handumi.config import (
    DEFAULT_RIG_CONFIG,
    load_optional_rig_section,
    load_rig_section,
)
from handumi.feetech.calibration import load_config, user_calibration_path
from handumi.robots.registry import (
    available_robot_names,
    load_robot_config,
)
from handumi.tracking.base import TrackingProvider

DEFAULT_SPATIAL = Path("outputs/calibration/spatial.yaml")
DEFAULT_SESSION = Path("outputs/calibration/session.yaml")
DEFAULT_MAX_TIP_DISTANCE_M = 0.35
DEFAULT_MIRROR_TOLERANCE_M = 0.005
DEFAULT_MAX_SESSION_RMS_MM = 8.0
DEFAULT_MAX_TIP_SEPARATION_MM = 15.0
DEFAULT_MAX_TABLE_Z_MM = 15.0

Status = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class Check:
    status: Status
    name: str
    detail: str


def _add(checks: list[Check], status: Status, name: str, detail: str) -> None:
    checks.append(Check(status, name, detail))


def _device_and_robot(args: argparse.Namespace) -> tuple[str, str]:
    recording = load_optional_rig_section(args.rig_config, "recording")
    device = args.device or str(recording.get("device") or "meta")
    robot = args.robot or str(recording.get("robot") or "")
    if not robot:
        raise SystemExit(
            "Select --robot or set recording.robot in configs/rig.yaml so the "
            "controller-to-TCP calibration can be verified."
        )
    return device, robot


def _verify_grippers(args: argparse.Namespace, checks: list[Check]) -> None:
    try:
        config = load_config(args.rig_config, args.gripper_calibration)
    except (OSError, ValueError, SystemExit) as exc:
        _add(checks, "FAIL", "grippers", str(exc))
        return

    incomplete = [
        side for side in ("left", "right") if not getattr(config, side).is_complete
    ]
    if incomplete:
        _add(
            checks,
            "FAIL",
            "grippers",
            f"incomplete calibration for {', '.join(incomplete)} in "
            f"{args.gripper_calibration}",
        )
        return

    details = []
    for side in ("left", "right"):
        calibration = getattr(config, side)
        span = int(calibration.open_ticks) - int(calibration.closed_ticks)
        if span == 0 or float(calibration.max_width_mm) <= 0:
            _add(
                checks,
                "FAIL",
                "grippers",
                f"{side} has an invalid tick span or maximum width",
            )
            return
        details.append(
            f"{side}: span={span:+d} ticks, max={float(calibration.max_width_mm):.1f} mm"
        )
    _add(checks, "PASS", "grippers", "; ".join(details))


def _verify_spatial_and_session(
    args: argparse.Namespace,
    device: str,
    checks: list[Check],
) -> None:
    try:
        spatial = load_yaml(args.spatial)
    except (OSError, ValueError) as exc:
        _add(checks, "FAIL", "spatial", f"cannot load {args.spatial}: {exc}")
        return
    if spatial.get("kind") != "handumi_spatial_calibration":
        _add(checks, "FAIL", "spatial", f"invalid kind in {args.spatial}")
        return

    try:
        rig_board = CharucoBoardSpec.from_dict(
            load_rig_section(args.rig_config, "spatial_calibration").get("board")
        )
        spatial_board = CharucoBoardSpec.from_dict(spatial.get("board"))
    except (ValueError, SystemExit) as exc:
        _add(checks, "FAIL", "spatial", str(exc))
        return
    if spatial_board != rig_board:
        _add(
            checks,
            "FAIL",
            "spatial",
            "ChArUco board differs from configs/rig.yaml; recalibrate with the configured board",
        )
        return

    cameras = spatial.get("cameras") or {}
    missing_cameras = [
        name
        for name in ("left_wrist", "right_wrist", "workspace")
        if name not in cameras
    ]
    mounts = spatial.get("controller_camera") or {}
    missing_mounts = [side for side in ("left", "right") if side not in mounts]
    if missing_cameras or missing_mounts:
        missing = [
            *(f"camera {name}" for name in missing_cameras),
            *(f"{side} mount" for side in missing_mounts),
        ]
        _add(
            checks, "FAIL", "spatial", f"missing {', '.join(missing)} in {args.spatial}"
        )
        return

    bad_intrinsics = []
    for name in ("left_wrist", "right_wrist", "workspace"):
        error = float((cameras[name] or {}).get("mean_error_px", float("inf")))
        if not np.isfinite(error) or error > args.max_intrinsics_error_px:
            bad_intrinsics.append(f"{name}={error:.3f}px")
    bad_mounts = []
    for side in ("left", "right"):
        rms = float(
            ((mounts[side] or {}).get("metrics") or {}).get(
                "translation_rms_mm", float("inf")
            )
        )
        if not np.isfinite(rms) or rms > args.max_mount_rms_mm:
            bad_mounts.append(f"{side}={rms:.2f}mm")
    if bad_intrinsics or bad_mounts:
        detail = "; ".join(
            filter(
                None,
                [
                    f"intrinsics over limit: {', '.join(bad_intrinsics)}"
                    if bad_intrinsics
                    else "",
                    f"mounts over limit: {', '.join(bad_mounts)}" if bad_mounts else "",
                ],
            )
        )
        _add(checks, "FAIL", "spatial", detail)
    else:
        _add(
            checks,
            "PASS",
            "spatial",
            f"3 camera intrinsics and 2 controller mounts in {args.spatial}",
        )

    try:
        session = load_yaml(args.session)
    except (OSError, ValueError) as exc:
        _add(checks, "FAIL", "session", f"cannot load {args.session}: {exc}")
        return
    if session.get("kind") != "handumi_session_calibration":
        _add(checks, "FAIL", "session", f"invalid kind in {args.session}")
        return
    if session.get("spatial_calibration_sha256") != calibration_hash(spatial):
        _add(
            checks,
            "FAIL",
            "session",
            "spatial hash does not match the selected spatial file",
        )
        return
    session_device = str(session.get("tracking_device") or "meta")
    if session_device != device:
        _add(
            checks,
            "FAIL",
            "session",
            f"calibrated for {session_device}, selected device is {device}",
        )
        return
    table_from_device = session.get("table_from_device") or session.get(
        "table_from_quest"
    )
    if not isinstance(table_from_device, dict):
        _add(checks, "FAIL", "session", "missing or invalid table_from_device pose")
        return
    try:
        pose = pose7_from_dict(table_from_device)
    except (KeyError, TypeError, ValueError):
        _add(checks, "FAIL", "session", "missing or invalid table_from_device pose")
        return
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    rms = float((session.get("metrics") or {}).get("translation_rms_mm", float("inf")))
    if not np.all(np.isfinite(pose)) or not np.isclose(quaternion_norm, 1.0, atol=1e-3):
        _add(
            checks,
            "FAIL",
            "session",
            "table_from_device is not a finite normalized pose",
        )
    elif not np.isfinite(rms) or rms > args.max_session_rms_mm:
        _add(
            checks,
            "FAIL",
            "session",
            f"translation RMS {rms:.2f} mm exceeds {args.max_session_rms_mm:.2f} mm",
        )
    else:
        _add(
            checks,
            "PASS",
            "session",
            f"{device}, translation RMS={rms:.2f} mm, spatial hash matches",
        )

    table_cameras = session.get("table_from_camera") or {}
    if "workspace" not in table_cameras:
        _add(
            checks,
            "WARN",
            "workspace camera",
            "not present in the session; run `handumi calibrate spatial workspace` if it is in use",
        )


def _mirror_expected(left: np.ndarray, device: str) -> np.ndarray:
    expected = np.asarray(left, dtype=float).copy()
    expected[0 if device == "pico" else 1] *= -1.0
    return expected


def _verify_tcp(
    args: argparse.Namespace,
    *,
    device: str,
    robot: str,
    checks: list[Check],
) -> ControllerTcpCalibration | None:
    try:
        robot_config = load_robot_config(robot)
        path, selection = calibration_path_for_robot_device(
            robot,
            device,
            explicit_path=args.controller_tcp_calibration,
        )
    except (KeyError, OSError, ValueError, SystemExit) as exc:
        _add(checks, "FAIL", "controller TCP", str(exc))
        return None
    if not path.exists():
        _add(checks, "FAIL", "controller TCP", f"selected file does not exist: {path}")
        return None

    if args.controller_tcp_calibration is None and robot_config.handumi_gripper:
        expected_name = f"{device}_{robot_config.handumi_gripper}.yaml"
        if path.name != expected_name:
            _add(
                checks,
                "FAIL",
                "TCP selection",
                f"{robot}/{device} selects {path.name}, expected assembly-specific {expected_name}",
            )
        else:
            _add(checks, "PASS", "TCP selection", selection)
    else:
        _add(checks, "PASS", "TCP selection", selection)

    try:
        calibration = load_controller_tcp_calibration(path)
    except (OSError, ValueError, SystemExit) as exc:
        _add(checks, "FAIL", "controller TCP", str(exc))
        return None

    for side, pose in (("left", calibration.left), ("right", calibration.right)):
        if not np.all(np.isfinite(pose)):
            _add(
                checks,
                "FAIL",
                "controller TCP",
                f"{side} pose contains non-finite values",
            )
            return None
        norm = float(np.linalg.norm(pose[3:]))
        if not np.isclose(norm, 1.0, atol=1e-3):
            _add(
                checks,
                "FAIL",
                "controller TCP",
                f"{side} quaternion norm is {norm:.6f}",
            )
            return None

    expected_right = _mirror_expected(calibration.left[:3], device)
    mirror_error_m = float(np.max(np.abs(calibration.right[:3] - expected_right)))
    if mirror_error_m > args.mirror_tolerance_m:
        _add(
            checks,
            "FAIL",
            "TCP mirror",
            f"{device} translation mirror error {mirror_error_m * 1000:.2f} mm exceeds {args.mirror_tolerance_m * 1000:.2f} mm",
        )
    else:
        axis = "x" if device == "pico" else "y"
        _add(
            checks,
            "PASS",
            "TCP mirror",
            f"{axis} sign flip; max error={mirror_error_m * 1000:.2f} mm",
        )

    distances = {
        side: float(np.linalg.norm(pose[:3]))
        for side, pose in (("left", calibration.left), ("right", calibration.right))
    }
    invalid_distance = [
        f"{side}={distance:.3f}m"
        for side, distance in distances.items()
        if distance <= 0.02 or distance > args.max_tip_distance_m
    ]
    if invalid_distance:
        _add(
            checks,
            "FAIL",
            "TCP distance",
            f"outside (0.02, {args.max_tip_distance_m:.3f}] m: {', '.join(invalid_distance)}; widen --max-tip-distance-m only for a genuinely longer tip",
        )
    else:
        _add(
            checks,
            "PASS",
            "TCP distance",
            f"left={distances['left']:.3f} m, right={distances['right']:.3f} m",
        )
    return calibration


def _verify_robot_table(
    args: argparse.Namespace, robot: str, checks: list[Check]
) -> None:
    try:
        path = args.table_calibration or local_calibration_path(
            robot,
            rig_config=args.rig_config,
        )
    except SystemExit as exc:
        _add(checks, "FAIL", "robot/table", str(exc))
        return
    if path is None:
        _add(
            checks,
            "WARN",
            "robot/table",
            f"no lab-local calibration for {robot}; absolute-table replay on this "
            "hardware needs one. Create "
            f"{args.rig_config.parent / 'calibration' / 'table' / 'local' / f'{robot}.yaml'} "
            f"or configure deployment.table_calibrations.{robot} in {args.rig_config}",
        )
        return
    try:
        calibration = load_deployment_calibration(
            path,
            expected_robot=robot,
            profile="local",
        )
    except (OSError, ValueError, SystemExit) as exc:
        _add(checks, "FAIL", "robot/table", f"cannot load {path}: {exc}")
        return
    if calibration.scope != "physical":
        _add(
            checks,
            "FAIL",
            "robot/table",
            f"{path} has scope={calibration.scope!r}; physical verification requires scope: physical",
        )
        return
    if not calibration.verified:
        _add(
            checks,
            "WARN",
            "robot/table",
            f"{path} is present but still marked verified: false",
        )
    else:
        _add(checks, "PASS", "robot/table", str(path))


def verify_static(
    args: argparse.Namespace,
) -> tuple[list[Check], ControllerTcpCalibration | None, str, str]:
    device, robot = _device_and_robot(args)
    checks: list[Check] = []
    _verify_grippers(args, checks)
    _verify_spatial_and_session(args, device, checks)
    calibration = _verify_tcp(args, device=device, robot=robot, checks=checks)
    _verify_robot_table(args, robot, checks)
    return checks, calibration, device, robot


def _print_checks(checks: list[Check]) -> None:
    width = max((len(check.name) for check in checks), default=0)
    for check in checks:
        print(f"[{check.status}] {check.name:<{width}}  {check.detail}")


def _capture_position(
    tracker: TrackingProvider,
    side: str,
    *,
    samples: int,
    interval_s: float,
) -> tuple[np.ndarray, float]:
    positions: list[np.ndarray] = []
    deadline = time.monotonic() + max(3.0, samples * interval_s * 4.0)
    while len(positions) < samples and time.monotonic() < deadline:
        sample = tracker.latest()
        if sample.streaming and getattr(sample, f"{side}_tracked"):
            pose = np.asarray(getattr(sample, f"{side}_tcp_pose"), dtype=float)
            if np.all(np.isfinite(pose[:3])):
                positions.append(pose[:3].copy())
        time.sleep(interval_s)
    if len(positions) < samples:
        raise SystemExit(
            f"Only {len(positions)}/{samples} valid {side} tracking samples were received."
        )
    values = np.stack(positions)
    center = np.median(values, axis=0)
    jitter_mm = float(np.sqrt(np.mean(np.sum((values - center) ** 2, axis=1))) * 1000.0)
    return center, jitter_mm


def _run_live_verification(
    args: argparse.Namespace,
    calibration: ControllerTcpCalibration,
    *,
    device: str,
) -> list[Check]:
    from handumi.scripts.setup.calibrate_spatial import _connect_tracker

    session = load_yaml(args.session)
    table_from_device = session.get("table_from_device") or session.get(
        "table_from_quest"
    )
    if not isinstance(table_from_device, dict):
        raise SystemExit("Session calibration is missing table_from_device.")
    workspace_pose = pose7_from_dict(table_from_device)
    args.device = device
    tracker = _connect_tracker(args, calibration=calibration)
    set_workspace = getattr(tracker, "set_workspace_from_device_pose", None)
    if set_workspace is None:
        tracker.stop()
        raise SystemExit(
            f"{device} tracking provider cannot apply a session calibration."
        )
    set_workspace(workspace_pose, locked=True)

    try:
        input(
            "Touch the LEFT tip to a marked reference point, hold still, then press Enter: "
        )
        left_point, left_jitter = _capture_position(
            tracker, "left", samples=args.samples, interval_s=args.sample_interval_s
        )
        input("Touch the RIGHT tip to the SAME point, hold still, then press Enter: ")
        right_point, right_jitter = _capture_position(
            tracker, "right", samples=args.samples, interval_s=args.sample_interval_s
        )
        input("Touch BOTH tips to the table, hold still, then press Enter: ")
        left_table, left_table_jitter = _capture_position(
            tracker, "left", samples=args.samples, interval_s=args.sample_interval_s
        )
        right_table, right_table_jitter = _capture_position(
            tracker, "right", samples=args.samples, interval_s=args.sample_interval_s
        )
    finally:
        tracker.stop()

    checks: list[Check] = []
    separation_mm = float(np.linalg.norm(left_point - right_point) * 1000.0)
    max_jitter = max(left_jitter, right_jitter)
    if separation_mm > args.max_tip_separation_mm:
        _add(
            checks,
            "FAIL",
            "physical tips",
            f"separation={separation_mm:.2f} mm (limit {args.max_tip_separation_mm:.2f} mm), jitter={max_jitter:.2f} mm",
        )
    else:
        _add(
            checks,
            "PASS",
            "physical tips",
            f"separation={separation_mm:.2f} mm, jitter={max_jitter:.2f} mm",
        )

    z_mm = np.abs(np.array([left_table[2], right_table[2]])) * 1000.0
    max_table_jitter = max(left_table_jitter, right_table_jitter)
    if float(np.max(z_mm)) > args.max_table_z_mm:
        _add(
            checks,
            "FAIL",
            "physical table z",
            f"left={left_table[2] * 1000:.2f} mm, right={right_table[2] * 1000:.2f} mm (limit +/-{args.max_table_z_mm:.2f} mm)",
        )
    else:
        _add(
            checks,
            "PASS",
            "physical table z",
            f"left={left_table[2] * 1000:.2f} mm, right={right_table[2] * 1000:.2f} mm, jitter={max_table_jitter:.2f} mm",
        )
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--robot", choices=available_robot_names())
    parser.add_argument("--device", choices=("meta", "pico"))
    parser.add_argument(
        "--gripper-calibration", type=Path, default=user_calibration_path()
    )
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--controller-tcp-calibration", type=Path)
    parser.add_argument("--table-calibration", type=Path)
    parser.add_argument("--max-intrinsics-error-px", type=float, default=0.8)
    parser.add_argument("--max-mount-rms-mm", type=float, default=8.0)
    parser.add_argument(
        "--max-session-rms-mm", type=float, default=DEFAULT_MAX_SESSION_RMS_MM
    )
    parser.add_argument(
        "--mirror-tolerance-m", type=float, default=DEFAULT_MIRROR_TOLERANCE_M
    )
    parser.add_argument(
        "--max-tip-distance-m", type=float, default=DEFAULT_MAX_TIP_DISTANCE_M
    )
    parser.add_argument(
        "--max-tip-separation-mm", type=float, default=DEFAULT_MAX_TIP_SEPARATION_MM
    )
    parser.add_argument("--max-table-z-mm", type=float, default=DEFAULT_MAX_TABLE_Z_MM)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval-s", type=float, default=0.03)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip the guided live tip/table check.",
    )
    parser.add_argument("--quest-ip")
    parser.add_argument("--tcp-port", type=int)
    parser.add_argument("--sync-port", type=int)
    parser.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos"
    )
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--pico-adb", action="store_true")
    transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.sample_interval_s < 0:
        raise SystemExit("--sample-interval-s cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    checks, calibration, device, robot = verify_static(args)
    print(f"Calibration verification: robot={robot}, device={device}")
    _print_checks(checks)
    failures = [check for check in checks if check.status == "FAIL"]
    if failures:
        raise SystemExit(
            f"Static calibration verification failed ({len(failures)} check(s))."
        )
    if args.static_only:
        warnings = sum(check.status == "WARN" for check in checks)
        print(f"Static calibration verification passed ({warnings} warning(s)).")
        return
    if calibration is None:
        raise SystemExit(
            "No valid controller-to-TCP calibration is available for the live check."
        )

    live_checks = _run_live_verification(args, calibration, device=device)
    _print_checks(live_checks)
    live_failures = [check for check in live_checks if check.status == "FAIL"]
    if live_failures:
        raise SystemExit(
            f"Physical calibration verification failed ({len(live_failures)} check(s))."
        )
    print("Complete calibration verification passed.")


if __name__ == "__main__":
    main()
