#!/usr/bin/env python3
"""Inspect and optionally replay the HandUMI-to-Piper gripper conversion.

This is a hardware diagnostic, not a pytest test.  It reproduces the gripper
path used by real teleop:

    Feetech encoder ticks -> HandUMI normalized opening -> Piper width in mm
    -> Piper SDK command in micrometres.

The default ``observe`` mode only reads the HandUMI encoders and prints the
commands that teleop would send; it never opens a Piper gripper.  ``command``
uses the same conversion and streams it to Piper after an explicit
confirmation.  The arm joints are held at their current feedback positions;
this script never homes or follows teleop arm targets.

Examples::

    # Read both HandUMI grippers and show their equivalent Piper commands.
    .venv/bin/python tests/real/piper/debug_handumi_to_piper_gripper.py

    # Check Piper's configured 66 mm range, independent of HandUMI encoders.
    .venv/bin/python tests/real/piper/debug_handumi_to_piper_gripper.py \
        --fixed-openings 0,0.25,0.5,0.75,1 --mode command \
        --confirm "RUN PIPER GRIPPER DEBUG"

    # Replay the live HandUMI opening conversion onto the real Piper grippers.
    .venv/bin/python tests/real/piper/debug_handumi_to_piper_gripper.py \
        --mode command --duration-s 30 --confirm "RUN PIPER GRIPPER DEBUG" \
        --csv /tmp/handumi-to-piper-gripper.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Iterable

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    assert_calibrated,
    load_config,
    user_calibration_path,
)
from handumi.feetech.bus import FeetechBus
from handumi.feetech.gripper import _EncoderUnwrapper
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import (
    PiperCanEnvironment,
    load_piper_can_settings,
)
from handumi.robots.registry import load_robot_config


LOG = logging.getLogger("handumi.piper_gripper_debug")
CONFIRMATION = "RUN PIPER GRIPPER DEBUG"
SIDES = ("left", "right")
DEFAULT_SCAN_IDS = tuple(range(1, 9))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("observe", "command"),
        default="observe",
        help="Observe only (default), or send the computed widths to Piper.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Which side(s) to print and command.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=15.0,
        help="Live-HandUMI observation/replay duration in seconds.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=20.0,
        help="Feetech read and Piper target-update cadence in live mode.",
    )
    parser.add_argument(
        "--print-rate-hz",
        type=float,
        default=5.0,
        help="Maximum console table rate in live mode.",
    )
    parser.add_argument(
        "--fixed-openings",
        type=_parse_openings,
        default=None,
        metavar="FRACTION[,FRACTION...]",
        help=(
            "Bypass HandUMI and command/preview these normalized openings, for "
            "example 0,0.5,1. Useful for isolating Piper max width."
        ),
    )
    parser.add_argument(
        "--step-duration-s",
        type=float,
        default=2.0,
        help="Hold time for each --fixed-openings value in command mode.",
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local Feetech wiring and Piper CAN configuration.",
    )
    parser.add_argument(
        "--feetech-port",
        default=None,
        help="Temporarily use one shared Feetech port instead of rig.yaml.",
    )
    parser.add_argument(
        "--piper-max-width-m",
        type=float,
        default=None,
        help=(
            "Override piper.yaml gripper_max_width_m for this run only. "
            "Does not edit configuration."
        ),
    )
    parser.add_argument(
        "--repair-can",
        action="store_true",
        help="Allow CAN repair with sudo if configured interfaces are unavailable.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f'Required for --mode command: --confirm "{CONFIRMATION}".',
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with raw ticks and every conversion stage.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first Feetech read error instead of continuing per side.",
    )
    return parser.parse_args(argv)


def _parse_openings(value: str) -> tuple[float, ...]:
    try:
        openings = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("openings must be comma-separated numbers") from exc
    if not openings or any(not 0.0 <= opening <= 1.0 for opening in openings):
        raise argparse.ArgumentTypeError("every opening must be in the range [0, 1]")
    return openings


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be > 0.")
    if args.rate_hz <= 0.0 or args.print_rate_hz <= 0.0:
        raise SystemExit("--rate-hz and --print-rate-hz must be > 0.")
    if args.step_duration_s <= 0.0:
        raise SystemExit("--step-duration-s must be > 0.")
    if args.piper_max_width_m is not None and args.piper_max_width_m <= 0.0:
        raise SystemExit("--piper-max-width-m must be > 0.")
    if args.mode == "command" and args.confirm != CONFIRMATION:
        raise SystemExit(
            "Real gripper motion is disabled. Clear the workspace, then re-run with "
            f'--confirm "{CONFIRMATION}".'
        )


def _with_feetech_port(config: FeetechConfig, port: str | None) -> FeetechConfig:
    if port is None:
        return config
    return FeetechConfig(
        port=port,
        baudrate=config.baudrate,
        protocol_version=config.protocol_version,
        left=config.left,
        right=config.right,
    )


def _side_port(config: FeetechConfig, calibration: GripperCalibration) -> str:
    port = calibration.port or config.port
    if not port:
        raise ValueError(
            "Feetech port is not configured. Set a shared `port` or per-side "
            "`left.port` / `right.port`."
        )
    return port


def _bus_by_port(config: FeetechConfig) -> dict[str, FeetechBus]:
    ports = {
        _side_port(config, config.left),
        _side_port(config, config.right),
    }
    return {
        port: FeetechBus(
            port=port,
            baudrate=config.baudrate,
            protocol_version=config.protocol_version,
        )
        for port in ports
    }


def _piper_command(normalized: float, max_width_mm: float) -> tuple[float, int]:
    """Match PiperBackend.write plus PiperCanEnvironment width conversion."""
    opening = min(1.0, max(0.0, float(normalized)))
    width_mm = opening * max_width_mm
    return width_mm, int(round(max(0.0, width_mm) * 1000.0))


def _side_piper_max_width_mm(
    widths_mm: float | dict[str, float],
    side: str,
) -> float:
    return float(widths_mm[side] if isinstance(widths_mm, dict) else widths_mm)


def _row_from_calibration(
    *,
    side: str,
    calibration: GripperCalibration,
    normalized: float,
    piper_max_width_mm: float,
    source: str,
    ticks: int | None,
) -> dict[str, object]:
    piper_width_mm, piper_microm = _piper_command(normalized, piper_max_width_mm)
    handumi_width_mm = (
        None
        if calibration.max_width_mm is None
        else normalized * calibration.max_width_mm
    )
    span_ticks = (
        None
        if calibration.closed_ticks is None or calibration.open_ticks is None
        else calibration.open_ticks - calibration.closed_ticks
    )
    raw_normalized = (
        None
        if ticks is None or span_ticks in (None, 0) or calibration.closed_ticks is None
        else (ticks - calibration.closed_ticks) / span_ticks
    )
    range_status = "unknown"
    if raw_normalized is not None:
        if raw_normalized < 0.0:
            range_status = "below_closed"
        elif raw_normalized > 1.0:
            range_status = "beyond_open"
        else:
            range_status = "within_calibration"
    return {
        "source": source,
        "side": side,
        "ticks": ticks,
        "closed_ticks": calibration.closed_ticks,
        "open_ticks": calibration.open_ticks,
        "span_ticks": span_ticks,
        "raw_normalized": raw_normalized,
        "range_status": range_status,
        "normalized": normalized,
        "handumi_width_mm": handumi_width_mm,
        "handumi_max_width_mm": calibration.max_width_mm,
        "piper_width_mm": piper_width_mm,
        "piper_microm": piper_microm,
        "piper_max_width_mm": piper_max_width_mm,
    }


def _error_row(
    *,
    side: str,
    calibration: GripperCalibration,
    piper_max_width_mm: float,
    source: str,
    error: str,
) -> dict[str, object]:
    return {
        **_row_from_calibration(
            side=side,
            calibration=calibration,
            normalized=0.0,
            piper_max_width_mm=piper_max_width_mm,
            source=source,
            ticks=None,
        ),
        "error": error,
    }


def _selected_sides(args: argparse.Namespace) -> tuple[str, ...]:
    return SIDES if args.side == "both" else (args.side,)


def _rows_from_live_reading(
    buses: dict[str, FeetechBus],
    unwrappers: dict[str, _EncoderUnwrapper],
    config: FeetechConfig,
    piper_max_width_mm: float | dict[str, float],
    *,
    fail_fast: bool,
    sides: Iterable[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side in sides:
        calibration = getattr(config, side)
        try:
            bus = buses[_side_port(config, calibration)]
            ticks = unwrappers[side](bus.read_position(calibration.servo_id))
            rows.append(
                _row_from_calibration(
                    side=side,
                    calibration=calibration,
                    normalized=calibration.normalized_width(ticks),
                    piper_max_width_mm=_side_piper_max_width_mm(
                        piper_max_width_mm, side
                    ),
                    source="feetech",
                    ticks=ticks,
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic should keep running.
            if fail_fast:
                raise
            rows.append(
                _error_row(
                    side=side,
                    calibration=calibration,
                    piper_max_width_mm=_side_piper_max_width_mm(
                        piper_max_width_mm, side
                    ),
                    source="feetech-error",
                    error=str(exc),
                )
            )
    return rows


def _rows_from_fixed_opening(
    opening: float,
    config: FeetechConfig,
    piper_max_width_mm: float | dict[str, float],
    *,
    sides: Iterable[str],
) -> list[dict[str, object]]:
    return [
        _row_from_calibration(
            side=side,
            calibration=getattr(config, side),
            normalized=opening,
            piper_max_width_mm=_side_piper_max_width_mm(
                piper_max_width_mm, side
            ),
            source="fixed",
            ticks=None,
        )
        for side in sides
    ]


def _log_rows(rows: Iterable[dict[str, object]]) -> None:
    for row in rows:
        if row.get("error"):
            LOG.warning("%s %s read failed: %s", row["source"], row["side"], row["error"])
            continue
        ticks = "-" if row["ticks"] is None else str(row["ticks"])
        handumi_width_mm = row["handumi_width_mm"]
        handumi_max_width_mm = row["handumi_max_width_mm"]
        handumi = (
            "unavailable"
            if handumi_width_mm is None or handumi_max_width_mm is None
            else f"{float(handumi_width_mm):.2f}/{float(handumi_max_width_mm):.2f} mm"
        )
        LOG.info(
            "%s %s ticks=%s raw=%s normalized=%.4f %s HandUMI=%s -> Piper=%.2f/%.2f mm (%d um)",
            row["source"],
            row["side"],
            ticks,
            (
                "-"
                if row["raw_normalized"] is None
                else f"{float(row['raw_normalized']):.4f}"
            ),
            float(row["normalized"]),
            row["range_status"],
            handumi,
            float(row["piper_width_mm"]),
            float(row["piper_max_width_mm"]),
            int(row["piper_microm"]),
        )


def _csv_writer(path: Path | None):
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    columns = [
        "elapsed_s",
        "source",
        "side",
        "ticks",
        "closed_ticks",
        "open_ticks",
        "span_ticks",
        "raw_normalized",
        "range_status",
        "normalized",
        "handumi_width_mm",
        "handumi_max_width_mm",
        "piper_width_mm",
        "piper_microm",
        "piper_max_width_mm",
        "commanded",
        "error",
    ]
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    return handle, writer


def _write_rows(writer, rows: Iterable[dict[str, object]], *, elapsed_s: float, commanded: bool) -> None:
    if writer is None:
        return
    for row in rows:
        writer.writerow(
            {
                "elapsed_s": f"{elapsed_s:.6f}",
                **row,
                "commanded": commanded,
            }
        )


def _open_feetech_buses(config: FeetechConfig) -> dict[str, FeetechBus]:
    buses = _bus_by_port(config)
    for bus in buses.values():
        bus.open()
    for port, bus in buses.items():
        seen = bus.scan(DEFAULT_SCAN_IDS)
        LOG.info(
            "Feetech scan on %s saw IDs: %s",
            port,
            ", ".join(str(servo_id) for servo_id in seen) if seen else "none",
        )
    return buses


def _close_feetech_buses(buses: dict[str, FeetechBus]) -> None:
    for bus in buses.values():
        bus.close()


def _open_piper_environment(args: argparse.Namespace, piper_config) -> PiperCanEnvironment:
    settings = load_piper_can_settings(args.rig_config, piper_config.real)
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=args.repair_can,
    )
    environment = PiperCanEnvironment(settings)
    environment.connect()
    # PiperJointStreamer seeds its targets from feedback, so arm targets begin at
    # the current pose. This provides the driver required for gripper commands
    # without a home trajectory or teleop arm movement.
    environment.start_streaming_current_pose()
    return environment


def _resolved_piper_max_widths(
    environment: PiperCanEnvironment | None,
    fallback_max_width_mm: float,
) -> float | dict[str, float]:
    if environment is None:
        return fallback_max_width_mm
    return {
        side: environment.gripper_ranges.get(side).max_mm
        if side in environment.gripper_ranges
        else fallback_max_width_mm
        for side in SIDES
    }


def _send_rows(environment: PiperCanEnvironment, rows: Iterable[dict[str, object]]) -> None:
    environment.set_gripper_widths_mm(
        {str(row["side"]): float(row["piper_width_mm"]) for row in rows}
    )
    environment.raise_if_failed()


def _run_fixed(args: argparse.Namespace, config: FeetechConfig, piper_max_width_mm: float) -> None:
    piper_config = load_robot_config("piper") if args.mode == "command" else None
    environment = (
        _open_piper_environment(args, piper_config) if piper_config is not None else None
    )
    resolved_widths = _resolved_piper_max_widths(environment, piper_max_width_mm)
    handle, writer = _csv_writer(args.csv)
    started = time.perf_counter()
    try:
        for opening in args.fixed_openings:
            rows = _rows_from_fixed_opening(
                opening,
                config,
                resolved_widths,
                sides=_selected_sides(args),
            )
            _log_rows(rows)
            _write_rows(
                writer,
                rows,
                elapsed_s=time.perf_counter() - started,
                commanded=environment is not None,
            )
            if environment is not None:
                _send_rows(environment, rows)
                time.sleep(args.step_duration_s)
    finally:
        if environment is not None:
            environment.close()
        if handle is not None:
            handle.close()


def _run_live(args: argparse.Namespace, config: FeetechConfig, piper_max_width_mm: float) -> None:
    piper_config = load_robot_config("piper") if args.mode == "command" else None
    environment = (
        _open_piper_environment(args, piper_config) if piper_config is not None else None
    )
    resolved_widths = _resolved_piper_max_widths(environment, piper_max_width_mm)
    handle, writer = _csv_writer(args.csv)
    period_s = 1.0 / args.rate_hz
    print_period_s = 1.0 / args.print_rate_hz
    next_print = 0.0
    started = time.perf_counter()
    try:
        buses = _open_feetech_buses(config)
        unwrappers = {side: _EncoderUnwrapper() for side in SIDES}
        try:
            while True:
                elapsed_s = time.perf_counter() - started
                if elapsed_s > args.duration_s:
                    break
                rows = _rows_from_live_reading(
                    buses,
                    unwrappers,
                    config,
                    resolved_widths,
                    fail_fast=args.fail_fast,
                    sides=_selected_sides(args),
                )
                commandable_rows = [row for row in rows if not row.get("error")]
                if environment is not None and commandable_rows:
                    _send_rows(environment, commandable_rows)
                if elapsed_s >= next_print:
                    _log_rows(rows)
                    next_print += print_period_s
                _write_rows(
                    writer,
                    rows,
                    elapsed_s=elapsed_s,
                    commanded=environment is not None and bool(commandable_rows),
                )
                remaining = (
                    started
                    + (int(elapsed_s / period_s) + 1) * period_s
                    - time.perf_counter()
                )
                if remaining > 0.0:
                    time.sleep(remaining)
        finally:
            _close_feetech_buses(buses)
    finally:
        if environment is not None:
            environment.close()
        if handle is not None:
            handle.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    config = _with_feetech_port(load_config(args.rig_config), args.feetech_port)
    if args.fixed_openings is None:
        assert_calibrated(config, source=user_calibration_path())
    # The gripper limit lives in piper.yaml.  Do not call load_embodiment()
    # here: it constructs the full URDF/JAX model, which is unnecessary for a
    # hardware gripper diagnostic and can reserve all available GPU memory.
    configured_width_m = load_robot_config("piper").gripper_max_width_m
    piper_max_width_m = args.piper_max_width_m or configured_width_m
    piper_max_width_mm = piper_max_width_m * 1000.0
    LOG.info(
        "Mode=%s; Piper max width=%.3f m (%.1f mm)%s.",
        args.mode,
        piper_max_width_m,
        piper_max_width_mm,
        " [temporary override]" if args.piper_max_width_m is not None else "",
    )
    for side in SIDES:
        calibration = getattr(config, side)
        if calibration.is_complete:
            LOG.info(
                "HandUMI %s calibration: closed=%d open=%d span=%d ticks, max=%.2f mm.",
                side,
                calibration.closed_ticks,
                calibration.open_ticks,
                calibration.open_ticks - calibration.closed_ticks,
                calibration.max_width_mm,
            )
        else:
            LOG.info(
                "HandUMI %s calibration is incomplete; fixed Piper tests remain available.",
                side,
            )
    try:
        if args.fixed_openings is not None:
            _run_fixed(args, config, piper_max_width_mm)
        else:
            _run_live(args, config, piper_max_width_mm)
    except KeyboardInterrupt:
        LOG.info("Stopped by user.")


if __name__ == "__main__":
    main()
