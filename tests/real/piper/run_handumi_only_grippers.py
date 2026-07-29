#!/usr/bin/env python3
"""Drive only the Piper grippers from the calibrated HandUMI grippers.

Unlike full teleoperation, this diagnostic holds both arm joint targets at
their current feedback pose. Each Piper controller is queried for its own
gripper range, so the same normalized HandUMI opening can map to different
    left and right command widths.

Transient Feetech read failures are isolated per side. The affected Piper
keeps receiving its last valid target while this script retries the encoder;
the other side continues updating normally.

Example::

    .venv/bin/python tests/real/piper/run_handumi_only_grippers.py \
        --confirm "RUN HANDUMI PIPER GRIPPERS"
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    assert_calibrated,
    load_config,
    user_calibration_path,
)
from handumi.feetech.gripper import _EncoderUnwrapper
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import PiperCanEnvironment, load_piper_can_settings
from handumi.robots.registry import load_robot_config


LOG = logging.getLogger("handumi.grippers_only")
CONFIRMATION = "RUN HANDUMI PIPER GRIPPERS"
SIDES = ("left", "right")
DEFAULT_SCAN_IDS = tuple(range(1, 9))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Run time in seconds.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=20.0,
        help="HandUMI read/target update rate.",
    )
    parser.add_argument(
        "--print-rate-hz",
        type=float,
        default=5.0,
        help="Maximum status log rate.",
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local Feetech and Piper wiring configuration.",
    )
    parser.add_argument(
        "--repair-can",
        action="store_true",
        help="Allow CAN repair with sudo if an interface is unavailable.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f'Required to move grippers: --confirm "{CONFIRMATION}".',
    )
    return parser.parse_args(argv)


def _port(config: FeetechConfig, calibration: GripperCalibration) -> str:
    port = calibration.port or config.port
    if not port:
        raise SystemExit(
            "Missing Feetech port; configure a shared or per-side port in rig.yaml."
        )
    return port


def _open_buses(config: FeetechConfig) -> dict[str, FeetechBus]:
    buses = {
        port: FeetechBus(
            port=port,
            baudrate=config.baudrate,
            protocol_version=config.protocol_version,
        )
        for port in {_port(config, config.left), _port(config, config.right)}
    }
    for port, bus in buses.items():
        bus.open()
        seen = bus.scan(DEFAULT_SCAN_IDS)
        LOG.info(
            "Feetech %s IDs: %s",
            port,
            ", ".join(map(str, seen)) if seen else "none",
        )
    return buses


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be > 0.")
    if args.rate_hz <= 0.0 or args.print_rate_hz <= 0.0:
        raise SystemExit("--rate-hz and --print-rate-hz must be > 0.")
    if args.confirm != CONFIRMATION:
        raise SystemExit(
            "Real gripper motion is disabled. Clear the workspace, then re-run "
            f'with --confirm "{CONFIRMATION}".'
        )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    feetech = load_config(args.rig_config)
    assert_calibrated(feetech, source=user_calibration_path())
    robot = load_robot_config("piper")
    fallback_max_width_mm = robot.gripper_max_width_m * 1000.0
    settings = load_piper_can_settings(args.rig_config, robot.real)
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=args.repair_can,
    )

    environment = PiperCanEnvironment(settings)
    buses: dict[str, FeetechBus] = {}
    try:
        environment.connect()
        environment.start_streaming_current_pose()
        buses = _open_buses(feetech)
        unwrappers = {side: _EncoderUnwrapper() for side in SIDES}
        period_s = 1.0 / args.rate_hz
        print_period_s = 1.0 / args.print_rate_hz
        started = time.perf_counter()
        next_print = started
        next_tick = started
        last_openings: dict[str, float] = {}
        last_ticks: dict[str, int] = {}
        last_targets: dict[str, int] = {}
        failed_reads = {side: 0 for side in SIDES}
        last_errors = {side: "" for side in SIDES}
        while time.perf_counter() - started < args.duration_s:
            openings: dict[str, float] = {}
            for side in SIDES:
                calibration = getattr(feetech, side)
                try:
                    ticks = unwrappers[side](
                        buses[_port(feetech, calibration)].read_position(
                            calibration.servo_id
                        )
                    )
                    opening = calibration.normalized_width(ticks)
                except (OSError, RuntimeError) as exc:
                    failed_reads[side] += 1
                    last_errors[side] = str(exc)
                    if side in last_openings:
                        openings[side] = last_openings[side]
                    continue

                if failed_reads[side]:
                    LOG.info(
                        "%s Feetech recovered after %d failed read(s).",
                        side,
                        failed_reads[side],
                    )
                failed_reads[side] = 0
                last_errors[side] = ""
                last_ticks[side] = ticks
                last_openings[side] = opening
                openings[side] = opening

            if openings:
                last_targets.update(
                    environment.set_gripper_openings(
                        openings,
                        fallback_max_width_mm=fallback_max_width_mm,
                    )
                )
            environment.raise_if_failed()
            now = time.perf_counter()
            if now >= next_print:
                for side in SIDES:
                    gripper_range = environment.gripper_ranges.get(side)
                    range_text = (
                        f"{gripper_range.min_mm:.1f}..{gripper_range.max_mm:.1f} mm "
                        f"[{gripper_range.source}]"
                        if gripper_range is not None
                        else (
                            f"max={fallback_max_width_mm:.1f} mm "
                            "[robot config fallback]"
                        )
                    )
                    if failed_reads[side]:
                        hold_status = (
                            "holding last value"
                            if side in last_openings
                            else "awaiting first valid value"
                        )
                        status = (
                            f"read failed x{failed_reads[side]}; {hold_status}: "
                            f"{last_errors[side]}"
                        )
                    else:
                        status = "ok"
                    LOG.info(
                        "%s ticks=%s normalized=%s -> Piper=%s um; range=%s; %s",
                        side,
                        last_ticks.get(side, "-"),
                        (
                            f"{last_openings[side]:.4f}"
                            if side in last_openings
                            else "-"
                        ),
                        last_targets.get(side, "-"),
                        range_text,
                        status,
                    )
                next_print = now + print_period_s

            next_tick = max(next_tick + period_s, time.perf_counter())
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
    except KeyboardInterrupt:
        LOG.info("Stopped by user.")
    finally:
        for bus in buses.values():
            bus.close()
        environment.close()


if __name__ == "__main__":
    main()
