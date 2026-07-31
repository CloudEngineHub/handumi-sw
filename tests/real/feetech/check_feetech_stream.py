#!/usr/bin/env python3
"""Measure the real HandUMI Feetech stream and exercise automatic recovery.

This diagnostic never writes servo registers or enables torque.  With
``--force-reconnect`` it closes one serial adapter before starting the sampler
to verify that the runtime reopens it within the allowed recovery window.

Examples:

    .venv/bin/python tests/real/feetech/check_feetech_stream.py
    .venv/bin/python tests/real/feetech/check_feetech_stream.py --force-reconnect
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech import FeetechGripperPair, FeetechGripperSampler
from handumi.feetech.calibration import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--force-reconnect", action="store_true")
    parser.add_argument("--max-recovery-ms", type=float, default=250.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0 or args.sample_hz <= 0.0:
        raise SystemExit("--duration-s and --sample-hz must be greater than zero.")

    pair = FeetechGripperPair(load_config(args.rig_config))
    sampler = FeetechGripperSampler(
        pair,
        sample_hz=args.sample_hz,
        buffer_seconds=args.duration_s + 1.0,
    )
    pair.open()
    if args.force_reconnect:
        # Deliberately reproduce a lost/wedged adapter without touching servo
        # EEPROM, torque, IDs, or calibration.
        pair._buses[pair._left_port].close()

    started = time.monotonic()
    try:
        sampler.start(timeout_s=max(1.0, args.max_recovery_ms / 1000.0 * 2.0))
        startup_ms = (time.monotonic() - started) * 1000.0
        time.sleep(args.duration_s)
        samples = sampler.samples()
        latest = sampler.latest()
        latest_age_ns = (
            0 if latest is None else time.monotonic_ns() - latest.sample_time_ns
        )
    finally:
        sampler.stop()
        pair.close()

    if latest is None or len(samples) < 2:
        raise SystemExit("FAIL: fewer than two valid Feetech samples were received.")

    gaps_ms = [
        (new.sample_time_ns - old.sample_time_ns) / 1e6
        for old, new in zip(samples, samples[1:], strict=False)
    ]
    sorted_gaps = sorted(gaps_ms)
    p99_ms = sorted_gaps[int(0.99 * (len(sorted_gaps) - 1))]
    mean_period_s = statistics.fmean(gaps_ms) / 1000.0
    effective_hz = 1.0 / mean_period_s
    age_ms = latest_age_ns / 1e6
    expected_period_ms = 1000.0 / args.sample_hz

    print(
        f"samples={len(samples)} effective_hz={effective_hz:.2f} "
        f"median_ms={statistics.median(gaps_ms):.3f} "
        f"p99_ms={p99_ms:.3f} max_gap_ms={max(gaps_ms):.3f}"
    )
    print(
        f"startup_ms={startup_ms:.2f} latest_age_ms={age_ms:.2f} "
        f"errors={sampler.total_errors} reconnects={sampler.reconnect_count}"
    )
    print(
        f"left={latest.widths.left_ticks}/{latest.widths.left_mm:.2f}mm "
        f"right={latest.widths.right_ticks}/{latest.widths.right_mm:.2f}mm"
    )

    failures: list[str] = []
    if effective_hz < args.sample_hz * 0.90:
        failures.append("effective sample rate below 90% of target")
    if p99_ms > expected_period_ms * 2.0:
        failures.append("p99 sample interval above two periods")
    if max(gaps_ms) > max(100.0, expected_period_ms * 4.0):
        failures.append("maximum valid-sample gap is too large")
    if age_ms > max(50.0, expected_period_ms * 3.0):
        failures.append("latest sample is stale")
    if args.force_reconnect:
        if sampler.reconnect_count < 1:
            failures.append("forced adapter loss did not trigger reconnect")
        if startup_ms > args.max_recovery_ms:
            failures.append(
                f"forced recovery exceeded {args.max_recovery_ms:.0f} ms"
            )

    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("PASS: Feetech stream meets latency and recovery thresholds.")


if __name__ == "__main__":
    main()
