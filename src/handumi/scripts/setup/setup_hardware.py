#!/usr/bin/env python3
"""Interactive hardware setup for novice HandUMI + real robot users."""

from __future__ import annotations

import argparse
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.setup import ensure_feetech_serial_permissions, run_feetech_wizard
from handumi.real.can_setup import (
    ensure_can_interfaces_ready,
    ensure_rig_config,
    run_piper_can_wizard,
)
from handumi.real.piper_can import load_piper_can_settings
from handumi.tracking.pico import prepare_pico_adb_session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("piper",), default="piper")
    parser.add_argument("--device", choices=("pico", "meta"), default="pico")
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--restart-ms", type=int, default=100)
    parser.add_argument(
        "--skip-can-map",
        action="store_true",
        help="Use the existing rig.yaml CAN mapping instead of the replug wizard.",
    )
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Do not run sudo/ip-link repair after mapping CAN.",
    )
    parser.add_argument(
        "--skip-feetech-map",
        action="store_true",
        help="Use the existing rig.yaml Feetech mapping instead of the replug wizard.",
    )
    parser.add_argument("--feetech-start-id", type=int, default=0)
    parser.add_argument("--feetech-end-id", type=int, default=20)
    parser.add_argument(
        "--skip-pico",
        action="store_true",
        help="Skip ADB reverse and PICO keep-awake setup.",
    )
    parser.add_argument("--skip-adb-check", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    ensure_rig_config(args.rig_config)

    if not args.skip_feetech_map:
        ensure_feetech_serial_permissions()

    if args.robot == "piper":
        if not args.skip_can_map:
            run_piper_can_wizard(
                rig_config=args.rig_config,
                bitrate=args.bitrate,
                restart_ms=args.restart_ms,
            )
        settings = load_piper_can_settings(args.rig_config)
        if not args.skip_can_repair:
            ensure_can_interfaces_ready(
                [settings.left_port, settings.right_port],
                bitrate=settings.bitrate,
                restart_ms=settings.restart_ms,
            )
            print("CAN listo.")

    if not args.skip_feetech_map:
        run_feetech_wizard(
            rig_config=args.rig_config,
            start_id=args.feetech_start_id,
            end_id=args.feetech_end_id,
        )
        print("Feetech listo.")

    if args.device == "pico" and not args.skip_pico:
        prepare_pico_adb_session(skip_adb_check=args.skip_adb_check)
        print("PICO listo por USB/ADB.")

    print("\nSetup listo. Prueba:")
    print(f"  uv run handumi-teleop-real --device {args.device} --robot {args.robot}")


if __name__ == "__main__":
    main()
