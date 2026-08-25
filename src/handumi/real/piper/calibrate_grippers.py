"""Capture each Piper gripper's physical closed position as logical zero.

Some Piper controllers do not accept the SDK's hardware-zero command. This
calibration records the fresh raw feedback selected by the operator and uses it
as the per-side minimum for both commands and normalized feedback.

The arm joints receive no position targets during this procedure. The selected
gripper motor is disabled while the operator closes it by hand.

Run from the repository with::

    uv run handumi calibrate piper-grippers
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready

if TYPE_CHECKING:
    from handumi.real.piper.driver import PiperCanEnvironment

log = logging.getLogger("handumi.piper_gripper_calibration")


def user_piper_gripper_calibration_path() -> Path:
    """Return the machine-local Piper gripper calibration file."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "handumi" / "piper_grippers.yaml"


def load_piper_gripper_zeros(path: Path | None = None) -> dict[str, int]:
    """Load measured closed feedback values in Piper micrometers."""
    source = path or user_piper_gripper_calibration_path()
    if not source.exists():
        return {}
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    zeros: dict[str, int] = {}
    for side in ("left", "right"):
        side_data = data.get(side) or {}
        value = side_data.get("closed_microm")
        if value is not None:
            zeros[side] = int(value)
    return zeros


def save_piper_gripper_zero(
    side: str,
    closed_microm: int,
    path: Path | None = None,
) -> Path:
    """Persist one measured closed value while preserving the other side."""
    if side not in {"left", "right"}:
        raise ValueError(f"invalid Piper side: {side!r}")
    destination = path or user_piper_gripper_calibration_path()
    zeros = load_piper_gripper_zeros(destination)
    zeros[side] = int(closed_microm)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {
        name: {"closed_microm": value}
        for name, value in zeros.items()
        if name in {"left", "right"}
    }
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set the current fully-closed physical Piper gripper position as "
            "zero. Arm joints are not commanded."
        )
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--side", choices=("left", "right", "both"), default="both"
    )
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Fail instead of attempting to bring CAN interfaces up.",
    )
    return parser


def _watch_until_enter(
    environment: PiperCanEnvironment,
    side: str,
    interval_s: float,
) -> int | None:
    print(
        f"\n[{side}] El motor del gripper esta deshabilitado. "
        "Cierralo suavemente a mano y pulsa ENTER para fijar ese punto como 0."
    )
    latest: int | None = None
    while True:
        arm = environment.arms[side]
        sample = arm.read_gripper_microm()
        if sample is not None:
            latest = sample
            value = f"{sample / 1000.0:+.3f} mm (raw={sample:+d} um)"
        else:
            value = "SIN_FEEDBACK"
        sys.stdout.write(f"\r  feedback={value:<40}")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], interval_s)
        if ready:
            sys.stdin.readline()
            sys.stdout.write("\n")
            return latest


def main(argv: list[str] | None = None) -> None:
    # Imported lazily so driver.py can reuse the calibration persistence
    # helpers above without creating a circular import.
    from handumi.real.piper.driver import (
        PiperCanEnvironment,
        load_piper_can_settings,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    if args.interval_s <= 0.0:
        raise SystemExit("--interval-s must be > 0")
    sides = ("left", "right") if args.side == "both" else (args.side,)
    settings = load_piper_can_settings(args.rig_config)
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=not args.skip_can_repair,
    )

    environment = PiperCanEnvironment(settings)
    output: Path | None = None
    try:
        environment.connect()
        print(
            "\nEsta operacion NO mueve las articulaciones del brazo. Mantén el "
            "espacio de trabajo despejado y no fuerces los dedos al cerrarlos."
        )
        for side in sides:
            environment.disable_gripper(side)
            before = _watch_until_enter(environment, side, args.interval_s)
            if before is None:
                raise RuntimeError(
                    f"{side}: no hay feedback; no es seguro fijar el cero"
                )
            output = save_piper_gripper_zero(side, before)
            print(
                f"  {side}: raw={before:+d} um ahora corresponde a "
                "opening=0.000000 [OK]"
            )
    finally:
        environment.close()

    assert output is not None
    print(
        f"\nCalibracion guardada en {output}. Vuelve a ejecutar teleop o el "
        "monitor de tests/real para verificarla."
    )


__all__ = [
    "load_piper_gripper_zeros",
    "save_piper_gripper_zero",
    "user_piper_gripper_calibration_path",
]


if __name__ == "__main__":
    main()
