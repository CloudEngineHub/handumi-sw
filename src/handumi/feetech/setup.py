"""Interactive Feetech wiring setup helpers for ``configs/rig.yaml``."""

from __future__ import annotations

import glob
import getpass
import grp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from handumi.config import DEFAULT_RIG_CONFIG, EXAMPLE_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import default_config

DEFAULT_SERIAL_PATTERNS = ("/dev/ttyACM*", "/dev/ttyUSB*")
CommandRunner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class FeetechServoRef:
    side: str
    port: str
    servo_id: int


def list_feetech_serial_ports(
    patterns: Iterable[str] = DEFAULT_SERIAL_PATTERNS,
) -> set[str]:
    """Return candidate USB serial ports for Feetech adapters."""
    ports: set[str] = set()
    for pattern in patterns:
        ports.update(glob.glob(pattern))
    return set(sorted(ports))


def scan_feetech_ids(
    port: str,
    *,
    start_id: int,
    end_id: int,
    baudrate: int,
    protocol_version: int,
    bus_cls=FeetechBus,
) -> list[int]:
    if start_id > end_id:
        raise SystemExit("--feetech-start-id must be <= --feetech-end-id.")
    _assert_serial_port_access(port)
    with bus_cls(
        port=port,
        baudrate=baudrate,
        protocol_version=protocol_version,
    ) as bus:
        return [int(servo_id) for servo_id in bus.scan(range(start_id, end_id + 1))]


def identify_feetech_by_replug(
    side_label: str,
    *,
    start_id: int,
    end_id: int,
    baudrate: int,
    protocol_version: int,
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    list_ports_fn: Callable[[], set[str]] = list_feetech_serial_ports,
    scan_ids_fn: Callable[[str], list[int]] | None = None,
    used_ports: set[str] | None = None,
) -> FeetechServoRef:
    """Identify one Feetech adapter by disconnect/reconnect and scan its ID."""
    used_ports = set(used_ports or set())
    print_fn(f"\nIdentifying {side_label} HandUMI Feetech.")
    input_fn(f"Unplug the {side_label} Feetech adapter, then press Enter.")
    disconnected = list_ports_fn()
    input_fn(f"Plug in ONLY the {side_label} Feetech adapter, then press Enter.")
    print_fn("  Waiting for a new serial port...")

    deadline = time.time() + timeout_s
    next_status_s = time.time() + 3.0
    last_seen: set[str] = set()
    while time.time() < deadline:
        current = list_ports_fn()
        last_seen = current
        added = sorted((current - disconnected) - used_ports)
        if len(added) == 1:
            return _scan_identified_feetech(
                side_label,
                added[0],
                start_id=start_id,
                end_id=end_id,
                baudrate=baudrate,
                protocol_version=protocol_version,
                scan_ids_fn=scan_ids_fn,
                print_fn=print_fn,
            )
        if len(added) > 1:
            raise SystemExit(
                "Multiple new serial ports were detected: "
                f"{', '.join(added)}. Connect only one device per step."
            )
        if used_ports:
            existing_unused = sorted(current - used_ports)
            if len(existing_unused) == 1:
                print_fn(
                    "  No new port appeared, but one Feetech port is still unassigned; "
                    "using it."
                )
                return _scan_identified_feetech(
                    side_label,
                    existing_unused[0],
                    start_id=start_id,
                    end_id=end_id,
                    baudrate=baudrate,
                    protocol_version=protocol_version,
                    scan_ids_fn=scan_ids_fn,
                    print_fn=print_fn,
                )
        now = time.time()
        if now >= next_status_s:
            print_fn(
                "  Still waiting for a new port. "
                f"Current ports: {_format_ports(current)}"
            )
            next_status_s = now + 3.0
        time.sleep(poll_s)
    raise SystemExit(
        f"Could not detect the {side_label} Feetech within {timeout_s:.0f}s.\n"
        f"Ports before: {_format_ports(disconnected)}\n"
        f"Ports now: {_format_ports(last_seen)}\n"
        "Check that the adapter appears as /dev/ttyACM* or /dev/ttyUSB*, "
        "that the USB cable carries data, and that only this Feetech was plugged in."
    )


def _scan_identified_feetech(
    side_label: str,
    port: str,
    *,
    start_id: int,
    end_id: int,
    baudrate: int,
    protocol_version: int,
    scan_ids_fn: Callable[[str], list[int]] | None,
    print_fn: Callable[[str], None],
) -> FeetechServoRef:
    _assert_serial_port_access(port)
    scanner = scan_ids_fn or (
        lambda detected_port: scan_feetech_ids(
            detected_port,
            start_id=start_id,
            end_id=end_id,
            baudrate=baudrate,
            protocol_version=protocol_version,
        )
    )
    ids = scanner(port)
    if len(ids) == 1:
        ref = FeetechServoRef(side=side_label, port=port, servo_id=int(ids[0]))
        print_fn(f"  {side_label}: detected port={ref.port}, servo_id={ref.servo_id}")
        return ref
    if not ids:
        raise SystemExit(
            f"{port} appeared for {side_label}, but no servo replied. "
            "Check power, wiring, baudrate, and the scanned ID range."
        )
    raise SystemExit(
        f"{port} has multiple Feetech IDs {ids}. "
        "Connect only one servo or change IDs with handumi-set-servo-id."
    )


def _format_ports(ports: set[str]) -> str:
    return ", ".join(sorted(ports)) if ports else "none"


def ensure_feetech_serial_permissions(
    *,
    list_ports_fn: Callable[[], set[str]] = list_feetech_serial_ports,
    runner: CommandRunner = subprocess.run,
    user: str | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Preflight serial permissions before the interactive setup starts."""
    ports = sorted(list_ports_fn())
    if not ports:
        return

    missing_groups: dict[str, list[str]] = {}
    blocked_ports: list[str] = []
    for port in ports:
        if os.access(port, os.R_OK | os.W_OK):
            continue
        group_name = _serial_port_group_name(port)
        if group_name and group_name not in _current_group_names():
            missing_groups.setdefault(group_name, []).append(port)
            continue
        blocked_ports.append(port)

    if missing_groups:
        target_user = user or getpass.getuser()
        details = "; ".join(
            f"{group}: {', '.join(group_ports)}"
            for group, group_ports in sorted(missing_groups.items())
        )
        print_fn(f"Feetech needs serial permissions ({details}).")
        print_fn("Sudo is required to add your user to the serial device group.")
        sudo = runner(["sudo", "-v"], check=False)
        if sudo.returncode != 0:
            raise SystemExit("Could not obtain sudo; serial permissions were not changed.")
        for group in sorted(missing_groups):
            result = runner(
                ["sudo", "usermod", "-aG", group, target_user],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                raise SystemExit(f"Could not add {target_user} to {group}.\n{stderr}")
        raise SystemExit(
            "Serial permissions were updated.\n"
            "Log out and back in, or reboot, then run again:\n"
            "  uv run handumi-setup --robot piper --device pico"
        )

    if blocked_ports:
        raise SystemExit(
            "No permission to open these Feetech ports: "
            f"{', '.join(blocked_ports)}.\n"
            "Your user appears to be in the right group; check udev rules "
            "or whether another process is holding the port."
        )


def _assert_serial_port_access(port: str) -> None:
    if os.access(port, os.R_OK | os.W_OK):
        return
    hint = _serial_port_permission_hint(port)
    if hint:
        raise SystemExit(
            f"No permission to open {port}.\n"
            + "\n".join(hint)
        )
    raise SystemExit(f"No permission to open {port}.")


def _serial_port_permission_hint(port: str) -> list[str]:
    try:
        stat_result = os.stat(port)
    except OSError:
        return []
    group_name = _group_name_from_gid(stat_result.st_gid)

    if stat_result.st_gid in os.getgroups():
        return [
            "Your user is already in the port group, but access still failed.",
            "Check udev rules or whether another process is holding the port.",
        ]

    return [
        f"Add your user to the `{group_name}` group:",
        f"  sudo usermod -aG {group_name} $USER",
        "Then log out and back in, or reboot.",
    ]


def _serial_port_group_name(port: str) -> str:
    try:
        return _group_name_from_gid(os.stat(port).st_gid)
    except OSError:
        return ""


def _group_name_from_gid(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _current_group_names() -> set[str]:
    names: set[str] = set()
    for gid in os.getgroups():
        names.add(_group_name_from_gid(gid))
    return names


def save_feetech_mapping(
    *,
    rig_config: Path,
    left: FeetechServoRef,
    right: FeetechServoRef,
    baudrate: int,
    protocol_version: int,
) -> None:
    ensure_rig_config(rig_config)
    with rig_config.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    data["feetech"] = {
        "baudrate": int(baudrate),
        "protocol_version": int(protocol_version),
        "left": {"servo_id": int(left.servo_id), "port": left.port},
        "right": {"servo_id": int(right.servo_id), "port": right.port},
    }

    with rig_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_feetech_wizard(
    *,
    rig_config: Path = DEFAULT_RIG_CONFIG,
    start_id: int = 0,
    end_id: int = 20,
    baudrate: int | None = None,
    protocol_version: int | None = None,
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    list_ports_fn: Callable[[], set[str]] = list_feetech_serial_ports,
    scan_ids_fn: Callable[[str], list[int]] | None = None,
) -> tuple[FeetechServoRef, FeetechServoRef]:
    """Wizard that writes left/right Feetech port and ID assignments."""
    defaults = default_config()
    baudrate = int(baudrate if baudrate is not None else defaults.baudrate)
    protocol_version = int(
        protocol_version if protocol_version is not None else defaults.protocol_version
    )

    print_fn("Feetech wizard: RIGHT gripper first, then LEFT gripper.")
    right = identify_feetech_by_replug(
        "right",
        start_id=start_id,
        end_id=end_id,
        baudrate=baudrate,
        protocol_version=protocol_version,
        timeout_s=timeout_s,
        poll_s=poll_s,
        input_fn=input_fn,
        print_fn=print_fn,
        list_ports_fn=list_ports_fn,
        scan_ids_fn=scan_ids_fn,
    )
    left = identify_feetech_by_replug(
        "left",
        start_id=start_id,
        end_id=end_id,
        baudrate=baudrate,
        protocol_version=protocol_version,
        timeout_s=timeout_s,
        poll_s=poll_s,
        input_fn=input_fn,
        print_fn=print_fn,
        list_ports_fn=list_ports_fn,
        scan_ids_fn=scan_ids_fn,
        used_ports={right.port},
    )
    if left.port == right.port:
        raise SystemExit(f"Left and right Feetech are using the same port: {left.port}")

    save_feetech_mapping(
        rig_config=rig_config,
        left=left,
        right=right,
        baudrate=baudrate,
        protocol_version=protocol_version,
    )
    print_fn(
        f"Saved to {rig_config}: "
        f"left={left.port}/id{left.servo_id}, right={right.port}/id{right.servo_id}"
    )
    return left, right


def ensure_rig_config(path: Path = DEFAULT_RIG_CONFIG) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLE_RIG_CONFIG, path)


__all__ = [
    "FeetechServoRef",
    "ensure_feetech_serial_permissions",
    "identify_feetech_by_replug",
    "list_feetech_serial_ports",
    "run_feetech_wizard",
    "save_feetech_mapping",
    "scan_feetech_ids",
]
