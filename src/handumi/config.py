"""Shared loading for the machine-local HandUMI rig configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_RIG_CONFIG = Path("configs/rig.yaml")
SIDES = ("left", "right")
CAMERA_ROLES = (*SIDES, "workspace")


def camera_role(name: str, config: object = None) -> str | None:
    """Return a camera's physical role without constraining its logical name.

    New rigs may declare ``role`` explicitly.  The name-based aliases keep old
    rigs working and make the common ``left``, ``right``, and ``top`` names do
    the expected thing without extra configuration.
    """
    if isinstance(config, dict) and config.get("role") is not None:
        role = str(config["role"]).strip().lower()
        if role not in CAMERA_ROLES:
            raise SystemExit(
                f"Camera {name!r} has invalid role {role!r}; choose from: "
                f"{', '.join(CAMERA_ROLES)}."
            )
        return role
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "left": "left",
        "left_wrist": "left",
        "right": "right",
        "right_wrist": "right",
        "workspace": "workspace",
        "top": "workspace",
    }
    return aliases.get(normalized)


def resolve_camera_role(cameras: dict[str, Any], role: str) -> str:
    """Resolve a physical role to the user-defined camera key."""
    if role not in CAMERA_ROLES:
        raise ValueError(f"Unknown camera role: {role!r}")
    explicit = [
        name
        for name, entry in cameras.items()
        if isinstance(entry, dict)
        and entry.get("role") is not None
        and camera_role(name, entry) == role
    ]
    candidates = explicit or [
        name for name, entry in cameras.items() if camera_role(name, entry) == role
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise SystemExit(
            f"Multiple cameras have role {role!r}: {', '.join(candidates)}. "
            "Set a unique 'role' on the intended camera in the rig configuration."
        )
    raise SystemExit(
        f"No camera with role {role!r} is configured. Available cameras: "
        f"{', '.join(cameras) or '(none)'}. Set role: {role} on the intended camera."
    )


def resolve_active_sides(
    side: str | None = None, *, available: tuple[str, ...] = SIDES
) -> tuple[str, ...]:
    """Select existing arms; omission follows the embodiment's topology."""
    selected = available if side is None else SIDES if side == "both" else (side,)
    if not selected or any(value not in available for value in selected):
        raise ValueError(f"Side {side!r} is unavailable; this embodiment has {available}.")
    return selected


def dataset_active_sides(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Legacy captures are bilateral; new captures declare their active sensors."""
    values = metadata.get("active_sides", SIDES)
    if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
        raise ValueError("handumi.active_sides must contain left, right, or both.")
    if any(side not in SIDES for side in values):
        raise ValueError("handumi.active_sides must contain left, right, or both.")
    return tuple(side for side in SIDES if side in values)


_PACKAGE_ROOT = Path(__file__).resolve().parent
EXAMPLE_RIG_CONFIG = (
    Path("configs/rig.example.yaml")
    if Path("configs/rig.example.yaml").exists()
    else _PACKAGE_ROOT / "configs" / "rig.example.yaml"
)


def load_rig_section(path: Path, section: str) -> dict[str, Any]:
    """Load one mapping from the unified rig YAML."""
    data = load_rig_config(path)
    value = data.get(section)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing or invalid '{section}' section in {path}.")
    return value


def load_rig_config(path: Path = DEFAULT_RIG_CONFIG) -> dict[str, Any]:
    """Load the complete rig mapping so optional UX defaults can coexist."""
    if not path.exists():
        raise SystemExit(
            f"Missing rig configuration: {path}.\n"
            f"Create it with: cp {EXAMPLE_RIG_CONFIG} {DEFAULT_RIG_CONFIG}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid rig configuration mapping: {path}.")
    return data


def load_optional_rig_section(
    path: Path,
    section: str,
) -> dict[str, Any]:
    """Return an optional rig section without making old rig files invalid."""
    value = load_rig_config(path).get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid '{section}' section in {path}; expected a mapping.")
    return value
