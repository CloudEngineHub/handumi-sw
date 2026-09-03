"""Joint-vector layouts of LeRobot follower robots, and how to reach them.

HandUMI's canonical joint vector (:mod:`handumi.dataset.canonical`) stores arm
joints in radians and one gripper opening in meters per side. That is the
representation to keep: it is physical, so the same numbers mean the same
motion on any embodiment. A LeRobot robot plugin, however, records the vector
*its* motor bus produces, and LeRobot leaves the encoding to each plugin's
``MotorNormMode``: a Feetech leader normalizes to ``[-100, 100]``, a Damiao
bus reports degrees, a CAN driver may apply its own limits and signs. A
dataset only holds the resulting numbers, never the encoding that produced
them, so the encoding has to be written down here, per embodiment, citing the
plugin it was read from.

This module is that record. Each :class:`ExternalJointLayout` names a LeRobot
``robot_type`` and gives, per column, a :class:`JointEncoding` mirroring the
plugin's norm mode. Two directions share it: the exporter (canonical ->
plugin vector) and the replay viewer (plugin vector -> canonical), so they
cannot drift apart. Adding an embodiment means adding one layout, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from handumi.dataset.canonical import SIDES, canonical_joint_layout

# ---------------------------------------------------------------------------
# Per-column encoding, mirroring lerobot.motors.MotorNormMode
# ---------------------------------------------------------------------------

RANGE_M100_100 = "range_m100_100"
RANGE_0_100 = "range_0_100"
DEGREES = "degrees"
MODES = (RANGE_M100_100, RANGE_0_100, DEGREES)


@dataclass(frozen=True)
class JointEncoding:
    """How one column encodes a joint, in the plugin's own terms.

    Arm columns:
      * ``range_m100_100``: ``sign * q`` mapped linearly from
        ``[lower_rad, upper_rad]`` onto ``[-100, 100]``.
      * ``degrees``: ``sign * degrees(q)``.

    Gripper columns (input is the opening fraction ``0..1``):
      * ``range_0_100``: ``100 * fraction``.
      * ``degrees``: the gripper motor angle, interpolated from ``closed_rad``
        to ``open_rad`` and reported in degrees, as a Damiao bus does.
    """

    mode: str
    sign: int = 1
    lower_rad: float | None = None
    upper_rad: float | None = None
    closed_rad: float | None = None
    open_rad: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown encoding mode {self.mode!r}; expected one of {MODES}.")
        if self.sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1")
        if self.mode == RANGE_M100_100:
            if self.lower_rad is None or self.upper_rad is None:
                raise ValueError("range_m100_100 needs lower_rad and upper_rad")
            if self.upper_rad <= self.lower_rad:
                raise ValueError("range_m100_100 needs lower_rad < upper_rad")

    # ----- ranges -----

    @property
    def driver_range(self) -> tuple[float, float] | None:
        """Values the plugin's driver accepts, or None when it takes any angle."""
        if self.mode == RANGE_M100_100:
            return (-100.0, 100.0)
        if self.mode == RANGE_0_100:
            return (0.0, 100.0)
        return None

    @property
    def rad_per_unit(self) -> float | None:
        """Radians per one unit of the encoded value (arm columns)."""
        if self.mode == RANGE_M100_100:
            assert self.lower_rad is not None and self.upper_rad is not None
            return (self.upper_rad - self.lower_rad) / 200.0
        if self.mode == DEGREES:
            return float(np.pi / 180.0)
        return None

    # ----- arm joints -----

    def arm_from_rad(self, q: np.ndarray) -> np.ndarray:
        signed = self.sign * np.asarray(q, dtype=np.float64)
        if self.mode == RANGE_M100_100:
            assert self.lower_rad is not None and self.upper_rad is not None
            return (signed - self.lower_rad) * 200.0 / (self.upper_rad - self.lower_rad) - 100.0
        if self.mode == DEGREES:
            return np.degrees(signed)
        raise ValueError(f"{self.mode} is not an arm-joint encoding")

    def arm_to_rad(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.mode == RANGE_M100_100:
            assert self.lower_rad is not None and self.upper_rad is not None
            signed = self.lower_rad + (value + 100.0) * (self.upper_rad - self.lower_rad) / 200.0
        elif self.mode == DEGREES:
            signed = np.radians(value)
        else:
            raise ValueError(f"{self.mode} is not an arm-joint encoding")
        return self.sign * signed

    # ----- gripper -----

    def gripper_from_fraction(self, fraction: np.ndarray) -> np.ndarray:
        fraction = np.asarray(fraction, dtype=np.float64)
        if self.mode == RANGE_0_100:
            return 100.0 * fraction
        if self.mode == DEGREES:
            closed, opened = self._gripper_endpoints()
            return np.degrees(closed + fraction * (opened - closed))
        raise ValueError(f"{self.mode} is not a gripper encoding")

    def gripper_to_fraction(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.mode == RANGE_0_100:
            return value / 100.0
        if self.mode == DEGREES:
            closed, opened = self._gripper_endpoints()
            return (np.radians(value) - closed) / (opened - closed)
        raise ValueError(f"{self.mode} is not a gripper encoding")

    def _gripper_endpoints(self) -> tuple[float, float]:
        if self.closed_rad is None or self.open_rad is None or self.open_rad == self.closed_rad:
            raise ValueError("a degrees gripper encoding needs distinct closed_rad and open_rad")
        return float(self.closed_rad), float(self.open_rad)

    def describe(self) -> str:
        if self.mode == RANGE_M100_100:
            return (
                f"{self.mode} over [{self.lower_rad:.4f}, {self.upper_rad:.4f}] rad"
                f"{' (sign -1)' if self.sign < 0 else ''}"
            )
        if self.mode == DEGREES and self.closed_rad is not None and self.open_rad is not None:
            closed, opened = self._gripper_endpoints()
            return f"degrees, closed={np.degrees(closed):.1f} open={np.degrees(opened):.1f}"
        if self.mode == DEGREES:
            return f"degrees{' (sign -1)' if self.sign < 0 else ''}"
        return self.mode

    def as_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "sign": self.sign,
            "lower_rad": self.lower_rad,
            "upper_rad": self.upper_rad,
            "closed_rad": self.closed_rad,
            "open_rad": self.open_rad,
        }


# ---------------------------------------------------------------------------
# Layout description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalJointLayout:
    """One LeRobot follower's joint vector for a bimanual arm + gripper.

    Column order is ``left`` then ``right``; within a side, ``arm_names`` in
    order followed by ``gripper_name``, each carrying ``.pos``. This is how
    LeRobot's ``bi_*_follower`` robots build ``action_features``.
    """

    name: str
    robot: str
    robot_type: str
    arm_names: tuple[str, ...]
    gripper_name: str
    arm_encodings: tuple[JointEncoding, ...]
    gripper_encoding: JointEncoding
    source: str
    default_camera_map: dict[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    # Mirrors the plugin's ``use_degrees`` config: "optional" when the plugin
    # exposes the flag (Feetech-based followers), "always" when its bus only
    # speaks degrees (Damiao), "never" when the plugin has no such option.
    degrees_option: str = "never"
    use_degrees: bool = False

    def __post_init__(self) -> None:
        if len(self.arm_encodings) != len(self.arm_names):
            raise ValueError("one encoding per arm joint is required")
        if self.degrees_option not in ("optional", "always", "never"):
            raise ValueError("degrees_option must be optional, always or never")

    def with_use_degrees(self, enabled: bool) -> ExternalJointLayout:
        """The layout LeRobot's ``use_degrees=<enabled>`` would record.

        Exactly what the plugins do: every arm joint switches between its
        normalized range and plain degrees, the gripper keeps its own mode.
        Limits and signs are kept on the encoding so the variant can be
        switched back.
        """
        if enabled == self.use_degrees:
            return self
        if self.degrees_option == "always":
            if not enabled:
                raise ValueError(f"{self.robot_type} only records degrees.")
            return self
        if self.degrees_option == "never":
            raise ValueError(
                f"The {self.robot_type} plugin has no use_degrees option; its driver "
                "only accepts the normalized vector."
            )
        mode = DEGREES if enabled else RANGE_M100_100
        base = self.name.removesuffix("_degrees")
        return replace(
            self,
            name=f"{base}_degrees" if enabled else base,
            arm_encodings=tuple(replace(enc, mode=mode) for enc in self.arm_encodings),
            use_degrees=enabled,
        )

    @property
    def names(self) -> list[str]:
        return [
            f"{side}_{joint}.pos"
            for side in SIDES
            for joint in (*self.arm_names, self.gripper_name)
        ]

    @property
    def per_side(self) -> int:
        return len(self.arm_names) + 1

    @property
    def size(self) -> int:
        return len(SIDES) * self.per_side

    @property
    def output_suffix(self) -> str:
        return self.robot_type

    def encoding(self, column: int) -> JointEncoding:
        k = column % self.per_side
        return self.gripper_encoding if k == len(self.arm_names) else self.arm_encodings[k]

    def is_gripper(self, column: int) -> bool:
        return column % self.per_side == len(self.arm_names)

    def describe(self) -> str:
        arms = "; ".join(
            f"{name}: {enc.describe()}" for name, enc in zip(self.arm_names, self.arm_encodings)
        )
        return f"{arms}; {self.gripper_name}: {self.gripper_encoding.describe()}"

    def as_metadata(self) -> dict[str, object]:
        return {
            "layout": self.name,
            "robot_type": self.robot_type,
            "robot": self.robot,
            "source": self.source,
            "names": list(self.names),
            "columns": {
                name: self.encoding(column).as_metadata() for column, name in enumerate(self.names)
            },
            "assumptions": list(self.assumptions),
            "use_degrees": self.use_degrees,
            "degrees_option": self.degrees_option,
        }


def _tenths(lo: int, hi: int) -> tuple[float, float]:
    return float(np.radians(lo * 0.1)), float(np.radians(hi * 0.1))


# AgileX Piper, as XHUMAN's LeRobot plugin drives it over CAN. It normalizes
# each joint to [-100, 100] over the firmware joint limits (queried live;
# these defaults when the query fails) after flipping joints 1, 4 and 6 so a
# mirrored leader maps intuitively. The gripper is one [0, 100] channel over
# a per-machine measured opening. HandUMI commands the same robot in URDF
# radians with no sign change (handumi/real/piper/driver.py), so the URDF
# convention is the raw SDK convention and these signs are exactly what
# separates the two vectors. Firmware limits match the URDF to 4 decimals.
_PIPER_LIMITS = ((-1500, 1500), (0, 1800), (-1700, 0), (-1000, 1000), (-700, 700), (-1200, 1200))
_PIPER_SIGNS = (-1, 1, 1, -1, 1, -1)

BI_PIPER_FOLLOWER = ExternalJointLayout(
    name="lerobot_bi_piper_follower",
    robot="piper",
    robot_type="bi_piper_follower",
    arm_names=("shoulder_pan", "shoulder_lift", "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll"),
    gripper_name="gripper",
    arm_encodings=tuple(
        JointEncoding(RANGE_M100_100, sign=sign, lower_rad=_tenths(*lim)[0], upper_rad=_tenths(*lim)[1])
        for lim, sign in zip(_PIPER_LIMITS, _PIPER_SIGNS)
    ),
    gripper_encoding=JointEncoding(RANGE_0_100),
    source=(
        "XHUMAN xhuman/robots/piper/piper_sdk_interface.py "
        "(DEFAULT_JOINT_LIMITS, JOINT_SIGNS, get_status/set_joint_positions) and "
        "xhuman/robots/bi_piper_follower/bi_piper_follower.py (JOINT_NAMES)"
    ),
    default_camera_map={
        "observation.images.left_wrist": "observation.images.left",
        "observation.images.workspace": "observation.images.top",
        "observation.images.right_wrist": "observation.images.right",
    },
    degrees_option="never",
    assumptions=(
        "Firmware joint limits equal DEFAULT_JOINT_LIMITS; the plugin queries them "
        "from the arm at connect time and does not store them in the dataset.",
        "Gripper 0..100 spans the embodiment's full opening; the plugin's per-machine "
        "gripper calibration (closed/open micrometers) is not part of the dataset.",
    ),
)

# OpenArm v1, as LeRobot's own openarm_follower plugin drives it over CAN FD.
# Every motor, gripper included, uses MotorNormMode.DEGREES: the dataset holds
# plain degrees from the calibration zero (arm hanging straight down, gripper
# closed), which is also the URDF zero. HandUMI's real backend sends URDF
# radians to the same motors (handumi/real/openarm/driver.py) and interpolates
# the gripper motor from closed_position_rad to open_position_rad, the values
# openarmv1.yaml documents from the official zero calibration.
BI_OPENARM_FOLLOWER = ExternalJointLayout(
    name="lerobot_bi_openarm_follower",
    robot="openarmv1",
    robot_type="bi_openarm_follower",
    arm_names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"),
    gripper_name="gripper",
    arm_encodings=tuple(JointEncoding(DEGREES) for _ in range(7)),
    gripper_encoding=JointEncoding(DEGREES, closed_rad=0.0, open_rad=-1.0471975511965976),
    source=(
        "lerobot src/lerobot/robots/openarm_follower/openarm_follower.py "
        "(MotorNormMode.DEGREES for all motors, calibrate() zero pose) and "
        "config_openarm_follower.py (motor_config names); gripper endpoints from "
        "HandUMI configs/robots/openarmv1.yaml real.gripper"
    ),
    degrees_option="always",
    use_degrees=True,
    assumptions=(
        "The LeRobot calibration zero (arm hanging straight down, gripper closed) "
        "coincides with the URDF zero, so degrees = degrees(URDF radians) with no "
        "sign change; verify on hardware before deploying.",
        "Gripper motor spans 0 rad (closed) to -60 deg (open) as in the official "
        "OpenArm v1 zero calibration.",
    ),
)

EXTERNAL_LAYOUTS: dict[str, ExternalJointLayout] = {
    BI_PIPER_FOLLOWER.robot_type: BI_PIPER_FOLLOWER,
    BI_OPENARM_FOLLOWER.robot_type: BI_OPENARM_FOLLOWER,
}

# Tolerance when checking a ranged encoding's limits against the URDF the
# embodiment loads. URDFs round (3.14 for pi, 1.22 for 70 deg), so a few
# milliradians are expected; a whole different joint is not.
LIMIT_TOLERANCE_RAD = 0.01


def external_layout_for_name(name: str) -> ExternalJointLayout:
    try:
        return EXTERNAL_LAYOUTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(EXTERNAL_LAYOUTS))
        raise ValueError(f"Unknown external layout {name!r}; known: {known}.") from exc


def external_layouts_for_robot(robot: str) -> list[ExternalJointLayout]:
    """Layouts that describe a HandUMI embodiment (usually zero or one)."""
    return [layout for layout in EXTERNAL_LAYOUTS.values() if layout.robot == robot]


def detect_external_layout(info: dict[str, object]) -> ExternalJointLayout | None:
    """Recognize a dataset already in an external layout.

    A HandUMI export records the layout name in ``handumi.state_layout``; a
    dataset the plugin recorded itself carries only its ``robot_type``.
    """
    handumi = info.get("handumi")
    recorded = handumi.get("state_layout") if isinstance(handumi, dict) else None
    for layout in EXTERNAL_LAYOUTS.values():
        if recorded == layout.name:
            return layout
        if layout.degrees_option == "optional" and recorded == layout.with_use_degrees(True).name:
            return layout.with_use_degrees(True)
    robot_type = info.get("robot_type")
    for layout in EXTERNAL_LAYOUTS.values():
        if robot_type == layout.robot_type:
            return layout
    return None


# ---------------------------------------------------------------------------
# Column bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ColumnMap:
    """Where each side's arm joints and gripper sit in both vectors."""

    canonical_arm: dict[str, list[int]]
    canonical_gripper: dict[str, int]
    external_arm: dict[str, list[int]]
    external_gripper: dict[str, int]


def _column_map(layout: ExternalJointLayout, runtime) -> _ColumnMap:
    canonical = canonical_joint_layout(runtime)
    canonical_arm: dict[str, list[int]] = {}
    canonical_gripper: dict[str, int] = {}
    for side in SIDES:
        arm = [
            canonical.names.index(f"{joint}.pos")
            for joint in runtime.arm_joint_names(side)
            if f"{joint}.pos" in canonical.names
        ]
        if len(arm) != len(layout.arm_names):
            raise ValueError(
                f"{layout.name} expects {len(layout.arm_names)} arm joints per side; "
                f"{runtime.name} declares {len(arm)} for {side}."
            )
        gripper = f"{side}_gripper.width_m"
        if gripper not in canonical.names:
            raise ValueError(
                f"{layout.name} expects a gripper column per side; "
                f"{runtime.name} declares none for {side}."
            )
        canonical_arm[side] = arm
        canonical_gripper[side] = canonical.names.index(gripper)
    if canonical.size != layout.size:
        raise ValueError(
            f"Canonical layout of {runtime.name} has {canonical.size} columns, "
            f"{layout.name} has {layout.size}."
        )
    per_side = layout.per_side
    return _ColumnMap(
        canonical_arm=canonical_arm,
        canonical_gripper=canonical_gripper,
        external_arm={
            side: list(range(index * per_side, index * per_side + len(layout.arm_names)))
            for index, side in enumerate(SIDES)
        },
        external_gripper={
            side: index * per_side + len(layout.arm_names) for index, side in enumerate(SIDES)
        },
    )


def check_layout_limits(layout: ExternalJointLayout, runtime) -> None:
    """Fail loudly if a ranged encoding's limits are not the URDF's.

    A ``range_m100_100`` value only means something over the limits the
    plugin normalized with. If those and the URDF disagree by more than
    rounding, either the URDF changed or the plugin did, and exporting would
    silently rescale every joint. Degree encodings carry no limits to check.
    """
    columns = _column_map(layout, runtime)
    canonical = canonical_joint_layout(runtime)
    urdf_lower = np.asarray(runtime.robot.joints.lower_limits, dtype=np.float64)
    urdf_upper = np.asarray(runtime.robot.joints.upper_limits, dtype=np.float64)
    for side in SIDES:
        for k, column in enumerate(columns.canonical_arm[side]):
            encoding = layout.arm_encodings[k]
            if encoding.mode != RANGE_M100_100:
                continue
            joint_index = canonical.indices[column]
            assert joint_index is not None
            for label, expected, actual in (
                ("lower", encoding.lower_rad, urdf_lower[joint_index]),
                ("upper", encoding.upper_rad, urdf_upper[joint_index]),
            ):
                assert expected is not None
                if abs(expected - actual) > LIMIT_TOLERANCE_RAD:
                    raise ValueError(
                        f"{layout.name} {label} limit for {side} {layout.arm_names[k]} is "
                        f"{expected:.4f} rad but the {runtime.name} URDF says {actual:.4f} rad "
                        f"({runtime.joint_names[joint_index]}). Refusing to export with "
                        "mismatched limits."
                    )


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def _gripper_max_width(runtime) -> float:
    return max(float(runtime.config.gripper_max_width_m), 1e-6)


def from_canonical(canonical: np.ndarray, *, layout: ExternalJointLayout, runtime) -> np.ndarray:
    """Canonical (rad + m) -> the plugin's vector. No clipping.

    Values are left unclipped on purpose: a driver that rejects rather than
    clips must be able to *see* an out-of-range value to warn about it.
    """
    canonical = np.asarray(canonical, dtype=np.float64)
    if canonical.ndim != 2:
        raise ValueError(f"canonical states must be 2-D, got shape {canonical.shape}.")
    columns = _column_map(layout, runtime)
    if canonical.shape[1] != layout.size:
        raise ValueError(f"Expected {layout.size} canonical columns, got {canonical.shape[1]}.")
    out = np.empty((len(canonical), layout.size), dtype=np.float64)
    for side in SIDES:
        for k, (src, dst) in enumerate(zip(columns.canonical_arm[side], columns.external_arm[side])):
            out[:, dst] = layout.arm_encodings[k].arm_from_rad(canonical[:, src])
        fraction = canonical[:, columns.canonical_gripper[side]] / _gripper_max_width(runtime)
        out[:, columns.external_gripper[side]] = layout.gripper_encoding.gripper_from_fraction(fraction)
    return out.astype(np.float32)


def to_canonical(external: np.ndarray, *, layout: ExternalJointLayout, runtime) -> np.ndarray:
    """The plugin's vector -> canonical (rad + m). No clipping."""
    external = np.asarray(external, dtype=np.float64)
    if external.ndim != 2:
        raise ValueError(f"external states must be 2-D, got shape {external.shape}.")
    columns = _column_map(layout, runtime)
    if external.shape[1] != layout.size:
        raise ValueError(f"Expected {layout.size} {layout.name} columns, got {external.shape[1]}.")
    out = np.empty((len(external), layout.size), dtype=np.float64)
    for side in SIDES:
        for k, (dst, src) in enumerate(zip(columns.canonical_arm[side], columns.external_arm[side])):
            out[:, dst] = layout.arm_encodings[k].arm_to_rad(external[:, src])
        fraction = layout.gripper_encoding.gripper_to_fraction(external[:, columns.external_gripper[side]])
        out[:, columns.canonical_gripper[side]] = fraction * _gripper_max_width(runtime)
    return out.astype(np.float32)


def clip_to_driver_range(
    external: np.ndarray, *, layout: ExternalJointLayout, tolerance_rad: float
) -> tuple[np.ndarray, dict[str, int], float]:
    """Clip overshoots small enough to be solver noise; leave larger ones alone.

    An IK solver with a soft limit constraint settles a hair past a joint
    limit -- microradians to a millidegree -- and a driver with a hard
    accepted range rejects that as firmly as a real violation. Clipping
    within ``tolerance_rad`` keeps the command the driver would have accepted
    anyway; anything beyond stays out of range so the exporter can report it
    as the genuine problem it is. Columns without a driver range (degrees)
    are untouched.

    Returns the clipped array, per-column clip counts, and the largest
    overshoot seen in radians (before clipping, over every ranged column).
    """
    external = np.asarray(external, dtype=np.float64).copy()
    counts: dict[str, int] = {}
    worst_rad = 0.0
    for column, name in enumerate(layout.names):
        encoding = layout.encoding(column)
        bounds = encoding.driver_range
        if bounds is None:
            continue
        lower, upper = bounds
        values = external[:, column]
        over = np.maximum(np.maximum(values - upper, lower - values), 0.0)
        if layout.is_gripper(column):
            # Only float round-off at exactly closed/open is ever clipped here.
            tolerance = 1e-4
        else:
            rad_per_unit = encoding.rad_per_unit
            assert rad_per_unit is not None
            worst_rad = max(worst_rad, float(over.max()) * rad_per_unit)
            tolerance = float(tolerance_rad) / rad_per_unit
        small = (over > 0.0) & (over <= tolerance)
        if small.any():
            values[small] = np.clip(values[small], lower, upper)
            counts[name] = int(small.sum())
    return external.astype(np.float32), counts, worst_rad


def out_of_range_counts(external: np.ndarray, *, layout: ExternalJointLayout) -> dict[str, int]:
    """How many values the plugin's driver would reject, per column name."""
    external = np.asarray(external, dtype=np.float64)
    counts: dict[str, int] = {}
    for column, name in enumerate(layout.names):
        bounds = layout.encoding(column).driver_range
        if bounds is None:
            continue
        values = external[:, column]
        bad = int(((values < bounds[0]) | (values > bounds[1])).sum())
        if bad:
            counts[name] = bad
    return counts


__all__ = [
    "BI_OPENARM_FOLLOWER",
    "BI_PIPER_FOLLOWER",
    "DEGREES",
    "EXTERNAL_LAYOUTS",
    "ExternalJointLayout",
    "JointEncoding",
    "LIMIT_TOLERANCE_RAD",
    "RANGE_0_100",
    "RANGE_M100_100",
    "check_layout_limits",
    "clip_to_driver_range",
    "detect_external_layout",
    "external_layout_for_name",
    "external_layouts_for_robot",
    "from_canonical",
    "out_of_range_counts",
    "to_canonical",
]
