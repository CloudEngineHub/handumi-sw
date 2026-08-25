"""AgileX Piper CAN backend used by real HandUMI teleop.

The teleop script computes one IK configuration ``q`` at the live tracking
rate. This module turns that ``q`` into Piper SDK joint units and streams the
latest target on a fixed-rate CAN thread so the robot receives smooth,
bounded joint commands even if the IK loop jitters.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np
import yaml

from handumi.config import DEFAULT_RIG_CONFIG, EXAMPLE_RIG_CONFIG
from handumi.real.piper.calibrate_grippers import load_piper_gripper_zeros
from handumi.real.streamer import (
    AccelerationLimitedJointTrajectory,
    next_periodic_deadline,
)

if TYPE_CHECKING:
    from handumi.robots.registry import RobotRealConfig

log = logging.getLogger("handumi.real.piper")

RAD_TO_MDEG = 1000.0 * 180.0 / np.pi
MDEG_TO_RAD = 1.0 / RAD_TO_MDEG
ARM_JOINT_COUNT = 6
SIDE_NAMES = ("left", "right")
GRIPPER_FEEDBACK_STALE_S = 0.25


@dataclass(frozen=True)
class PiperGripperRange:
    """One Piper gripper command range, in the SDK's micrometer units."""

    min_microm: int
    max_microm: int
    source: str

    def __post_init__(self) -> None:
        if self.max_microm <= self.min_microm:
            raise ValueError("gripper max_microm must be greater than min_microm")

    @property
    def min_mm(self) -> float:
        return self.min_microm / 1000.0

    @property
    def max_mm(self) -> float:
        return self.max_microm / 1000.0

    def command_for_opening(self, opening: float) -> int:
        fraction = float(np.clip(opening, 0.0, 1.0))
        return int(
            round(self.min_microm + fraction * (self.max_microm - self.min_microm))
        )


@dataclass(frozen=True)
class PiperGripperFeedback:
    """One fresh physical gripper sample and its normalized interpretation."""

    measured_microm: int
    opening: float
    gripper_range: PiperGripperRange

    @property
    def measured_mm(self) -> float:
        return self.measured_microm / 1000.0

    @property
    def calibrated_mm(self) -> float:
        """Physical travel relative to the measured closed position."""
        return (self.measured_microm - self.gripper_range.min_microm) / 1000.0


@dataclass(frozen=True)
class PiperCanSettings:
    """Resolved Piper real-teleop settings from robot defaults + local rig."""

    left_port: str
    right_port: str
    bitrate: int = 1_000_000
    restart_ms: int = 100
    command_rate_hz: float = 100.0
    max_joint_speed_deg_s: float = 180.0
    max_joint_acceleration_deg_s2: float = 720.0
    home_max_joint_speed_deg_s: float = 20.0
    home_timeout_s: float = 30.0
    home_tolerance_deg: float = 3.0
    startup_speed_percent: int = 10
    speed_percent: int = 80
    enable_timeout_s: float = 10.0
    gripper_effort: int = 1000
    left_gripper_max_width_mm: float | None = None
    right_gripper_max_width_mm: float | None = None
    left_gripper_closed_microm: int | None = None
    right_gripper_closed_microm: int | None = None


def load_piper_can_settings(
    rig_config: Path = DEFAULT_RIG_CONFIG,
    real_config: RobotRealConfig | None = None,
) -> PiperCanSettings:
    """Load Piper CAN ports from ``rig.yaml`` and command defaults from robot YAML."""
    if not rig_config.exists():
        raise SystemExit(
            f"Missing rig configuration: {rig_config}.\n"
            f"Create it with: cp {EXAMPLE_RIG_CONFIG} {DEFAULT_RIG_CONFIG}"
        )
    with rig_config.open("r", encoding="utf-8") as handle:
        rig: dict[str, Any] = yaml.safe_load(handle) or {}

    can = (((rig.get("robots") or {}).get("piper") or {}).get("can") or {})
    gripper = (((rig.get("robots") or {}).get("piper") or {}).get("gripper") or {})
    if not isinstance(can, dict):
        raise SystemExit(
            f"Missing or invalid 'robots.piper.can' section in {rig_config}."
        )
    missing = [key for key in ("left_port", "right_port") if not can.get(key)]
    if missing:
        raise SystemExit(
            f"Missing Piper CAN setting(s) in {rig_config}: {', '.join(missing)}."
        )

    # Keep this hardware-only module importable without loading the kinematics
    # stack (and therefore JAX). Teleop supplies RobotRealConfig explicitly;
    # standalone Piper utilities use the identical defaults declared here.
    defaults = real_config or PiperCanSettings(left_port="", right_port="")
    calibrated_zeros = load_piper_gripper_zeros()
    return PiperCanSettings(
        left_port=str(can["left_port"]),
        right_port=str(can["right_port"]),
        bitrate=int(can.get("bitrate", 1_000_000)),
        restart_ms=int(can.get("restart_ms", 100)),
        command_rate_hz=defaults.command_rate_hz,
        max_joint_speed_deg_s=defaults.max_joint_speed_deg_s,
        max_joint_acceleration_deg_s2=defaults.max_joint_acceleration_deg_s2,
        home_max_joint_speed_deg_s=defaults.home_max_joint_speed_deg_s,
        home_timeout_s=defaults.home_timeout_s,
        home_tolerance_deg=defaults.home_tolerance_deg,
        startup_speed_percent=defaults.startup_speed_percent,
        speed_percent=defaults.speed_percent,
        gripper_effort=defaults.gripper_effort,
        left_gripper_max_width_mm=_optional_gripper_width_mm(gripper, "left"),
        right_gripper_max_width_mm=_optional_gripper_width_mm(gripper, "right"),
        left_gripper_closed_microm=calibrated_zeros.get("left"),
        right_gripper_closed_microm=calibrated_zeros.get("right"),
    )


def _optional_gripper_width_mm(gripper: dict[str, Any], side: str) -> float | None:
    value = gripper.get(f"{side}_max_width_mm")
    if value is None:
        value_m = gripper.get(f"{side}_max_width_m")
        if value_m is None:
            return None
        value = float(value_m) * 1000.0
    value_mm = float(value)
    if value_mm <= 0.0:
        raise SystemExit(f"robots.piper.gripper.{side}_max_width_mm must be > 0.")
    return value_mm


def piper_arm_joint_indices(
    actuated_names: list[str] | tuple[str, ...], side: str
) -> list[int]:
    """Return the six Piper arm-joint indices for ``side`` in URDF order."""
    if side not in SIDE_NAMES:
        raise ValueError(f"expected side in {SIDE_NAMES}, got {side!r}")
    names = list(actuated_names)
    wanted = [f"{side}_joint{i}" for i in range(1, ARM_JOINT_COUNT + 1)]
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"missing Piper joints in URDF: {', '.join(missing)}")
    return [names.index(name) for name in wanted]


def q_to_piper_mdeg(
    q: np.ndarray,
    actuated_names: list[str] | tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Convert full robot ``q`` in radians to Piper SDK milli-degree joints."""
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    return {
        side: np.rint(q_arr[piper_arm_joint_indices(actuated_names, side)] * RAD_TO_MDEG)
        .astype(np.int64)
        .reshape(ARM_JOINT_COUNT)
        for side in SIDE_NAMES
    }


def piper_mdeg_to_q(
    *,
    left_mdeg: np.ndarray,
    right_mdeg: np.ndarray,
    actuated_names: list[str] | tuple[str, ...],
    base_q: np.ndarray,
) -> np.ndarray:
    """Write Piper feedback milli-degrees into a full robot ``q`` vector."""
    q = np.asarray(base_q, dtype=np.float32).copy()
    for side, values in (("left", left_mdeg), ("right", right_mdeg)):
        indices = piper_arm_joint_indices(actuated_names, side)
        q[indices] = (
            np.asarray(values, dtype=np.float32)[:ARM_JOINT_COUNT] * MDEG_TO_RAD
        )
    return q


def step_mdeg_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_step_mdeg: float,
) -> np.ndarray:
    """Move one command sample toward ``target`` by at most ``max_step_mdeg`` per joint."""
    current_f = np.asarray(current, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    if max_step_mdeg <= 0.0:
        return np.rint(target_f).astype(np.int64)
    delta = np.clip(
        target_f - current_f,
        -float(max_step_mdeg),
        float(max_step_mdeg),
    )
    return np.rint(current_f + delta).astype(np.int64)


def format_mdeg(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{int(v):7d}" for v in np.asarray(values).reshape(-1)) + "]"


class PiperArm(Protocol):
    """Interface that any Piper-compatible arm must implement."""

    port: str

    def read_mdeg(self) -> np.ndarray: ...
    def read_gripper_range(self, timeout_s: float = 1.0) -> PiperGripperRange: ...
    def read_gripper_microm(self) -> int | None: ...
    def disable_gripper(self) -> None: ...
    def disable_arm(self) -> None: ...
    def send_mdeg(self, cmd: np.ndarray) -> None: ...
    def send_gripper_microm(self, opening_microm: int, effort: int) -> None: ...
    def disconnect(self) -> None: ...


ArmFactory = Callable[[str, int, int, float, int], PiperArm]


class PiperSdkArm:
    """One physical Piper arm through ``piper_sdk``."""

    def __init__(
        self,
        port: str,
        speed_percent: int,
        startup_speed_percent: int,
        enable_timeout_s: float,
        gripper_effort: int,
    ) -> None:
        try:
            from piper_sdk import C_PiperInterface_V2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing piper_sdk. Install real Piper support with: uv sync --extra piper"
            ) from exc

        self.port = port
        self.speed_percent = int(speed_percent)
        self.startup_speed_percent = int(startup_speed_percent)
        if not 1 <= self.startup_speed_percent <= 100:
            raise ValueError("startup_speed_percent must be in [1, 100]")
        self.gripper_effort = int(gripper_effort)
        self.arm = C_PiperInterface_V2(port)
        self._last_gripper_feedback_timestamp = 0.0
        self._last_gripper_feedback_at_s: float | None = None
        self.arm.ConnectPort()
        time.sleep(0.2)

        initial_mdeg = self.read_mdeg()
        status = self.arm.GetArmStatus().arm_status
        log.info(
            "[%s] Initial Piper status: ctrl_mode=%d motion_status=%d.",
            self.port,
            status.ctrl_mode,
            status.motion_status,
        )

        # Reset is destructive: Piper documents that it removes motor power and
        # lets the arm fall. It is required after teach/high-follow mode or to
        # clear a controller fault, but must not run on every application restart.
        # A normal second teleop run is already in CAN/MOVE-J mode.
        if self._requires_reset(status):
            log.warning("[%s] Resetting Piper before leaving teach/fault mode.", port)
            self.arm.ResetPiper()

        # Select a deliberately slow mode before enabling. Most importantly,
        # seed the first JointCtrl with feedback captured before any reset so the
        # controller cannot execute a retained go-zero/native target at startup.
        self.set_joint_mode(speed_percent=self.startup_speed_percent)

        deadline = time.time() + float(enable_timeout_s)
        while not self.arm.EnablePiper():
            if time.time() > deadline:
                raise TimeoutError(f"{self.port}: timed out enabling Piper")
            time.sleep(0.02)
        self.send_mdeg(initial_mdeg)
        time.sleep(0.05)
        self.set_joint_mode(speed_percent=self.speed_percent)
        self.send_mdeg(initial_mdeg)

    def _requires_reset(self, status: Any) -> bool:
        if int(getattr(status, "ctrl_mode", 0)) == 0x02:
            return True
        if int(getattr(status, "arm_status", 0)) != 0:
            return True
        get_mode = getattr(self.arm, "GetArmModeCtrl", None)
        if get_mode is None:
            return False
        mode_feedback = get_mode()
        if float(getattr(mode_feedback, "time_stamp", 0.0)) <= 0.0:
            return False
        mode = getattr(mode_feedback, "ctrl_151", None)
        return int(getattr(mode, "mit_mode", 0)) == 0xAD

    def set_joint_mode(self, *, speed_percent: int | None = None) -> None:
        speed = self.speed_percent if speed_percent is None else int(speed_percent)
        self.arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)

    def read_mdeg(self) -> np.ndarray:
        joint_state = self.arm.GetArmJointMsgs().joint_state
        return np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ],
            dtype=np.int64,
        )

    def read_gripper_range(self, timeout_s: float = 1.0) -> PiperGripperRange:
        """Read this controller's gripper range, falling back to SDK parameters.

        ``GetSDKGripperRangeParam`` is only a process-local SDK setting. The
        0x477/0x47E query is what obtains the gripper type/range configured in
        this particular arm controller.
        """
        sdk_min_m, sdk_max_m = self.arm.GetSDKGripperRangeParam()
        sdk_range = PiperGripperRange(
            min_microm=int(round(float(sdk_min_m) * 1_000_000.0)),
            max_microm=int(round(float(sdk_max_m) * 1_000_000.0)),
            source="sdk",
        )
        previous = self.arm.GetGripperTeachingPendantParamFeedback()
        previous_timestamp = float(previous.time_stamp)
        self.arm.ArmParamEnquiryAndConfig(0x04)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() <= deadline:
            feedback = self.arm.GetGripperTeachingPendantParamFeedback()
            timestamp = float(feedback.time_stamp)
            params = feedback.arm_gripper_teaching_param_feedback
            max_range_mm = int(params.max_range_config)
            if timestamp > previous_timestamp and max_range_mm > 0:
                return PiperGripperRange(
                    min_microm=sdk_range.min_microm,
                    max_microm=max_range_mm * 1000,
                    source="controller",
                )
            time.sleep(0.02)
        log.warning(
            "[%s] Controller did not return gripper parameters; using SDK range "
            "%.1f..%.1f mm.",
            self.port,
            sdk_range.min_mm,
            sdk_range.max_mm,
        )
        return sdk_range

    def read_gripper_microm(self) -> int | None:
        """Return fresh measured gripper travel, never the commanded target."""
        feedback = self.arm.GetArmGripperMsgs()
        timestamp = float(feedback.time_stamp)
        if timestamp <= 0.0:
            return None
        now_s = time.monotonic()
        if timestamp != self._last_gripper_feedback_timestamp:
            self._last_gripper_feedback_timestamp = timestamp
            self._last_gripper_feedback_at_s = now_s
        if (
            self._last_gripper_feedback_at_s is None
            or now_s - self._last_gripper_feedback_at_s > GRIPPER_FEEDBACK_STALE_S
        ):
            return None
        return int(feedback.gripper_state.grippers_angle)

    def disable_gripper(self) -> None:
        """Release the gripper motor so it can be positioned by hand."""
        self.arm.GripperCtrl(0, self.gripper_effort, 0x00, 0x00)

    def disable_arm(self) -> None:
        """Release all six arm-joint motors so the arm can be moved by hand."""
        self.arm.DisableArm(7, 0x01)

    def send_mdeg(self, cmd: np.ndarray) -> None:
        values = [int(v) for v in np.asarray(cmd, dtype=np.int64)[:ARM_JOINT_COUNT]]
        self.arm.JointCtrl(*values)

    def send_gripper_microm(self, opening_microm: int, effort: int) -> None:
        self.arm.GripperCtrl(int(opening_microm), int(effort), 0x01, 0)

    def disconnect(self) -> None:
        disconnect = getattr(self.arm, "DisconnectPort", None)
        if disconnect is not None:
            disconnect()


class PiperJointStreamer:
    """Latest-target, fixed-rate sender for both real Piper arms."""

    def __init__(
        self,
        arms: dict[str, PiperArm],
        *,
        command_rate_hz: float,
        max_joint_speed_deg_s: float,
        max_joint_acceleration_deg_s2: float | None = None,
        gripper_effort: int,
    ) -> None:
        if max_joint_speed_deg_s <= 0.0:
            raise ValueError("max_joint_speed_deg_s must be > 0")
        if not arms:
            raise ValueError("no Piper arms connected")
        if command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be > 0")
        self.arms = arms
        self.command_rate_hz = float(command_rate_hz)
        self.gripper_effort = int(gripper_effort)
        self.max_step_mdeg = max(1.0, max_joint_speed_deg_s * 1000.0 / command_rate_hz)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._commanded = {
            side: arm.read_mdeg().astype(np.int64) for side, arm in self.arms.items()
        }
        self._targets = {side: cmd.copy() for side, cmd in self._commanded.items()}
        acceleration = (
            max_joint_speed_deg_s * command_rate_hz
            if max_joint_acceleration_deg_s2 is None
            else max_joint_acceleration_deg_s2
        )
        if acceleration <= 0.0:
            raise ValueError("max_joint_acceleration_deg_s2 must be > 0")
        self.max_acceleration_mdeg_s2 = acceleration * 1000.0
        self._trajectories = {
            side: AccelerationLimitedJointTrajectory(
                cmd,
                sample_rate_hz=command_rate_hz,
                max_velocity=max_joint_speed_deg_s * 1000.0,
                max_acceleration=self.max_acceleration_mdeg_s2,
            )
            for side, cmd in self._commanded.items()
        }
        self._gripper_targets: dict[str, int | None] = {
            side: None for side in self.arms
        }
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-piper-can-streamer",
            daemon=True,
        )

    def start(self) -> None:
        log.info(
            "Piper stream: %.1f Hz, velocity <= %.1f deg/s, "
            "acceleration <= %.1f deg/s^2",
            self.command_rate_hz,
            self.max_step_mdeg * self.command_rate_hz / 1000.0,
            self.max_acceleration_mdeg_s2 / 1000.0,
        )
        self._thread.start()

    def set_max_joint_speed_deg_s(self, max_joint_speed_deg_s: float) -> None:
        if max_joint_speed_deg_s <= 0.0:
            raise ValueError("max_joint_speed_deg_s must be > 0")
        with self._lock:
            self.max_step_mdeg = max(
                1.0,
                max_joint_speed_deg_s * 1000.0 / self.command_rate_hz,
            )
            for trajectory in self._trajectories.values():
                trajectory.set_limits(max_velocity=max_joint_speed_deg_s * 1000.0)
        log.info("Piper stream max step: %.3f deg/tick", self.max_step_mdeg / 1000.0)

    def set_targets(self, targets: dict[str, np.ndarray]) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                if side in self._targets:
                    self._targets[side] = (
                        np.asarray(target, dtype=np.int64)[:ARM_JOINT_COUNT].copy()
                    )

    def set_gripper_targets_microm(self, targets: dict[str, int | None]) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                if side in self._gripper_targets:
                    self._gripper_targets[side] = (
                        None if target is None else int(target)
                    )

    def latest_commands(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: cmd.copy() for side, cmd in self._commanded.items()}

    def hold_current_commands(self) -> dict[str, np.ndarray]:
        """Cancel pending motion and hold the latest scheduled joint commands."""
        self.raise_if_failed()
        with self._lock:
            held = {side: cmd.copy() for side, cmd in self._commanded.items()}
            self._targets = {side: cmd.copy() for side, cmd in held.items()}
            for side, cmd in held.items():
                self._trajectories[side].reset(cmd)
        return held

    def feedback_mdeg(self) -> dict[str, np.ndarray]:
        return {
            side: arm.read_mdeg().astype(np.int64) for side, arm in self.arms.items()
        }

    def max_command_error_mdeg(self) -> float:
        with self._lock:
            errors = [
                float(np.max(np.abs(self._commanded[side] - target)))
                for side, target in self._targets.items()
            ]
        return max(errors, default=0.0)

    def max_feedback_error_mdeg(self) -> float:
        with self._lock:
            targets = {side: target.copy() for side, target in self._targets.items()}
        errors = []
        for side, target in targets.items():
            feedback = self.arms[side].read_mdeg()
            errors.append(float(np.max(np.abs(feedback - target))))
        return max(errors, default=0.0)

    def wait_until_targets(
        self,
        *,
        timeout_s: float,
        tolerance_mdeg: float,
        print_period_s: float = 0.5,
    ) -> None:
        deadline = time.perf_counter() + float(timeout_s)
        last_print = 0.0
        while True:
            self.raise_if_failed()
            cmd_error = self.max_command_error_mdeg()
            feedback_error = self.max_feedback_error_mdeg()
            if max(cmd_error, feedback_error) <= tolerance_mdeg:
                return
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError(
                    "timed out waiting for Piper target "
                    f"(cmd_err={cmd_error / 1000.0:.2f}deg, "
                    f"feedback_err={feedback_error / 1000.0:.2f}deg)"
                )
            if now - last_print >= print_period_s:
                log.info(
                    "Piper homing: cmd_err=%.2fdeg feedback_err=%.2fdeg",
                    cmd_error / 1000.0,
                    feedback_error / 1000.0,
                )
                last_print = now
            time.sleep(0.05)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Piper command streamer failed") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def _run(self) -> None:
        period = 1.0 / self.command_rate_hz
        next_time = time.perf_counter()
        try:
            while not self._stop.is_set():
                with self._lock:
                    gripper_targets = self._gripper_targets.copy()
                    next_commands = {
                        side: np.rint(
                            self._trajectories[side].step(self._targets[side])
                        ).astype(np.int64)
                        for side in self.arms
                    }
                    # Publish the scheduled command before sending. A concurrent
                    # hold then captures this exact tick and prevents later motion.
                    self._commanded.update(
                        {side: cmd.copy() for side, cmd in next_commands.items()}
                    )

                for side, arm in self.arms.items():
                    cmd = next_commands[side]
                    arm.send_mdeg(cmd)
                    gripper = gripper_targets.get(side)
                    if gripper is not None:
                        arm.send_gripper_microm(gripper, self.gripper_effort)

                now = time.perf_counter()
                next_time = next_periodic_deadline(next_time, period, now)
                remaining = next_time - now
                if remaining > 0.0:
                    self._stop.wait(remaining)
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            log.error("Piper command streamer failed: %s", exc)


class PiperCanEnvironment:
    """Owns Piper CAN arms plus the smooth latest-target streamer."""

    def __init__(
        self,
        settings: PiperCanSettings,
        *,
        arm_factory: ArmFactory = PiperSdkArm,
    ) -> None:
        self.settings = settings
        self.arm_factory = arm_factory
        self.arms: dict[str, PiperArm] = {}
        self.gripper_ranges: dict[str, PiperGripperRange] = {}
        self.streamer: PiperJointStreamer | None = None

    def connect(self) -> None:
        if self.arms:
            return
        for side, port in (
            ("left", self.settings.left_port),
            ("right", self.settings.right_port),
        ):
            log.info("Connecting Piper %s on %s.", side, port)
            self.arms[side] = self.arm_factory(
                port,
                self.settings.speed_percent,
                self.settings.startup_speed_percent,
                self.settings.enable_timeout_s,
                self.settings.gripper_effort,
            )
            try:
                discovered = self.arms[side].read_gripper_range()
            except Exception as exc:  # noqa: BLE001 - discovery has safe fallback.
                log.warning(
                    "Piper %s on %s cannot report its gripper range: %s",
                    side,
                    port,
                    exc,
                )
                continue
            configured_max_mm = getattr(
                self.settings, f"{side}_gripper_max_width_mm"
            )
            if configured_max_mm is not None:
                discovered = PiperGripperRange(
                    min_microm=discovered.min_microm,
                    max_microm=int(round(configured_max_mm * 1000.0)),
                    source="rig override",
                )
            calibrated_closed = getattr(
                self.settings, f"{side}_gripper_closed_microm"
            )
            if calibrated_closed is not None:
                discovered = PiperGripperRange(
                    min_microm=int(calibrated_closed),
                    max_microm=discovered.max_microm,
                    source=f"{discovered.source} + calibrated closed zero",
                )
            self.gripper_ranges[side] = discovered
            log.info(
                "Piper %s gripper range: %.1f..%.1f mm (%s).",
                side,
                discovered.min_mm,
                discovered.max_mm,
                discovered.source,
            )

    def start_streaming_current_pose(self) -> None:
        """Start the command stream without moving arm joints."""
        if not self.arms:
            raise RuntimeError("connect() before start_streaming_current_pose()")
        if self.streamer is not None:
            return
        self.streamer = PiperJointStreamer(
            self.arms,
            command_rate_hz=self.settings.command_rate_hz,
            max_joint_speed_deg_s=self.settings.max_joint_speed_deg_s,
            max_joint_acceleration_deg_s2=self.settings.max_joint_acceleration_deg_s2,
            gripper_effort=self.settings.gripper_effort,
        )
        self.streamer.start()

    def home(self, home_targets_mdeg: dict[str, np.ndarray]) -> None:
        if not self.arms:
            raise RuntimeError("connect() before home()")
        self.streamer = PiperJointStreamer(
            self.arms,
            command_rate_hz=self.settings.command_rate_hz,
            max_joint_speed_deg_s=self.settings.home_max_joint_speed_deg_s,
            max_joint_acceleration_deg_s2=self.settings.max_joint_acceleration_deg_s2,
            gripper_effort=self.settings.gripper_effort,
        )
        self.streamer.start()
        self.move_home(home_targets_mdeg)

    def move_home(self, home_targets_mdeg: dict[str, np.ndarray]) -> None:
        """Move to home at the configured slow limit, then restore teleop speed."""
        if self.streamer is None:
            raise RuntimeError("home() before move_home()")
        log.info(
            "Homing Piper to XHUMAN pose: left=%s right=%s",
            format_mdeg(home_targets_mdeg["left"]),
            format_mdeg(home_targets_mdeg["right"]),
        )
        # A return-home request is a safety transition rather than another
        # live waypoint. Cancel any residual teleop velocity before starting
        # the deliberately slow home profile.
        self.streamer.hold_current_commands()
        self.streamer.set_max_joint_speed_deg_s(
            self.settings.home_max_joint_speed_deg_s
        )
        try:
            self.streamer.set_targets(home_targets_mdeg)
            self.streamer.wait_until_targets(
                timeout_s=self.settings.home_timeout_s,
                tolerance_mdeg=self.settings.home_tolerance_deg * 1000.0,
            )
            log.info("Piper home reached.")
        finally:
            self.streamer.set_max_joint_speed_deg_s(
                self.settings.max_joint_speed_deg_s
            )

    def set_q(self, q: np.ndarray, actuated_names: list[str] | tuple[str, ...]) -> None:
        self.set_targets(q_to_piper_mdeg(q, actuated_names))

    def set_gripper_widths_mm(self, widths_mm: dict[str, float]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before set_gripper_widths_mm()")
        targets: dict[str, int | None] = {
            side: int(round(max(0.0, float(width_mm)) * 1000.0))
            for side, width_mm in widths_mm.items()
        }
        self.streamer.set_gripper_targets_microm(targets)

    def disable_gripper(self, side: str) -> None:
        """Release one gripper without starting or changing arm joint motion."""
        try:
            arm = self.arms[side]
        except KeyError as exc:
            raise ValueError(f"Piper {side} is not connected") from exc
        arm.disable_gripper()

    def disable_arm(self, side: str) -> None:
        """Release one arm's six joint motors so it can be moved by hand.

        Stop the command streamer first if it is running: a live streamer
        would otherwise keep re-sending the last joint target on its next
        tick and re-engage the motors right after this call.
        """
        try:
            arm = self.arms[side]
        except KeyError as exc:
            raise ValueError(f"Piper {side} is not connected") from exc
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        arm.disable_arm()

    def set_gripper_openings(
        self,
        openings: dict[str, float],
        *,
        fallback_max_width_mm: float | dict[str, float],
    ) -> dict[str, int]:
        """Scale normalized openings with each connected Piper's own range."""
        if self.streamer is None:
            raise RuntimeError("home() before set_gripper_openings()")
        targets: dict[str, int] = {}
        for side, opening in openings.items():
            fallback_mm = (
                fallback_max_width_mm.get(side)
                if isinstance(fallback_max_width_mm, dict)
                else fallback_max_width_mm
            )
            if fallback_mm is None or float(fallback_mm) <= 0.0:
                raise ValueError(f"missing positive fallback width for {side}")
            gripper_range = self.gripper_ranges.get(side)
            if gripper_range is None:
                gripper_range = PiperGripperRange(
                    0,
                    int(round(float(fallback_mm) * 1000.0)),
                    "robot config fallback",
                )
            targets[side] = gripper_range.command_for_opening(opening)
        self.streamer.set_gripper_targets_microm(targets)
        return targets

    def set_targets(self, targets_mdeg: dict[str, np.ndarray]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before set_targets()")
        self.streamer.set_targets(targets_mdeg)

    def latest_commands_mdeg(self) -> dict[str, np.ndarray]:
        if self.streamer is None:
            return {}
        return self.streamer.latest_commands()

    def hold_current_commands_mdeg(self) -> dict[str, np.ndarray]:
        if self.streamer is None:
            raise RuntimeError("home() before hold_current_commands_mdeg()")
        return self.streamer.hold_current_commands()

    def feedback_mdeg(self) -> dict[str, np.ndarray]:
        return {
            side: arm.read_mdeg().astype(np.int64) for side, arm in self.arms.items()
        }

    def gripper_openings(
        self,
        *,
        fallback_max_width_mm: float | dict[str, float],
    ) -> dict[str, float]:
        """Return normalized physical gripper feedback for fresh CAN samples."""
        return {
            side: feedback.opening
            for side, feedback in self.gripper_feedback(
                fallback_max_width_mm=fallback_max_width_mm
            ).items()
        }

    def gripper_feedback(
        self,
        *,
        fallback_max_width_mm: float | dict[str, float],
    ) -> dict[str, PiperGripperFeedback]:
        """Return detailed fresh gripper feedback for runtime diagnostics."""
        samples: dict[str, PiperGripperFeedback] = {}
        for side, arm in self.arms.items():
            measured_microm = arm.read_gripper_microm()
            if measured_microm is None:
                continue
            gripper_range = self.gripper_ranges.get(side)
            if gripper_range is None:
                fallback_mm = (
                    fallback_max_width_mm.get(side)
                    if isinstance(fallback_max_width_mm, dict)
                    else fallback_max_width_mm
                )
                if fallback_mm is None or float(fallback_mm) <= 0.0:
                    continue
                gripper_range = PiperGripperRange(
                    0,
                    int(round(float(fallback_mm) * 1000.0)),
                    "robot config fallback",
                )
            span = gripper_range.max_microm - gripper_range.min_microm
            opening = float(
                np.clip(
                    (measured_microm - gripper_range.min_microm) / span,
                    0.0,
                    1.0,
                )
            )
            samples[side] = PiperGripperFeedback(
                measured_microm=measured_microm,
                opening=opening,
                gripper_range=gripper_range,
            )
        return samples

    def raise_if_failed(self) -> None:
        if self.streamer is not None:
            self.streamer.raise_if_failed()

    def close(self) -> None:
        try:
            if self.streamer is not None:
                self.streamer.stop()
        finally:
            for arm in self.arms.values():
                try:
                    arm.disconnect()
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    log.warning("Failed to disconnect Piper %s: %s", arm.port, exc)
            self.arms.clear()
            self.gripper_ranges.clear()
            self.streamer = None


__all__ = [
    "MDEG_TO_RAD",
    "RAD_TO_MDEG",
    "PiperCanEnvironment",
    "PiperCanSettings",
    "PiperGripperFeedback",
    "PiperJointStreamer",
    "PiperGripperRange",
    "PiperSdkArm",
    "format_mdeg",
    "load_piper_can_settings",
    "piper_arm_joint_indices",
    "piper_mdeg_to_q",
    "q_to_piper_mdeg",
    "step_mdeg_toward",
]
