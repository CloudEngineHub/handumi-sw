"""Optional OpenArm v1 backend using the official ``openarm_can`` bindings."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol

import numpy as np
import yaml

log = logging.getLogger(__name__)

SIDES: tuple[str, str] = ("left", "right")
ARM_DOF = 7
SEND_CAN_IDS = tuple(range(0x01, 0x08))
RECV_CAN_IDS = tuple(range(0x11, 0x18))
GRIPPER_SEND_CAN_ID = 0x08
GRIPPER_RECV_CAN_ID = 0x18
DEFAULT_KP = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
DEFAULT_KD = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)


@dataclass(frozen=True)
class OpenArmCanSettings:
    left_port: str = "can1"
    right_port: str = "can0"
    enable_fd: bool = True
    bitrate: int = 1_000_000
    dbitrate: int = 5_000_000
    command_rate_hz: float = 100.0
    max_joint_speed_rad_s: float = 1.0
    home_max_joint_speed_rad_s: float = 0.25
    home_timeout_s: float = 30.0
    home_tolerance_rad: float = 0.05
    watchdog_timeout_s: float = 0.15
    following_error_rad: float = 0.35
    kp: tuple[float, ...] = DEFAULT_KP
    kd: tuple[float, ...] = DEFAULT_KD


def load_openarm_settings(
    rig_config: Path,
    robot_real: dict[str, Any] | None = None,
) -> OpenArmCanSettings:
    """Combine portable robot defaults with machine-local CAN assignments."""
    robot_real = robot_real or {}
    data: dict[str, Any] = {}
    if rig_config.exists():
        with rig_config.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    can = ((data.get("robots") or {}).get("openarmv1") or {}).get("can") or {}
    control = robot_real.get("control") or {}
    gains = robot_real.get("gains") or {}
    return OpenArmCanSettings(
        left_port=str(can.get("left_port", "can1")),
        right_port=str(can.get("right_port", "can0")),
        enable_fd=bool(can.get("fd", True)),
        bitrate=int(can.get("bitrate", 1_000_000)),
        dbitrate=int(can.get("dbitrate", 5_000_000)),
        command_rate_hz=float(control.get("command_rate_hz", 100.0)),
        max_joint_speed_rad_s=float(control.get("max_joint_speed_rad_s", 1.0)),
        home_max_joint_speed_rad_s=float(
            control.get("home_max_joint_speed_rad_s", 0.25)
        ),
        home_timeout_s=float(control.get("home_timeout_s", 30.0)),
        home_tolerance_rad=float(control.get("home_tolerance_rad", 0.05)),
        watchdog_timeout_s=float(control.get("watchdog_timeout_s", 0.15)),
        following_error_rad=float(control.get("following_error_rad", 0.35)),
        kp=tuple(float(v) for v in gains.get("kp", DEFAULT_KP)),
        kd=tuple(float(v) for v in gains.get("kd", DEFAULT_KD)),
    )


def require_openarm_can() -> ModuleType:
    try:
        return importlib.import_module("openarm_can")
    except ImportError as exc:
        raise RuntimeError(
            "OpenArm real support is optional. Install the official C++ library "
            "and run `uv sync --extra openarm`."
        ) from exc


class OpenArmSide(Protocol):
    port: str

    def read_q(self) -> np.ndarray: ...

    def send(self, q: np.ndarray, gripper_opening: float) -> None: ...

    def close(self) -> None: ...


class OpenArmSdkSide:
    """One physical arm; all unstable SDK calls are contained here."""

    def __init__(
        self,
        port: str,
        *,
        enable_fd: bool,
        kp: tuple[float, ...],
        kd: tuple[float, ...],
        sdk: ModuleType | None = None,
    ) -> None:
        self.port = port
        self.sdk = sdk or require_openarm_can()
        self.kp = kp
        self.kd = kd
        motor_types = [
            self.sdk.MotorType.DM8009,
            self.sdk.MotorType.DM8009,
            self.sdk.MotorType.DM4340,
            self.sdk.MotorType.DM4340,
            self.sdk.MotorType.DM4310,
            self.sdk.MotorType.DM4310,
            self.sdk.MotorType.DM4310,
        ]
        self.arm = self.sdk.OpenArm(port, enable_fd)
        self.arm.init_arm_motors(motor_types, list(SEND_CAN_IDS), list(RECV_CAN_IDS))
        self.arm.init_gripper_motor(
            self.sdk.MotorType.DM4310,
            GRIPPER_SEND_CAN_ID,
            GRIPPER_RECV_CAN_ID,
        )
        self.arm.set_callback_mode_all(self.sdk.CallbackMode.STATE)
        self.arm.enable_all()
        self.arm.recv_all(2_000)

    def read_q(self) -> np.ndarray:
        self.arm.refresh_all()
        self.arm.recv_all(500)
        motors = self.arm.get_arm().get_motors()
        values = np.asarray(
            [motor.get_position() for motor in motors], dtype=np.float32
        )
        if values.shape != (ARM_DOF,):
            raise RuntimeError(
                f"OpenArm {self.port} returned {len(values)} joints; expected {ARM_DOF}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"OpenArm {self.port} returned non-finite joint feedback."
            )
        return values

    def send(self, q: np.ndarray, gripper_opening: float) -> None:
        params = [
            self.sdk.MITParam(kp, kd, float(target), 0.0, 0.0)
            for kp, kd, target in zip(self.kp, self.kd, q, strict=True)
        ]
        self.arm.get_arm().mit_control_all(params)
        # The high-level API defines 0=closed and 1=open independently of the
        # temporary motor-radian convention used by the bindings.
        self.arm.get_gripper().set_position(float(np.clip(gripper_opening, 0.0, 1.0)))
        self.arm.recv_all(500)

    def close(self) -> None:
        self.arm.disable_all()
        self.arm.recv_all(1_000)


SideFactory = Callable[..., OpenArmSide]


class OpenArmJointStreamer:
    """Velocity-limited latest-target streamer with a stale-command hold."""

    def __init__(
        self,
        arms: dict[str, OpenArmSide],
        settings: OpenArmCanSettings,
        initial_q: dict[str, np.ndarray],
    ) -> None:
        self.arms = arms
        self.settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="openarm-can", daemon=True
        )
        self._error: BaseException | None = None
        self._targets = {side: q.copy() for side, q in initial_q.items()}
        self._commanded = {side: q.copy() for side, q in initial_q.items()}
        self._feedback = {side: q.copy() for side, q in initial_q.items()}
        self._grippers = {side: 1.0 for side in SIDES}
        self._last_target_at = time.monotonic()
        self._max_speed = settings.max_joint_speed_rad_s

    def start(self) -> None:
        self._thread.start()

    def set_max_speed(self, value: float) -> None:
        with self._lock:
            self._max_speed = float(value)

    def set_targets(
        self,
        targets: dict[str, np.ndarray],
        grippers: dict[str, float] | None = None,
    ) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                q = np.asarray(target, dtype=np.float32)
                if q.shape != (ARM_DOF,) or not np.all(np.isfinite(q)):
                    raise ValueError(
                        f"Invalid OpenArm target for {side}: shape={q.shape}"
                    )
                self._targets[side] = q.copy()
            if grippers:
                self._grippers.update(
                    {
                        side: float(np.clip(value, 0.0, 1.0))
                        for side, value in grippers.items()
                    }
                )
            self._last_target_at = time.monotonic()

    def hold(self) -> dict[str, np.ndarray]:
        self.raise_if_failed()
        with self._lock:
            held = {side: q.copy() for side, q in self._feedback.items()}
            self._targets = {side: q.copy() for side, q in held.items()}
            self._commanded = {side: q.copy() for side, q in held.items()}
            self._last_target_at = time.monotonic()
        return held

    def feedback(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: q.copy() for side, q in self._feedback.items()}

    def wait_until_targets(self, *, timeout_s: float, tolerance_rad: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self.raise_if_failed()
            with self._lock:
                errors = [
                    float(np.max(np.abs(self._feedback[side] - self._targets[side])))
                    for side in self.arms
                ]
            if max(errors, default=0.0) <= tolerance_rad:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"OpenArm home timeout (max error={max(errors):.3f} rad)."
                )
            time.sleep(0.05)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("OpenArm command streamer failed") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def _run(self) -> None:
        period = 1.0 / self.settings.command_rate_hz
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    if now - self._last_target_at > self.settings.watchdog_timeout_s:
                        self._targets = {
                            side: q.copy() for side, q in self._commanded.items()
                        }
                    step = self._max_speed * period
                    commands = {
                        side: self._commanded[side]
                        + np.clip(
                            self._targets[side] - self._commanded[side], -step, step
                        )
                        for side in self.arms
                    }
                    grippers = self._grippers.copy()
                    self._commanded = {side: q.copy() for side, q in commands.items()}

                feedback: dict[str, np.ndarray] = {}
                for side, arm in self.arms.items():
                    arm.send(commands[side], grippers[side])
                    feedback[side] = arm.read_q()
                    error = float(np.max(np.abs(feedback[side] - commands[side])))
                    if error > self.settings.following_error_rad:
                        raise RuntimeError(
                            f"OpenArm {side} following error {error:.3f} rad exceeds "
                            f"{self.settings.following_error_rad:.3f} rad."
                        )
                with self._lock:
                    self._feedback = feedback
                next_tick += period
                if (remaining := next_tick - time.monotonic()) > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.monotonic()
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            log.error("OpenArm command streamer failed: %s", exc)


class OpenArmCanEnvironment:
    """Bimanual OpenArm backend implementing the generic teleop contract."""

    def __init__(
        self,
        settings: OpenArmCanSettings,
        *,
        side_factory: SideFactory = OpenArmSdkSide,
        active_sides: tuple[str, ...] = SIDES,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not active_sides or any(side not in SIDES for side in active_sides):
            raise ValueError(f"Invalid OpenArm active sides: {active_sides}")
        self.settings = settings
        self.side_factory = side_factory
        self.active_sides = tuple(dict.fromkeys(active_sides))
        self.joint_limits = joint_limits or {}
        self.arms: dict[str, OpenArmSide] = {}
        self.streamer: OpenArmJointStreamer | None = None

    def connect(self) -> None:
        if self.arms:
            return
        ports = {"left": self.settings.left_port, "right": self.settings.right_port}
        for side in self.active_sides:
            port = ports[side]
            log.info("Connecting OpenArm %s on %s.", side, port)
            self.arms[side] = self.side_factory(
                port,
                enable_fd=self.settings.enable_fd,
                kp=self.settings.kp,
                kd=self.settings.kd,
            )

    def prepare(self, *, repair: bool = True) -> None:
        from handumi.real.can_setup import ensure_can_fd_interfaces_ready

        ensure_can_fd_interfaces_ready(
            [
                self.settings.left_port if side == "left" else self.settings.right_port
                for side in self.active_sides
            ],
            bitrate=self.settings.bitrate,
            dbitrate=self.settings.dbitrate,
            repair=repair,
        )

    def _split_q(self, q: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for side in self.active_sides:
            names = [f"openarm_{side}_joint{i}" for i in range(1, ARM_DOF + 1)]
            values = np.asarray(
                [q[joint_names.index(name)] for name in names], dtype=np.float32
            )
            for name, value in zip(names, values, strict=True):
                limits = self.joint_limits.get(name)
                if limits is None:
                    continue
                lower, upper = limits
                if float(value) < lower - 1e-5 or float(value) > upper + 1e-5:
                    raise ValueError(
                        f"OpenArm target {name}={float(value):.4f} is outside "
                        f"URDF limits [{lower:.4f}, {upper:.4f}]."
                    )
            result[side] = values
        return result

    @staticmethod
    def _merge_q(
        arm_q: dict[str, np.ndarray], base_q: np.ndarray, joint_names: list[str]
    ) -> np.ndarray:
        q = np.asarray(base_q, dtype=np.float32).copy()
        for side, values in arm_q.items():
            for i, value in enumerate(values, start=1):
                q[joint_names.index(f"openarm_{side}_joint{i}")] = value
        return q

    def home(self, q: np.ndarray, joint_names: list[str]) -> None:
        if not self.arms:
            raise RuntimeError("connect() before home()")
        initial = {side: arm.read_q() for side, arm in self.arms.items()}
        self.streamer = OpenArmJointStreamer(self.arms, self.settings, initial)
        self.streamer.start()
        self.move_home(q, joint_names)

    def move_home(self, q: np.ndarray, joint_names: list[str]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before move_home()")
        self.streamer.set_max_speed(self.settings.home_max_joint_speed_rad_s)
        try:
            self.streamer.set_targets(
                self._split_q(q, joint_names),
                {side: 1.0 for side in self.active_sides},
            )
            self.streamer.wait_until_targets(
                timeout_s=self.settings.home_timeout_s,
                tolerance_rad=self.settings.home_tolerance_rad,
            )
        finally:
            self.streamer.set_max_speed(self.settings.max_joint_speed_rad_s)

    def command(
        self,
        q: np.ndarray,
        joint_names: list[str],
        gripper_openings: dict[str, float],
    ) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before command()")
        self.streamer.set_targets(self._split_q(q, joint_names), gripper_openings)

    def hold(self, base_q: np.ndarray, joint_names: list[str]) -> np.ndarray:
        if self.streamer is None:
            raise RuntimeError("home() before hold()")
        return self._merge_q(self.streamer.hold(), base_q, joint_names)

    def check_health(self) -> None:
        if self.streamer is not None:
            self.streamer.raise_if_failed()

    def close(self) -> None:
        error: BaseException | None = None
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except BaseException as exc:  # still disable every arm
                error = exc
        for side, arm in list(self.arms.items()):
            try:
                arm.close()
            except Exception as exc:  # pragma: no cover - hardware cleanup
                log.warning("Failed to disable OpenArm %s: %s", side, exc)
        self.arms.clear()
        self.streamer = None
        if error is not None:
            raise error


__all__ = [
    "ARM_DOF",
    "OpenArmCanEnvironment",
    "OpenArmCanSettings",
    "OpenArmJointStreamer",
    "load_openarm_settings",
    "require_openarm_can",
]
