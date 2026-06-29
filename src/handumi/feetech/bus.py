"""Low-level Feetech servo bus access for HandUMI gripper encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class FeetechUnavailableError(RuntimeError):
    """Raised when the Feetech SDK is not available in the active environment."""


@dataclass
class FeetechBus:
    port: str
    baudrate: int = 1_000_000
    protocol_version: int = 0

    def __post_init__(self) -> None:
        self._sdk: Any | None = None
        self._port_handler: Any | None = None
        self._packet_handler: Any | None = None

    def open(self) -> None:
        try:
            import scservo_sdk as sdk
        except ImportError as exc:
            raise FeetechUnavailableError(
                "Feetech SDK is not importable. Install/sync lerobot[feetech] "
                "or feetech-servo-sdk in this environment."
            ) from exc

        port_handler = sdk.PortHandler(self.port)
        if not port_handler.openPort():
            raise RuntimeError(f"Could not open Feetech port {self.port}.")
        if not port_handler.setBaudRate(self.baudrate):
            port_handler.closePort()
            raise RuntimeError(f"Could not set Feetech baudrate {self.baudrate}.")

        self._sdk = sdk
        self._port_handler = port_handler
        self._packet_handler = sdk.PacketHandler(self.protocol_version)

    def close(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._sdk = None
        self._port_handler = None
        self._packet_handler = None

    def __enter__(self) -> "FeetechBus":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ping(self, servo_id: int) -> bool:
        packet = self._require_packet()
        _, comm, error = packet.ping(self._require_port(), int(servo_id))
        return _comm_success(comm, self._sdk) and _no_error(error)

    def ping_model(self, servo_id: int) -> int | None:
        packet = self._require_packet()
        model_number, comm, error = packet.ping(self._require_port(), int(servo_id))
        if not _comm_success(comm, self._sdk) or not _no_error(error):
            return None
        return int(model_number)

    def scan(self, ids: Iterable[int]) -> list[int]:
        return [int(servo_id) for servo_id in ids if self.ping(int(servo_id))]

    def read_position(self, servo_id: int) -> int:
        packet = self._require_packet()
        value, comm, error = packet.read2ByteTxRx(
            self._require_port(),
            int(servo_id),
            _PRESENT_POSITION_ADDR,
        )
        if not _comm_success(comm, self._sdk) or not _no_error(error):
            raise RuntimeError(f"Failed to read Present_Position from servo {servo_id}.")
        return int(value)

    def write_servo_id(self, old_id: int, new_id: int) -> None:
        self.disable_torque(old_id)
        self._write_1_byte(old_id, _ID_ADDR, new_id, "ID")

    def disable_torque(self, servo_id: int) -> None:
        self._write_1_byte(servo_id, _TORQUE_ENABLE_ADDR, 0, "Torque_Enable")
        try:
            self._write_1_byte(servo_id, _LOCK_ADDR, 0, "Lock")
        except RuntimeError:
            pass

    def _write_1_byte(self, servo_id: int, address: int, value: int, name: str) -> None:
        packet = self._require_packet()
        comm, error = packet.write1ByteTxRx(
            self._require_port(),
            int(servo_id),
            int(address),
            int(value),
        )
        if not _comm_success(comm, self._sdk) or not _no_error(error):
            raise RuntimeError(f"Failed to write {name}={value} on servo {servo_id}.")

    def _require_packet(self):
        if self._packet_handler is None:
            raise RuntimeError("Feetech bus is not open.")
        return self._packet_handler

    def _require_port(self):
        if self._port_handler is None:
            raise RuntimeError("Feetech bus is not open.")
        return self._port_handler


def _comm_success(result: Any, sdk: Any | None) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, tuple):
        comm_result = result[-2] if len(result) >= 2 else result[-1]
    else:
        comm_result = result
    success = getattr(sdk, "COMM_SUCCESS", 0)
    return comm_result == success or comm_result == 0


def _no_error(error: Any) -> bool:
    return int(error) == 0


_ID_ADDR = 5
_TORQUE_ENABLE_ADDR = 40
_LOCK_ADDR = 55
_PRESENT_POSITION_ADDR = 56
