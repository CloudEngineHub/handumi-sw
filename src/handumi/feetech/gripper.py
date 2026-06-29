"""HandUMI gripper aperture sensing backed by Feetech servo encoders."""

from __future__ import annotations

from dataclasses import dataclass

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import FeetechConfig, GripperCalibration


@dataclass(frozen=True)
class GripperWidths:
    left: float
    right: float
    left_mm: float
    right_mm: float
    left_normalized: float
    right_normalized: float
    left_ticks: int
    right_ticks: int


class FeetechGripperPair:
    def __init__(self, config: FeetechConfig) -> None:
        self.config = config
        left_port = _side_port(config, config.left)
        right_port = _side_port(config, config.right)
        self._buses: dict[str, FeetechBus] = {}
        for port in {left_port, right_port}:
            self._buses[port] = FeetechBus(
                port=port,
                baudrate=config.baudrate,
                protocol_version=config.protocol_version,
            )
        self._left_port = left_port
        self._right_port = right_port

    def open(self) -> None:
        for bus in self._buses.values():
            bus.open()

    def close(self) -> None:
        for bus in self._buses.values():
            bus.close()

    def __enter__(self) -> "FeetechGripperPair":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_normalized_widths(self) -> GripperWidths:
        left = _read_width(self._buses[self._left_port], self.config.left)
        right = _read_width(self._buses[self._right_port], self.config.right)
        return GripperWidths(
            left=left["width_m"],
            right=right["width_m"],
            left_mm=left["width_mm"],
            right_mm=right["width_mm"],
            left_normalized=left["normalized"],
            right_normalized=right["normalized"],
            left_ticks=left["ticks"],
            right_ticks=right["ticks"],
        )


def _read_width(bus: FeetechBus, calibration: GripperCalibration) -> dict[str, float | int]:
    ticks = bus.read_position(calibration.servo_id)
    normalized = calibration.normalized_width(ticks)
    width_mm = calibration.width_mm(ticks)
    return {
        "ticks": ticks,
        "normalized": normalized,
        "width_mm": width_mm,
        "width_m": width_mm / 1000.0,
    }


def _side_port(config: FeetechConfig, calibration: GripperCalibration) -> str:
    port = calibration.port or config.port
    if not port:
        raise ValueError(
            "Feetech port is not configured. Set a shared `port` or per-side `left.port` / `right.port`."
        )
    return port
