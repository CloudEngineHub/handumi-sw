from argparse import Namespace
from pathlib import Path

from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.scripts.setup import calibrate_grippers


def test_calibration_is_default_command() -> None:
    args = calibrate_grippers.parse_args(["--side", "right"])

    assert args.command is None
    assert args.side == "right"


def test_monitor_remains_available() -> None:
    args = calibrate_grippers.parse_args(["monitor", "--duration-s", "5"])

    assert args.command == "monitor"
    assert args.duration_s == 5


def test_guided_calibration_prompts_width_then_homes_then_measures(
    monkeypatch,
) -> None:
    current = FeetechConfig(
        port="/dev/ttyUSB0",
        baudrate=1_000_000,
        protocol_version=0,
        left=GripperCalibration(servo_id=0),
        right=GripperCalibration(servo_id=1),
    )
    events: list[str] = []

    monkeypatch.setattr(calibrate_grippers, "load_config", lambda *_args: current)

    def prompt(label: str) -> float:
        events.append(f"width:{label}")
        return 72.5

    def home(**kwargs) -> None:
        events.append(f"home:{kwargs['side']}")

    def measure(**kwargs) -> tuple[int, int, float]:
        events.append(f"measure:{kwargs['side']}")
        assert kwargs["max_width_mm"] == 72.5
        return 100, 900, kwargs["max_width_mm"]

    saved: list[FeetechConfig] = []
    monkeypatch.setattr(calibrate_grippers, "_prompt_positive_float", prompt)
    monkeypatch.setattr(calibrate_grippers.home_servos, "_home_side", home)
    monkeypatch.setattr(calibrate_grippers, "_calibrate_side", measure)
    monkeypatch.setattr(
        calibrate_grippers,
        "save_calibration",
        lambda config, path: saved.append(config) or path,
    )

    calibrate_grippers.cmd_calibrate(
        Namespace(
            rig_config=Path("rig.yaml"),
            calibration_config=Path("calibration.yaml"),
            side="right",
            max_width_mm=None,
            left_max_width_mm=None,
            right_max_width_mm=None,
            interval_s=0.1,
        )
    )

    assert events == [
        "width:right max gripper opening in mm",
        "home:right",
        "measure:right",
    ]
    assert saved[0].right == GripperCalibration(1, 100, 900, 72.5)
