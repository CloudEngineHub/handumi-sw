import numpy as np

from handumi.real.piper import driver
from handumi.real.piper.calibrate_grippers import (
    load_piper_gripper_zeros,
    save_piper_gripper_zero,
)


def test_piper_gripper_zeros_are_measured_and_preserved(tmp_path) -> None:
    path = tmp_path / "piper_grippers.yaml"

    save_piper_gripper_zero("left", -5_300, path)
    save_piper_gripper_zero("right", -2_590, path)

    assert load_piper_gripper_zeros(path) == {
        "left": -5_300,
        "right": -2_590,
    }


def test_calibrated_closed_value_is_logical_zero_and_command_zero() -> None:
    gripper_range = driver.PiperGripperRange(-5_300, 100_000, "calibrated")

    assert gripper_range.command_for_opening(0.0) == -5_300
    assert gripper_range.command_for_opening(1.0) == 100_000


def test_environment_applies_each_measured_closed_value() -> None:
    class FakeArm:
        def __init__(self, port: str, *_args: object) -> None:
            self.port = port

        def read_gripper_range(self, timeout_s: float = 1.0):
            del timeout_s
            return driver.PiperGripperRange(0, 70_000, "controller")

        def read_gripper_microm(self) -> int:
            return -5_300 if self.port == "can0" else -2_590

    settings = driver.PiperCanSettings(
        left_port="can0",
        right_port="can1",
        left_gripper_closed_microm=-5_300,
        right_gripper_closed_microm=-2_590,
    )
    environment = driver.PiperCanEnvironment(
        settings, arm_factory=FakeArm  # type: ignore[arg-type]
    )

    environment.connect()

    assert environment.gripper_ranges["left"].min_microm == -5_300
    assert environment.gripper_ranges["right"].min_microm == -2_590
    assert environment.gripper_ranges["left"].command_for_opening(0.0) == -5_300
    assert environment.gripper_ranges["right"].command_for_opening(0.0) == -2_590
    assert environment.gripper_openings(fallback_max_width_mm=66.0) == {
        "left": 0.0,
        "right": 0.0,
    }
    feedback = environment.gripper_feedback(fallback_max_width_mm=66.0)
    assert feedback["left"].calibrated_mm == 0.0
    assert feedback["right"].calibrated_mm == 0.0


def test_streamer_preserves_negative_calibrated_gripper_target() -> None:
    class FakeArm:
        def read_mdeg(self) -> np.ndarray:
            return np.zeros(6, dtype=np.int64)

    streamer = driver.PiperJointStreamer(
        {"left": FakeArm()},  # type: ignore[dict-item]
        command_rate_hz=100.0,
        max_joint_speed_deg_s=180.0,
        gripper_effort=1000,
    )

    streamer.set_gripper_targets_microm({"left": -5_300})

    assert streamer._gripper_targets["left"] == -5_300  # noqa: SLF001
