import tempfile
import time
from pathlib import Path

import numpy as np

from handumi.real.openarm_can import (
    OpenArmCanEnvironment,
    OpenArmCanSettings,
    load_openarm_settings,
)
from handumi.robots.registry import load_embodiment


class FakeSide:
    instances: list["FakeSide"] = []
    port: str

    def __init__(self, port, **_kwargs):
        self.port = port
        self.q = np.zeros(7, dtype=np.float32)
        self.gripper = 1.0
        self.closed = False
        self.sent = []
        self.instances.append(self)

    def read_q(self):
        return self.q.copy()

    def send(self, q, gripper_opening):
        self.q = np.asarray(q, dtype=np.float32).copy()
        self.gripper = float(gripper_opening)
        self.sent.append(self.q.copy())

    def close(self):
        self.closed = True


def test_settings_combine_rig_ports_and_robot_control_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        rig = Path(tmp) / "rig.yaml"
        rig.write_text(
            "robots:\n"
            "  openarmv1:\n"
            "    can:\n"
            "      fd: true\n"
            "      bitrate: 1000000\n"
            "      dbitrate: 5000000\n"
            "      left_port: can7\n"
            "      right_port: can6\n",
            encoding="utf-8",
        )
        settings = load_openarm_settings(
            rig,
            {"control": {"command_rate_hz": 80, "watchdog_timeout_s": 0.2}},
        )

    assert settings.left_port == "can7"
    assert settings.right_port == "can6"
    assert settings.enable_fd
    assert settings.command_rate_hz == 80
    assert settings.watchdog_timeout_s == 0.2


def test_environment_streams_holds_and_disables_both_arms():
    FakeSide.instances.clear()
    settings = OpenArmCanSettings(
        command_rate_hz=500.0,
        max_joint_speed_rad_s=100.0,
        home_max_joint_speed_rad_s=100.0,
        watchdog_timeout_s=0.1,
    )
    environment = OpenArmCanEnvironment(settings, side_factory=FakeSide)
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    home = runtime.home_q("down")

    environment.connect()
    environment.home(home, names)
    target = home.copy()
    target[names.index("openarm_left_joint1")] = 0.2
    target[names.index("openarm_right_joint2")] = -0.1
    environment.command(target, names, {"left": 0.25, "right": 0.75})
    time.sleep(0.03)
    environment.check_health()
    held = environment.hold(home, names)
    environment.close()

    assert held[names.index("openarm_left_joint1")] > 0.0
    assert all(side.closed for side in FakeSide.instances)
    assert {round(side.gripper, 2) for side in FakeSide.instances} == {0.25, 0.75}


def test_invalid_target_shape_is_rejected():
    FakeSide.instances.clear()
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(command_rate_hz=100.0), side_factory=FakeSide
    )
    runtime = load_embodiment("openarmv1")
    environment.connect()
    environment.home(runtime.home_q(), list(runtime.joint_names))
    try:
        try:
            environment.streamer.set_targets({"left": np.zeros(3)})  # type: ignore[union-attr]
        except ValueError as exc:
            assert "Invalid OpenArm target" in str(exc)
        else:
            raise AssertionError("invalid target was accepted")
    finally:
        environment.close()


def test_single_side_connects_and_disables_only_selected_arm():
    FakeSide.instances.clear()
    runtime = load_embodiment("openarmv1")
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(
            command_rate_hz=500.0,
            max_joint_speed_rad_s=100.0,
            home_max_joint_speed_rad_s=100.0,
        ),
        side_factory=FakeSide,
        active_sides=("right",),
    )

    environment.connect()
    environment.home(runtime.home_q(), list(runtime.joint_names))
    environment.close()

    assert [side.port for side in FakeSide.instances] == ["can0"]
    assert FakeSide.instances[0].closed


def test_urdf_joint_limit_violation_is_rejected_before_streaming():
    FakeSide.instances.clear()
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(),
        side_factory=FakeSide,
        active_sides=("left",),
        joint_limits={"openarm_left_joint4": (0.0, 2.443461)},
    )
    invalid = runtime.home_q()
    invalid[names.index("openarm_left_joint4")] = -0.1

    with np.testing.assert_raises_regex(ValueError, "outside URDF limits"):
        environment._split_q(invalid, names)
