from __future__ import annotations

import sys
from types import SimpleNamespace

from handumi.real.piper import driver


def test_sdk_arm_resets_slave_state_before_enabling(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakePiperInterface:
        def __init__(self, port: str) -> None:
            calls.append(("init", port))

        def ConnectPort(self) -> None:
            calls.append(("connect",))

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=2, arm_status=0, motion_status=0
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=1,
                    joint_2=2,
                    joint_3=3,
                    joint_4=4,
                    joint_5=5,
                    joint_6=6,
                )
            )

        def ResetPiper(self) -> None:
            calls.append(("reset",))

        def MotionCtrl_2(self, *args: int) -> None:
            calls.append(("mode", *args))

        def EnablePiper(self) -> bool:
            calls.append(("enable",))
            return True

        def JointCtrl(self, *args: int) -> None:
            calls.append(("joint", *args))

    monkeypatch.setitem(
        sys.modules,
        "piper_sdk",
        SimpleNamespace(C_PiperInterface_V2=FakePiperInterface),
    )
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)

    driver.PiperSdkArm("can0", 80, 10, 1.0, 1000)

    assert calls == [
        ("init", "can0"),
        ("connect",),
        ("reset",),
        ("mode", 0x01, 0x01, 10, 0x00),
        ("enable",),
        ("joint", 1, 2, 3, 4, 5, 6),
        ("mode", 0x01, 0x01, 80, 0x00),
        ("joint", 1, 2, 3, 4, 5, 6),
    ]


def test_sdk_arm_reconnect_holds_feedback_without_reset(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakePiperInterface:
        def __init__(self, _port: str) -> None:
            pass

        def ConnectPort(self) -> None:
            pass

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=10,
                    joint_2=20,
                    joint_3=30,
                    joint_4=40,
                    joint_5=25_000,
                    joint_6=60,
                )
            )

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1, arm_status=0, motion_status=0
                )
            )

        def ResetPiper(self) -> None:
            calls.append(("reset",))

        def MotionCtrl_2(self, *args: int) -> None:
            calls.append(("mode", *args))

        def EnablePiper(self) -> bool:
            return True

        def JointCtrl(self, *args: int) -> None:
            calls.append(("joint", *args))

    monkeypatch.setitem(
        sys.modules,
        "piper_sdk",
        SimpleNamespace(C_PiperInterface_V2=FakePiperInterface),
    )
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)

    driver.PiperSdkArm("can0", 80, 10, 1.0, 1000)

    assert ("reset",) not in calls
    assert calls == [
        ("mode", 0x01, 0x01, 10, 0x00),
        ("joint", 10, 20, 30, 40, 25_000, 60),
        ("mode", 0x01, 0x01, 80, 0x00),
        ("joint", 10, 20, 30, 40, 25_000, 60),
    ]


def test_sdk_arm_rejects_stale_gripper_feedback(monkeypatch) -> None:
    now_s = 10.0

    class FakePiperInterface:
        def __init__(self, _port: str) -> None:
            pass

        def ConnectPort(self) -> None:
            pass

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1, arm_status=0, motion_status=0
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=0,
                    joint_2=0,
                    joint_3=0,
                    joint_4=0,
                    joint_5=0,
                    joint_6=0,
                )
            )

        def ResetPiper(self) -> None:
            pass

        def MotionCtrl_2(self, *_args: int) -> None:
            pass

        def EnablePiper(self) -> bool:
            return True

        def JointCtrl(self, *_args: int) -> None:
            pass

        def GetArmGripperMsgs(self):
            return SimpleNamespace(
                time_stamp=123.0,
                gripper_state=SimpleNamespace(grippers_angle=12_000),
            )

    monkeypatch.setitem(
        sys.modules,
        "piper_sdk",
        SimpleNamespace(C_PiperInterface_V2=FakePiperInterface),
    )
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(driver.time, "monotonic", lambda: now_s)
    arm = driver.PiperSdkArm("can0", 80, 10, 1.0, 1000)

    assert arm.read_gripper_microm() == 12_000
    now_s += driver.GRIPPER_FEEDBACK_STALE_S + 0.01
    assert arm.read_gripper_microm() is None


def test_environment_normalizes_measured_gripper_opening() -> None:
    class FakeArm:
        port = "can0"

        def read_gripper_microm(self) -> int:
            return 12_000

    environment = driver.PiperCanEnvironment(
        driver.PiperCanSettings(left_port="can0", right_port="can1")
    )
    environment.arms = {"left": FakeArm()}  # type: ignore[assignment]
    environment.gripper_ranges = {
        "left": driver.PiperGripperRange(2_000, 52_000, "test")
    }

    openings = environment.gripper_openings(fallback_max_width_mm=66.0)

    assert openings == {"left": 0.2}


def test_environment_exposes_detailed_gripper_feedback() -> None:
    class FakeArm:
        port = "can0"

        def read_gripper_microm(self) -> int:
            return 12_000

    environment = driver.PiperCanEnvironment(
        driver.PiperCanSettings(left_port="can0", right_port="can1")
    )
    environment.arms = {"left": FakeArm()}  # type: ignore[assignment]
    gripper_range = driver.PiperGripperRange(2_000, 52_000, "test")
    environment.gripper_ranges = {"left": gripper_range}

    feedback = environment.gripper_feedback(fallback_max_width_mm=66.0)

    assert feedback["left"] == driver.PiperGripperFeedback(
        measured_microm=12_000,
        opening=0.2,
        gripper_range=gripper_range,
    )
    assert feedback["left"].measured_mm == 12.0
