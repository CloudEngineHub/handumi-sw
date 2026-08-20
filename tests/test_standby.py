from handumi.teleop.standby import GripperHomeStandby


def test_missing_physical_feedback_cancels_close_timer() -> None:
    standby = GripperHomeStandby(hold_s=1.0)

    assert standby.update({"left": 0.0}, 0.0, ("left",)) == ((), ())
    assert standby.update({}, 2.0, ("left",)) == ((), ())
    assert standby.update({"left": 0.0}, 2.1, ("left",)) == ((), ())
    assert standby.update({"left": 0.0}, 3.2, ("left",)) == (("left",), ())


def test_handumi_can_wake_arm_without_robot_feedback() -> None:
    standby = GripperHomeStandby(initial_standby=True)

    assert standby.update(
        {},
        0.0,
        ("left",),
        wake_openings={"left": 1.0},
    ) == ((), ("left",))
