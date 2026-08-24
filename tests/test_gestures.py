from handumi.tracking.gestures import DoubleClapDetector


def test_reset_requires_reopen_before_counting_a_new_clap() -> None:
    detector = DoubleClapDetector()

    # Complete a right-hand double clap, ending with the gripper still closed.
    assert detector.update_side(30.0, 30.0, 0.0) is None
    assert detector.update_side(30.0, 5.0, 0.1) is None
    assert detector.update_side(30.0, 30.0, 0.2) is None
    assert detector.update_side(30.0, 5.0, 0.3) == "right"

    detector.reset()

    # The closure held across the episode boundary must not become clap one.
    assert detector.update_side(30.0, 5.0, 0.4) is None
    assert detector.update_side(30.0, 30.0, 0.5) is None
    assert detector.update_side(30.0, 5.0, 0.6) is None
    assert detector.update_side(30.0, 30.0, 0.7) is None
    assert detector.update_side(30.0, 5.0, 0.8) == "right"
