from unittest.mock import Mock

import numpy as np

from handumi.scripts.teleop_record import (
    _episode_gesture_action,
    _home_between_episodes,
)
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


def test_teleop_record_right_clap_starts_then_saves() -> None:
    assert _episode_gesture_action("right", recording=False) == "start"
    assert _episode_gesture_action("right", recording=True) == "save"


def test_teleop_record_left_clap_only_discards_active_episode() -> None:
    assert _episode_gesture_action("left", recording=False) is None
    assert _episode_gesture_action("left", recording=True) == "discard"
    assert _episode_gesture_action("both", recording=False) == "finish"
    assert _episode_gesture_action("both", recording=True) == "finish"


def test_teleop_record_homes_synchronously_before_ready() -> None:
    events: list[str] = []
    home_q = np.zeros(14, dtype=np.float32)
    real_env = Mock()
    real_env.home.side_effect = lambda q: events.append("home")
    controller = Mock()
    controller.reset.side_effect = lambda: (events.append("reset"), home_q)[1]
    command_stream = Mock()
    command_stream.stop.side_effect = lambda: events.append("stop")
    command_stream.clear_joint_rate_limits.side_effect = (
        lambda indices: events.append("clear_limits")
    )
    command_stream.submit.side_effect = lambda *args, **kwargs: events.append(
        "submit"
    )
    joint_filter = Mock()
    joint_filter.reset.side_effect = lambda q: events.append("reset_filter")
    home_standby = Mock()
    home_standby.enter_standby.side_effect = lambda sides: events.append("standby")

    _home_between_episodes(
        real_env=real_env,
        controller=controller,
        home_q=home_q,
        enabled_sides=("left", "right"),
        command_stream=command_stream,
        joint_filter=joint_filter,
        home_standby=home_standby,
        grippers=None,
        motion_joint_indices=tuple(range(12)),
        dashboard=None,
        episode=1,
        episode_total="inf",
        play_sounds=False,
    )

    assert events == [
        "stop",
        "home",
        "reset",
        "reset_filter",
        "standby",
        "clear_limits",
        "submit",
    ]
