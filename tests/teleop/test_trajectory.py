import threading
import time

import numpy as np
import pytest

from handumi.teleop.trajectory import (
    DelayedJointCommandBuffer,
    DelayedJointCommandPlayer,
)


def test_delayed_buffer_interpolates_at_playback_time():
    buffer = DelayedJointCommandBuffer(delay_s=0.08)
    buffer.reset(np.array([0.0, 2.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([1.0, 4.0]), {"left": 1.0}, time_s=1.04)

    q, openings = buffer.sample(1.10)

    np.testing.assert_allclose(q, [0.5, 3.0])
    assert openings["left"] == pytest.approx(0.5)


def test_delayed_buffer_holds_latest_command_on_underflow():
    buffer = DelayedJointCommandBuffer(delay_s=0.08)
    buffer.reset(np.array([0.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([1.0]), {"left": 1.0}, time_s=1.04)

    q, openings = buffer.sample(1.20)

    np.testing.assert_allclose(q, [1.0])
    assert openings == {"left": 1.0}


def test_delayed_buffer_bridges_short_underflow_at_recent_velocity():
    buffer = DelayedJointCommandBuffer(
        delay_s=0.02,
        max_extrapolation_s=0.03,
    )
    buffer.reset(np.array([0.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([1.0]), {"left": 1.0}, time_s=1.04)

    q, _ = buffer.sample(1.07)  # playback=1.05: extrapolate 10 ms

    np.testing.assert_allclose(q, [1.25])


def test_delayed_buffer_caps_extrapolation_during_long_tracking_pause():
    buffer = DelayedJointCommandBuffer(
        delay_s=0.0,
        max_extrapolation_s=0.03,
    )
    buffer.reset(np.array([0.0]), {}, time_s=1.00)
    buffer.push(np.array([1.0]), {}, time_s=1.04)

    q, _ = buffer.sample(2.00)

    np.testing.assert_allclose(q, [1.75])


def test_reset_discards_an_old_trajectory_epoch():
    buffer = DelayedJointCommandBuffer(delay_s=0.08)
    buffer.reset(np.array([0.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([10.0]), {"left": 1.0}, time_s=1.04)

    buffer.reset(np.array([3.0]), {"left": 0.25}, time_s=2.00)
    q, openings = buffer.sample(2.50)

    np.testing.assert_allclose(q, [3.0])
    assert openings == {"left": 0.25}


def test_player_publishes_from_its_fixed_rate_thread():
    published: list[float] = []
    received = threading.Event()

    def write(q, openings):
        del openings
        published.append(float(q[0]))
        if len(published) >= 3:
            received.set()

    player = DelayedJointCommandPlayer(
        write,
        command_rate_hz=100.0,
        delay_s=0.0,
    )
    player.start(np.array([2.0]), {"left": 0.5}, time_s=time.perf_counter())
    try:
        assert received.wait(timeout=0.2)
    finally:
        player.stop()

    assert published[:3] == [2.0, 2.0, 2.0]
    latest = player.latest()
    assert latest is not None
    np.testing.assert_allclose(latest[0], [2.0])
    assert latest[1] == {"left": 0.5}


def test_player_applies_ema_at_the_fixed_output_rate():
    published: list[tuple[float, float]] = []
    received_smoothed_target = threading.Event()

    def write(q, openings):
        value = float(q[0])
        published.append((value, openings["left"]))
        if value > 0.0:
            received_smoothed_target.set()

    player = DelayedJointCommandPlayer(
        write,
        command_rate_hz=100.0,
        delay_s=0.0,
        ema_time_constant_s=0.05,
    )
    now = time.perf_counter()
    player.start(np.array([0.0]), {"left": 0.0}, time_s=now)
    try:
        assert len(published) == 1 or _wait_until(lambda: bool(published))
        player.push(np.array([1.0]), {"left": 1.0}, time_s=now + 1e-6)
        assert received_smoothed_target.wait(timeout=0.2)
    finally:
        player.stop()

    first_filtered, first_gripper = next(
        command for command in published if command[0] > 0.0
    )
    expected_alpha = 1.0 - np.exp(-0.01 / 0.05)
    assert first_filtered == pytest.approx(expected_alpha, rel=1e-4)
    assert first_gripper == pytest.approx(1.0)
    latest = player.latest()
    assert latest is not None
    assert 0.0 < latest[0][0] < 1.0
    assert latest[1]["left"] == pytest.approx(1.0)


def test_player_does_not_apply_arm_playback_delay_to_gripper_target():
    published: list[tuple[float, float]] = []
    received_new_gripper = threading.Event()

    def write(q, openings):
        published.append((float(q[0]), openings["left"]))
        if openings["left"] == 1.0:
            received_new_gripper.set()

    player = DelayedJointCommandPlayer(
        write,
        command_rate_hz=100.0,
        delay_s=0.2,
    )
    now = time.perf_counter()
    player.start(np.array([0.0]), {"left": 0.0}, time_s=now)
    try:
        assert _wait_until(lambda: bool(published))
        player.push(np.array([1.0]), {"left": 1.0}, time_s=now + 0.01)
        assert received_new_gripper.wait(timeout=0.1)
    finally:
        player.stop()

    first_new_gripper = next(command for command in published if command[1] == 1.0)
    assert first_new_gripper[0] == pytest.approx(0.0)


def test_player_updates_gripper_without_adding_an_arm_waypoint():
    player = DelayedJointCommandPlayer(
        lambda q, openings: None,
        command_rate_hz=100.0,
        delay_s=0.04,
    )
    now = time.perf_counter()
    player.start(np.array([0.0]), {"left": 0.0}, time_s=now)
    try:
        player.update_openings({"left": 1.0})
        with player._target_openings_lock:
            assert player._target_openings == {"left": 1.0}
        assert len(player.buffer._commands) == 1
    finally:
        player.stop()


def test_player_slew_limits_recovered_gripper_jump_without_filtering_ticks():
    player = DelayedJointCommandPlayer(
        lambda q, openings: None,
        command_rate_hz=100.0,
        delay_s=0.0,
        gripper_max_rate_per_s=10.0,
    )

    _, initial = player._smooth(
        np.array([0.0]), {"left": 0.0}, alpha=1.0, gripper_max_step=0.1
    )
    _, recovered = player._smooth(
        np.array([0.0]), {"left": 1.0}, alpha=1.0, gripper_max_step=0.1
    )

    assert initial["left"] == 0.0
    assert recovered["left"] == pytest.approx(0.1)


def _wait_until(predicate, timeout_s=0.2):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"delay_s": -0.1}, "delay_s"),
        ({"delay_s": 0.1, "max_extrapolation_s": -0.1}, "max_extrapolation_s"),
        ({"delay_s": 0.1, "max_commands": 1}, "max_commands"),
    ),
)
def test_delayed_buffer_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DelayedJointCommandBuffer(**kwargs)


def test_player_rejects_negative_ema_time_constant():
    with pytest.raises(ValueError, match="ema_time_constant_s"):
        DelayedJointCommandPlayer(
            lambda q, openings: None,
            command_rate_hz=100.0,
            delay_s=0.04,
            ema_time_constant_s=-0.1,
        )
