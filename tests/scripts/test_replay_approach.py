"""Replay playback starts at the robot home pose (like sim teleop)."""

from __future__ import annotations

import numpy as np

from handumi.scripts.replay.replay_in_sim import joint_approach_ramp

HOME = np.array([0.0, -0.9, 0.6, 0.0], dtype=np.float32)
START = np.array([-0.22, -0.78, 0.18, 0.02], dtype=np.float32)


def test_ramp_starts_at_home_and_stops_before_the_episode() -> None:
    ramp = joint_approach_ramp(HOME, START, frames=30)
    assert ramp.shape == (30, HOME.size)
    np.testing.assert_allclose(ramp[0], HOME, atol=1e-6)
    # The solved trajectory supplies START itself, so the ramp must not repeat it.
    assert float(np.abs(ramp[-1] - START).max()) > 1e-4
    assert float(np.abs(ramp[-1] - START).max()) < float(np.abs(HOME - START).max())


def test_ramp_is_monotonic_and_smooth() -> None:
    ramp = joint_approach_ramp(HOME, START, frames=30)
    steps = np.diff(np.concatenate([ramp, START[None]]), axis=0)
    direction = np.sign(START - HOME)
    assert np.all(np.sign(steps) * direction >= 0)  # never overshoots or backs up
    # Smoothstep easing: no step may exceed the naive linear step by 2x.
    assert float(np.abs(steps).max()) < 2.0 * float(np.abs(START - HOME).max() / 30)


def test_ramp_is_skipped_when_disabled_or_already_home() -> None:
    assert joint_approach_ramp(HOME, START, frames=0).shape == (0, HOME.size)
    assert joint_approach_ramp(HOME, HOME.copy(), frames=30).shape == (0, HOME.size)
