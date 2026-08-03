import threading
import time
from types import SimpleNamespace

from handumi.teleop.tracking import (
    LatestTrackingSampler,
    TrackingRecoveryConfig,
    TrackingRecoveryPolicy,
)


def test_recovery_ignores_brief_loss_and_throttles_slow_reconnects():
    policy = TrackingRecoveryPolicy(
        TrackingRecoveryConfig(
            lost_debounce_s=0.3,
            recover_after_s=2.0,
            recover_period_s=5.0,
        )
    )

    assert not policy.note_missing(0.0)
    assert not policy.lost
    assert not policy.note_missing(0.29)
    assert policy.note_missing(0.3)
    assert policy.lost

    assert not policy.should_recover(2.29)
    assert policy.should_recover(2.31)

    # A recovery can itself take seconds.  Its completion must not immediately
    # authorize another SDK/service restart.
    assert not policy.should_recover(6.0)
    assert policy.should_recover(7.31)


def test_recovery_policy_reset_clears_pending_loss():
    policy = TrackingRecoveryPolicy()

    policy.note_missing(10.0)
    policy.reset()

    assert not policy.lost
    assert not policy.note_missing(10.1)


def test_recovery_can_use_known_age_of_a_stale_sample():
    policy = TrackingRecoveryPolicy(TrackingRecoveryConfig(lost_debounce_s=0.15))

    assert policy.note_missing(1.0, observed_since=0.8)


def test_latest_tracking_sampler_does_not_refresh_repeated_device_frame():
    read_twice = threading.Event()
    reads = 0

    def read():
        nonlocal reads
        reads += 1
        if reads >= 2:
            read_twice.set()
        return SimpleNamespace(
            device_time_ns=123,
            aligned_time_ns=time.monotonic_ns(),
            pc_monotonic_ns=time.monotonic_ns(),
        )

    sampler = LatestTrackingSampler(read, sample_rate_hz=200.0)
    sampler.start()
    try:
        assert read_twice.wait(timeout=0.2)
        first = sampler.latest()
        assert first is not None
        first_fresh_at = first.fresh_at_s
        time.sleep(0.015)
        latest = sampler.latest()
        assert latest is not None
    finally:
        sampler.stop()

    assert latest.received_at_s > first.received_at_s
    assert latest.fresh_at_s == first_fresh_at
