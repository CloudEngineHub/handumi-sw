from handumi.teleop.tracking import TrackingRecoveryConfig, TrackingRecoveryPolicy


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
