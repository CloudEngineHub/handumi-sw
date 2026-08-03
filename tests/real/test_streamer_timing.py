from handumi.real.streamer import next_periodic_deadline


def test_periodic_deadline_keeps_regular_cadence_when_on_time():
    assert next_periodic_deadline(1.00, 0.01, 1.005) == 1.01


def test_periodic_deadline_skips_backlog_after_scheduler_stall():
    # Five missed deadlines are discarded, not replayed as a command burst.
    assert next_periodic_deadline(1.00, 0.01, 1.055) == 1.065
