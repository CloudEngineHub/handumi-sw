import threading

import pytest

from handumi.teleop.common import BestEffortPeriodicWorker


def test_best_effort_worker_runs_peripheral_callback_off_the_caller_thread():
    called = threading.Event()
    callback_thread: list[int] = []

    def callback():
        callback_thread.append(threading.get_ident())
        called.set()

    caller_thread = threading.get_ident()
    worker = BestEffortPeriodicWorker(
        callback,
        rate_hz=100.0,
        thread_name="test-peripheral-worker",
    )
    worker.start()
    try:
        assert called.wait(timeout=0.2)
    finally:
        worker.close()

    assert callback_thread[0] != caller_thread
    assert worker.error is None


def test_best_effort_worker_rejects_nonpositive_rate():
    with pytest.raises(ValueError, match="rate_hz"):
        BestEffortPeriodicWorker(lambda: None, rate_hz=0.0, thread_name="test")
