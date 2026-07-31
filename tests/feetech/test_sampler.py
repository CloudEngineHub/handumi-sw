import time

from handumi.feetech import FeetechGripperSampler, GripperWidths


class _Pair:
    def __init__(self):
        self.value = 0

    def read_normalized_widths(self):
        self.value += 1
        return GripperWidths(
            left=self.value / 1000.0,
            right=self.value / 1000.0,
            left_mm=float(self.value),
            right_mm=float(self.value),
            left_normalized=0.1,
            right_normalized=0.1,
            left_ticks=self.value,
            right_ticks=self.value,
        )


class _TransientPair(_Pair):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def read_normalized_widths(self):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("transient serial timeout")
        return super().read_normalized_widths()


class _ReconnectPair(_Pair):
    def __init__(self):
        super().__init__()
        self.fail = True
        self.fast_calls = 0
        self.reconnects = 0

    def read_normalized_widths_fast(self):
        self.fast_calls += 1
        if self.fail:
            raise RuntimeError("wedged adapter")
        return super().read_normalized_widths()

    def reconnect(self):
        self.reconnects += 1
        self.fail = False


def test_sampler_keeps_native_rate_history():
    sampler = FeetechGripperSampler(_Pair(), sample_hz=200.0)
    sampler.start()
    try:
        time.sleep(0.03)
        latest = sampler.latest()
        assert latest is not None
        assert latest.sequence >= 3
        selected = sampler.sample_at(latest.sample_time_ns)
        assert selected == latest
        assert sampler.consecutive_errors == 0
    finally:
        sampler.stop()


def test_sampler_retains_last_sample_across_transient_read_failure():
    pair = _TransientPair()
    sampler = FeetechGripperSampler(pair, sample_hz=50.0)
    sampler.start()
    try:
        first = sampler.latest()
        assert first is not None

        deadline = time.monotonic() + 0.2
        while pair.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

        assert pair.calls >= 2
        assert sampler.latest() == first

        while sampler.consecutive_errors and time.monotonic() < deadline:
            time.sleep(0.005)
        recovered = sampler.latest()
        assert recovered is not None
        assert recovered.sequence > first.sequence
        assert sampler.consecutive_errors == 0
    finally:
        sampler.stop()


def test_sampler_reopens_wedged_adapter_and_recovers_without_blocking_caller():
    pair = _ReconnectPair()
    sampler = FeetechGripperSampler(
        pair,
        sample_hz=100.0,
        reconnect_after_errors=2,
    )

    started = time.monotonic()
    sampler.start(timeout_s=0.5)
    startup_s = time.monotonic() - started
    try:
        latest = sampler.latest()
        assert latest is not None
        assert pair.fast_calls >= 3
        assert pair.reconnects == 1
        assert sampler.reconnect_count == 1
        assert sampler.total_errors == 2
        assert sampler.consecutive_errors == 0
        assert startup_s < 0.2
    finally:
        sampler.stop()
