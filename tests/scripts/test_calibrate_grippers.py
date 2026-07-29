import io
import unittest
from unittest import mock

from handumi.scripts.setup import calibrate_grippers


class _FakeBus:
    def __init__(self, reads):
        self._reads = list(reads)

    def read_position(self, _servo_id):
        value = self._reads.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class CalibrateGrippersCaptureTest(unittest.TestCase):
    def test_capture_uses_last_good_tick_after_transient_read_failure(self):
        bus = _FakeBus([1234, RuntimeError("temporary serial failure")])
        stdin = io.StringIO("\n")

        with (
            mock.patch.object(calibrate_grippers.sys, "stdin", stdin),
            mock.patch.object(calibrate_grippers.sys, "stdout", io.StringIO()),
            mock.patch.object(
                calibrate_grippers.select,
                "select",
                side_effect=[([], [], []), ([stdin], [], [])],
            ),
        ):
            ticks = calibrate_grippers._capture_live_ticks(
                bus, servo_id=6, prompt="Capture", interval_s=0.0
            )

        self.assertEqual(ticks, 1234)

    def test_capture_requires_at_least_one_good_tick(self):
        bus = _FakeBus([RuntimeError("no serial data")])
        stdin = io.StringIO("\n")

        with (
            mock.patch.object(calibrate_grippers.sys, "stdin", stdin),
            mock.patch.object(calibrate_grippers.sys, "stdout", io.StringIO()),
            mock.patch.object(
                calibrate_grippers.select,
                "select",
                return_value=([stdin], [], []),
            ),
            self.assertRaisesRegex(RuntimeError, "no valid encoder value"),
        ):
            calibrate_grippers._capture_live_ticks(
                bus, servo_id=6, prompt="Capture", interval_s=0.0
            )


if __name__ == "__main__":
    unittest.main()
