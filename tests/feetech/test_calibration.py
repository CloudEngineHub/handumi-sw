import tempfile
import unittest
from pathlib import Path

from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    load_config,
    save_config,
)


class FeetechCalibrationTest(unittest.TestCase):
    def test_normalized_width(self):
        calibration = GripperCalibration(
            servo_id=1,
            closed_ticks=1000,
            open_ticks=2000,
        )
        self.assertEqual(calibration.normalized_width(900), 0.0)
        self.assertEqual(calibration.normalized_width(1000), 0.0)
        self.assertEqual(calibration.normalized_width(1500), 0.5)
        self.assertEqual(calibration.normalized_width(2000), 1.0)
        self.assertEqual(calibration.normalized_width(2100), 1.0)

    def test_inverted_ticks_are_supported(self):
        calibration = GripperCalibration(
            servo_id=2,
            closed_ticks=2000,
            open_ticks=1000,
        )
        self.assertEqual(calibration.normalized_width(2000), 0.0)
        self.assertEqual(calibration.normalized_width(1500), 0.5)
        self.assertEqual(calibration.normalized_width(1000), 1.0)

    def test_round_trip_config(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyUSB0"),
            right=GripperCalibration(1, 900, 1900, 75.0, "/dev/ttyUSB1"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feetech.yaml"
            save_config(config, path)
            loaded = load_config(path)
        self.assertEqual(loaded, config)

    def test_width_units(self):
        calibration = GripperCalibration(
            servo_id=0,
            closed_ticks=1000,
            open_ticks=2000,
            max_width_mm=80.0,
        )
        self.assertEqual(calibration.width_mm(1500), 40.0)
        self.assertEqual(calibration.width_m(1500), 0.04)


if __name__ == "__main__":
    unittest.main()
