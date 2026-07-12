import unittest

from handumi.scripts.setup.setup_hardware import parse_args


class SetupHardwareArgsTest(unittest.TestCase):
    def test_defaults_to_piper_pico(self):
        args = parse_args([])

        self.assertEqual(args.robot, "piper")
        self.assertEqual(args.device, "pico")
        self.assertEqual(args.bitrate, 1_000_000)
        self.assertEqual(args.restart_ms, 100)
        self.assertFalse(args.skip_can_map)
        self.assertFalse(args.skip_feetech_map)
        self.assertEqual(args.feetech_start_id, 0)
        self.assertEqual(args.feetech_end_id, 20)

    def test_can_skip_flags_are_available_for_repair_only_runs(self):
        args = parse_args(["--skip-can-map", "--skip-feetech-map", "--skip-pico"])

        self.assertTrue(args.skip_can_map)
        self.assertTrue(args.skip_feetech_map)
        self.assertTrue(args.skip_pico)


if __name__ == "__main__":
    unittest.main()
