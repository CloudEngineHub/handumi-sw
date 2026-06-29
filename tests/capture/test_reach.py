import unittest

import numpy as np

from handumi.capture.reach import compute_reach_features, empty_reach_features


class ReachFeatureTest(unittest.TestCase):
    def test_empty_reach_features_contains_expected_keys(self):
        features = empty_reach_features(feasible=True)

        self.assertEqual(features["observation.reach.any_episode_feasible"][0], 1)
        self.assertIn("observation.reach.piper_left_ratio", features)
        self.assertIn("observation.reach.openarm_episode_feasible", features)

    def test_compute_reach_features_uses_controller_displacement(self):
        pico_frame = {
            "observation.pico.left_controller_pose": np.array(
                [0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
            "observation.pico.right_controller_pose": np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        }

        features, metrics = compute_reach_features(
            pico_frame,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

        self.assertAlmostEqual(metrics["piper"]["left_ratio"], 1.0)
        self.assertTrue(metrics["piper"]["feasible"])
        self.assertEqual(features["observation.reach.piper_frame_feasible"][0], 1)


if __name__ == "__main__":
    unittest.main()

