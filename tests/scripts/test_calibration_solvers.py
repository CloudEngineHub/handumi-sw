import unittest

import numpy as np

from handumi.scripts.setup.calibrate_tcp_offset import solve_pivot
from handumi.scripts.setup.calibrate_workspace import solve_transform
from handumi.tracking.transforms import quat_to_matrix


def _rand_rotations(rng, n):
    quats = rng.standard_normal((n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return np.asarray([quat_to_matrix(q) for q in quats])


class SolvePivotTest(unittest.TestCase):
    def test_recovers_known_offset(self):
        rng = np.random.default_rng(7)
        t_true = np.array([0.14, -0.02, 0.05])
        c_true = np.array([0.3, 0.1, -0.4])
        R = _rand_rotations(rng, 200)
        # p_i = c - R_i @ t  (tip pinned at c)
        P = c_true - np.einsum("nij,j->ni", R, t_true)
        t, c, rms = solve_pivot(P, R)
        self.assertTrue(np.allclose(t, t_true, atol=1e-8))
        self.assertTrue(np.allclose(c, c_true, atol=1e-8))
        self.assertLess(rms, 1e-9)

    def test_noise_reflected_in_rms(self):
        rng = np.random.default_rng(8)
        t_true = np.array([0.14, 0.0, 0.0])
        R = _rand_rotations(rng, 300)
        P = -np.einsum("nij,j->ni", R, t_true) + rng.normal(0, 0.002, (300, 3))
        t, _, rms = solve_pivot(P, R)
        self.assertTrue(np.allclose(t, t_true, atol=0.005))
        self.assertGreater(rms, 0.0005)
        self.assertLess(rms, 0.01)


class SolveTransformTest(unittest.TestCase):
    def test_single_point_translation_only(self):
        w = np.array([[0.1, 0.2, -0.3]])
        b = np.array([[0.45, 0.2, 0.25]])
        t, yaw, rms = solve_transform(w, b, solve_yaw=False)
        self.assertTrue(np.allclose(t, [0.35, 0.0, 0.55], atol=1e-9))
        self.assertEqual(yaw, 0.0)
        self.assertLess(rms, 1e-9)

    def test_two_points_recover_yaw_and_translation(self):
        yaw_true = np.deg2rad(30.0)
        c, s = np.cos(yaw_true), np.sin(yaw_true)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        t_true = np.array([0.3, -0.1, 0.55])
        W = np.array([[0.2, 0.0, -0.3], [-0.1, 0.25, -0.2], [0.05, -0.15, -0.4]])
        B = W @ R.T + t_true
        t, yaw_deg, rms = solve_transform(W, B, solve_yaw=True)
        self.assertAlmostEqual(yaw_deg, 30.0, places=6)
        self.assertTrue(np.allclose(t, t_true, atol=1e-9))
        self.assertLess(rms, 1e-9)


if __name__ == "__main__":
    unittest.main()
