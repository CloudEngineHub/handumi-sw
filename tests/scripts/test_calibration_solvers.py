import unittest

import numpy as np

from handumi.scripts.setup.calibrate_tcp_offset import solve_pivot
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
        self.assertTrue(np.allclose(t, t_true, atol=1e-6))
        self.assertTrue(np.allclose(c, c_true, atol=1e-6))
        self.assertLess(rms, 1e-6)

    def test_noise_reflected_in_rms(self):
        rng = np.random.default_rng(8)
        t_true = np.array([0.14, 0.0, 0.0])
        R = _rand_rotations(rng, 300)
        P = -np.einsum("nij,j->ni", R, t_true) + rng.normal(0, 0.002, (300, 3))
        t, _, rms = solve_pivot(P, R)
        self.assertTrue(np.allclose(t, t_true, atol=0.005))
        self.assertGreater(rms, 0.0005)
        self.assertLess(rms, 0.01)


if __name__ == "__main__":
    unittest.main()
