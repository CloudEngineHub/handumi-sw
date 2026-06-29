import unittest

import numpy as np

from handumi.retargeting.handumi_to_robot import (
    quaternion_xyzw_to_matrix,
    raw_state_target_poses,
    split_raw_state,
)


class HandumiToRobotTest(unittest.TestCase):
    def test_quaternion_identity(self):
        rot = quaternion_xyzw_to_matrix(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-6)

    def test_split_raw_state(self):
        state = np.zeros(16, dtype=np.float32)
        state[0:7] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        state[7:14] = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
        state[14] = 0.25
        state[15] = 0.75

        raw = split_raw_state(state)

        np.testing.assert_allclose(raw.left_position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(raw.right_position, [4.0, 5.0, 6.0])
        self.assertAlmostEqual(raw.left_gripper_width, 0.25)
        self.assertAlmostEqual(raw.right_gripper_width, 0.75)

    def test_raw_state_target_poses(self):
        state = np.zeros(16, dtype=np.float32)
        state[3:7] = [0.0, 0.0, 0.0, 1.0]
        state[10:14] = [0.0, 0.0, 0.0, 1.0]

        left_pose, right_pose = raw_state_target_poses(state)

        np.testing.assert_allclose(left_pose[1], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(right_pose[1], np.eye(3), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

