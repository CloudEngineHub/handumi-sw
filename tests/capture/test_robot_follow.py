import unittest

import numpy as np

from handumi.capture.robot_follow import WRIST_ALIGN, raw_state_to_robot_targets
from handumi.dataset.raw import HANDUMI_RAW_STATE_SIZE


def _state(
    left_pos=(0.0, 0.0, 0.0),
    right_pos=(0.0, 0.0, 0.0),
    left_quat=(0.0, 0.0, 0.0, 1.0),
    right_quat=(0.0, 0.0, 0.0, 1.0),
    left_width=0.0,
    right_width=0.0,
) -> np.ndarray:
    state = np.zeros(HANDUMI_RAW_STATE_SIZE, dtype=np.float32)
    state[0:3] = left_pos
    state[3:7] = left_quat
    state[7:10] = right_pos
    state[10:14] = right_quat
    state[14] = left_width
    state[15] = right_width
    return state


class RawStateToRobotTargetsTest(unittest.TestCase):
    def test_wrist_align_is_a_rotation(self):
        self.assertAlmostEqual(float(np.linalg.det(WRIST_ALIGN)), 1.0, places=6)
        self.assertTrue(np.allclose(WRIST_ALIGN @ WRIST_ALIGN.T, np.eye(3), atol=1e-6))

    def test_translation_offset_applied(self):
        state = _state(left_pos=(0.3, 0.1, -0.4), right_pos=(0.3, -0.1, -0.4))
        t = raw_state_to_robot_targets(state, translation=np.array([0.1, 0.0, 0.55]))
        self.assertTrue(np.allclose(t["left"][0], [0.4, 0.1, 0.15], atol=1e-6))
        self.assertTrue(np.allclose(t["right"][0], [0.4, -0.1, 0.15], atol=1e-6))

    def test_yaw_rotates_positions_and_orientations(self):
        state = _state(left_pos=(1.0, 0.0, 0.0))
        t = raw_state_to_robot_targets(state, translation=np.zeros(3), yaw_deg=90.0)
        self.assertTrue(np.allclose(t["left"][0], [0.0, 1.0, 0.0], atol=1e-6))
        yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        self.assertTrue(np.allclose(t["left"][1], yaw90 @ WRIST_ALIGN, atol=1e-6))

    def test_identity_orientation_maps_to_wrist_align(self):
        t = raw_state_to_robot_targets(_state(), translation=np.zeros(3))
        self.assertTrue(np.allclose(t["left"][1], WRIST_ALIGN, atol=1e-6))
        self.assertTrue(np.allclose(t["right"][1], WRIST_ALIGN, atol=1e-6))

    def test_orientation_composes_on_the_right(self):
        # 90 deg yaw about workspace Z.
        s = np.sin(np.pi / 4)
        state = _state(left_quat=(0.0, 0.0, s, s))
        yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        t = raw_state_to_robot_targets(state, translation=np.zeros(3))
        self.assertTrue(np.allclose(t["left"][1], yaw90 @ WRIST_ALIGN, atol=1e-5))

    def test_gripper_widths_normalize_and_clip(self):
        state = _state(left_width=0.04, right_width=0.2)
        t = raw_state_to_robot_targets(
            state, translation=np.zeros(3), gripper_max_width_m=0.08
        )
        self.assertAlmostEqual(t["left_grip"], 0.5, places=6)
        self.assertAlmostEqual(t["right_grip"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
