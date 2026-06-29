import unittest

from handumi.dataset.raw import (
    HANDUMI_RAW_IMAGE_KEYS,
    HANDUMI_RAW_STATE_NAMES,
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
    raw_state_feature,
    validate_raw_state_shape,
)


class RawSchemaTest(unittest.TestCase):
    def test_raw_state_schema_has_expected_layout(self):
        self.assertEqual(HANDUMI_RAW_STATE_SIZE, 16)
        self.assertEqual(len(HANDUMI_RAW_STATE_NAMES), HANDUMI_RAW_STATE_SIZE)
        self.assertEqual(
            HANDUMI_RAW_STATE_NAMES[LEFT_GRIPPER_INDEX],
            "left_gripper_width",
        )
        self.assertEqual(
            HANDUMI_RAW_STATE_NAMES[RIGHT_GRIPPER_INDEX],
            "right_gripper_width",
        )
        self.assertEqual(
            HANDUMI_RAW_STATE_NAMES[LEFT_POSE_SLICE],
            (
                "left_x",
                "left_y",
                "left_z",
                "left_qx",
                "left_qy",
                "left_qz",
                "left_qw",
            ),
        )
        self.assertEqual(
            HANDUMI_RAW_STATE_NAMES[RIGHT_POSE_SLICE],
            (
                "right_x",
                "right_y",
                "right_z",
                "right_qx",
                "right_qy",
                "right_qz",
                "right_qw",
            ),
        )

    def test_raw_state_feature_matches_lerobot_shape(self):
        self.assertEqual(
            raw_state_feature(),
            {
                "dtype": "float32",
                "shape": [16],
                "names": list(HANDUMI_RAW_STATE_NAMES),
            },
        )

    def test_raw_image_keys_are_left_and_right_wrist(self):
        self.assertEqual(
            HANDUMI_RAW_IMAGE_KEYS,
            (
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ),
        )

    def test_validate_raw_state_shape_rejects_wrong_length(self):
        validate_raw_state_shape([0.0] * HANDUMI_RAW_STATE_SIZE)

        with self.assertRaisesRegex(ValueError, "Expected demo length 16, got 15"):
            validate_raw_state_shape([0.0] * 15, name="demo")


if __name__ == "__main__":
    unittest.main()
