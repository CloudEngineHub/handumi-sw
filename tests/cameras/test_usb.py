import unittest

from handumi.cameras.usb import build_camera_specs


class UsbCameraConfigTest(unittest.TestCase):
    def test_build_camera_specs_without_laptop_camera(self):
        specs, laptop_name = build_camera_specs(
            [0, 2],
            laptop_camera=False,
            laptop_cam_id=4,
            laptop_cam_name="laptop",
        )

        self.assertIsNone(laptop_name)
        self.assertEqual(
            specs,
            [
                {"id": 0, "name": "left_wrist", "is_laptop": False},
                {"id": 2, "name": "right_wrist", "is_laptop": False},
            ],
        )

    def test_build_camera_specs_reuses_named_camera_for_laptop(self):
        specs, laptop_name = build_camera_specs(
            [0, 2],
            laptop_camera=True,
            laptop_cam_id=9,
            laptop_cam_name="right_wrist",
        )

        self.assertEqual(laptop_name, "right_wrist")
        self.assertEqual(
            specs,
            [
                {"id": 0, "name": "left_wrist", "is_laptop": False},
                {"id": 9, "name": "right_wrist", "is_laptop": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()
