import unittest
import tempfile
from pathlib import Path

import yaml

from handumi.cameras.usb import build_camera_specs, resolve_camera_ids


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

    def test_resolve_camera_ids_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.yaml"
            with path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    {
                        "left_wrist": {"index_or_path": 3},
                        "right_wrist": {"index_or_path": 5},
                    },
                    fh,
                )

            self.assertEqual(resolve_camera_ids(None, path), [3, 5])

    def test_explicit_camera_ids_override_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.yaml"
            self.assertEqual(resolve_camera_ids([7, 8], path), [7, 8])


if __name__ == "__main__":
    unittest.main()
