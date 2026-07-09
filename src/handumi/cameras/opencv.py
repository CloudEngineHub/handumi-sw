"""OpenCV/LeRobot camera backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handumi.cameras.base import CameraDevice


@dataclass
class OpenCVCameraDevice(CameraDevice):
    """CameraDevice adapter around LeRobot's OpenCV camera implementation."""

    index_or_path: int | str
    fps: int
    width: int
    height: int

    def __post_init__(self) -> None:
        self._camera = None

    def connect(self) -> None:
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cfg = OpenCVCameraConfig(
            index_or_path=self.index_or_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        self._camera = OpenCVCamera(cfg)
        self._camera.connect()

    def async_read(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("OpenCV camera is not connected.")
        return self._camera.async_read()

    def disconnect(self) -> None:
        if self._camera is None:
            return
        self._camera.disconnect()
        self._camera = None
