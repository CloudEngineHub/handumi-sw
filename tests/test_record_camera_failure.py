from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from handumi.scripts.record import CameraCaptureError, record_episode
from handumi.tracking.base import ControllerPairSample


class _Dataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def clear_episode_buffer(self) -> None:
        self.frames.clear()


class _FailedCamera:
    output_width = 640
    output_height = 480

    def sample_at(self, target_time_ns: int):
        raise RuntimeError("USB device disappeared")


class _HealthyTracker:
    def latest(self) -> ControllerPairSample:
        now_ns = time.monotonic_ns()
        return replace(
            ControllerPairSample.empty("test"),
            left_tracked=True,
            right_tracked=True,
            pc_monotonic_ns=now_ns,
            aligned_time_ns=now_ns,
        )


def test_record_episode_treats_camera_loss_as_fatal() -> None:
    dataset = _Dataset()

    with pytest.raises(CameraCaptureError, match=r"left_wrist.*disconnected.*device ID"):
        record_episode(
            dataset=dataset,
            cameras=[_FailedCamera()],
            cam_names=["left_wrist"],
            tracker=_HealthyTracker(),
            grippers=None,
            episode_time_s=None,
            fps=30,
            task="test",
            cam_width=640,
            cam_height=480,
            stop_event=threading.Event(),
            manual_control=False,
            start_button="enter",
            repeat_button="a",
            finish_button="b",
            start_threshold=0.5,
            sensor_loss_timeout_s=0.0,
        )

    assert dataset.frames == []
