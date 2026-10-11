from __future__ import annotations

from subprocess import CompletedProcess

import numpy as np
import pytest

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.tracking import pico

from handumi.tracking.pico import (
    PICO_AUDIO_SERVICE_PORT,
    PICO_SERVICE_PORT,
    setup_adb_reverse,
)


def test_setup_adb_reverse_configures_tracking_and_audio_ports() -> None:
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        return CompletedProcess(command, 0, stdout="", stderr="")

    assert setup_adb_reverse(runner=runner)
    assert commands == [
        [
            "adb",
            "reverse",
            f"tcp:{PICO_SERVICE_PORT}",
            f"tcp:{PICO_SERVICE_PORT}",
        ],
        [
            "adb",
            "reverse",
            f"tcp:{PICO_AUDIO_SERVICE_PORT}",
            f"tcp:{PICO_AUDIO_SERVICE_PORT}",
        ],
    ]


def test_setup_adb_reverse_attempts_both_ports_when_one_fails() -> None:
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        port = command[2]
        return CompletedProcess(
            command,
            1 if port == f"tcp:{PICO_SERVICE_PORT}" else 0,
            stdout="",
            stderr="failed",
        )

    assert not setup_adb_reverse(runner=runner)
    assert [command[2] for command in commands] == [
        f"tcp:{PICO_SERVICE_PORT}",
        f"tcp:{PICO_AUDIO_SERVICE_PORT}",
    ]


@pytest.fixture
def provider(monkeypatch):
    pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    tracker = pico.PicoTrackingProvider(calibration=ControllerTcpCalibration(
        left=pose.copy(), right=pose.copy(), source=None,
    ))
    tracker.xrt = object()
    frame = {
        "observation.pico.timestamp_ns": [100],
        "observation.pico.left_controller_pose": [1, 2, 3, 0, 0, 0, 1],
        "observation.pico.right_controller_pose": [1, 2, 3, 0, 0, 0, 1],
        "observation.pico.headset_pose": pose,
    }
    monkeypatch.setattr(pico, "read_pico_frame", lambda *a, **k: frame)
    return tracker, frame


def test_frozen_timestamp_raises_and_cached_sample_keeps_original_time(provider, monkeypatch):
    tracker, _ = provider
    times = iter([10_000_000_000, 10_500_000_000, 11_000_000_000])
    monkeypatch.setattr(pico.time, "monotonic_ns", lambda: next(times))
    first = tracker.latest()
    cached = tracker.latest()
    assert cached.pc_monotonic_ns == first.pc_monotonic_ns
    with pytest.raises(pico.PicoTrackingFrozenError, match="timestamp has not advanced"):
        tracker.latest()


def test_stationary_controller_with_new_timestamps_is_valid(provider, monkeypatch):
    tracker, frame = provider
    times = iter([10_000_000_000, 12_000_000_000, 14_000_000_000])
    monkeypatch.setattr(pico.time, "monotonic_ns", lambda: next(times))
    for timestamp in (100, 200, 300):
        frame["observation.pico.timestamp_ns"] = [timestamp]
        assert tracker.latest().streaming
