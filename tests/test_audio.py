from __future__ import annotations

import wave
from pathlib import Path

import pytest

from handumi.audio import (
    AudioCaptureError,
    PicoAudioRecorder,
    validate_audio_files,
)


def _packet(sequence: int = 1) -> dict[str, object]:
    return {
        "data": b"\x01\x00" * 160,
        "sample_rate": 16_000,
        "channels": 1,
        "format": "pcm_s16le",
        "capture_timestamp_ns": sequence * 10_000_000,
        "sequence": sequence,
    }


def test_episode_wav_uses_lerobot_chunk_layout(tmp_path: Path) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    recorder.begin_episode()
    recorder._accept_packet(_packet())
    info = recorder.prepare_episode()
    path = recorder.commit_episode(1001, chunks_size=1000)

    assert path.relative_to(tmp_path).as_posix() == (
        "audio/observation.audio/chunk-001/file-001.wav"
    )
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == info["sample_frames"] == 160


def test_cancel_discards_audio_and_empty_episode_is_rejected(tmp_path: Path) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    recorder.begin_episode()
    recorder._accept_packet(_packet())
    recorder.cancel_episode()

    recorder.begin_episode()
    with pytest.raises(AudioCaptureError, match="No PICO audio"):
        recorder.prepare_episode()
    assert not (tmp_path / "audio").exists()


def test_audio_integrity_requires_every_episode(tmp_path: Path) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    recorder.begin_episode()
    recorder._accept_packet(_packet())
    recorder.prepare_episode()
    recorder.commit_episode(0)

    validate_audio_files(tmp_path, 1)
    with pytest.raises(RuntimeError, match="missing episode audio"):
        validate_audio_files(tmp_path, 2)
