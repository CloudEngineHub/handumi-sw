from __future__ import annotations

import wave
from pathlib import Path

import pytest

from handumi.audio import (
    AudioCaptureError,
    PicoAudioRecorder,
    audio_features,
    audio_metadata,
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


def test_missing_pico_audio_can_commit_aligned_silence_for_robot_episode(
    tmp_path: Path,
) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    recorder.begin_episode(1_000_000_000)
    for target_ns in (1_010_000_000, 1_020_000_000):
        frame, healthy = recorder.synchronized_frame(
            target_ns,
            target_ns,
            stale_timeout_s=0.25,
            max_sync_skew_s=0.01,
        )
        assert not healthy
        assert frame["observation.audio.healthy"].item() == 0

    info = recorder.prepare_episode(expected_frames=2)
    path = recorder.commit_episode(0)

    assert info["synthetic_silence"] is True
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0
        assert wav.readframes(wav.getnframes()) == bytes(wav.getnframes() * 2)


def test_audio_integrity_requires_every_episode(tmp_path: Path) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    recorder.begin_episode()
    recorder._accept_packet(_packet())
    recorder.prepare_episode()
    recorder.commit_episode(0)

    validate_audio_files(tmp_path, 1)
    with pytest.raises(RuntimeError, match="missing episode audio"):
        validate_audio_files(tmp_path, 2)


def test_audio_is_aligned_to_frame_targets_and_trimmed_to_episode_clock(
    tmp_path: Path,
) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    recorder.begin_episode(1_000_000_000)
    recorder._accept_packet(_packet(10), receive_time_ns=1_005_000_000)
    recorder._accept_packet(_packet(11), receive_time_ns=1_015_000_000)

    first, first_healthy = recorder.synchronized_frame(
        1_007_000_000,
        1_020_000_000,
        stale_timeout_s=0.25,
        max_sync_skew_s=0.01,
    )
    second, second_healthy = recorder.synchronized_frame(
        1_017_000_000,
        1_020_000_000,
        stale_timeout_s=0.25,
        max_sync_skew_s=0.01,
    )
    # teleop_record keeps one look-ahead row for action alignment; its final
    # pending target must not extend the committed episode audio.
    recorder.synchronized_frame(
        1_027_000_000,
        1_030_000_000,
        stale_timeout_s=0.25,
        max_sync_skew_s=0.01,
    )
    info = recorder.prepare_episode(expected_frames=2)
    path = recorder.commit_episode(0)

    assert first_healthy and second_healthy
    assert first["observation.audio.sample_index"].item() == 112
    assert second["observation.audio.sample_index"].item() == 272
    assert first["observation.audio.aligned_time_ns"].item() == 1_007_000_000
    assert second["observation.audio.sequence"].item() == 11
    assert info["episode_start_time_ns"] == 1_000_000_000
    assert info["first_frame_target_time_ns"] == 1_007_000_000
    assert info["last_frame_target_time_ns"] == 1_017_000_000
    with wave.open(str(path), "rb") as wav:
        assert wav.getnframes() == 432
        pcm = wav.readframes(wav.getnframes())
    assert pcm[: 80 * 2] == bytes(80 * 2)
    assert pcm[80 * 2 : 400 * 2] == b"\x01\x00" * 320


def test_audio_alignment_schema_is_explicit() -> None:
    assert set(audio_features()) == {
        "observation.audio.device_time_ns",
        "observation.audio.pc_monotonic_ns",
        "observation.audio.aligned_time_ns",
        "observation.audio.sample_index",
        "observation.audio.sample_rate",
        "observation.audio.sequence",
        "observation.audio.healthy",
    }
    metadata = audio_metadata(True)
    assert metadata["episode_aligned"] is True
    assert metadata["frame_aligned"] is True
    assert metadata["timestamp_clock"] == "pc_monotonic"
