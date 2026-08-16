"""Episode-aligned PICO microphone capture for HandUMI datasets."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("handumi.audio")

AUDIO_KEY = "observation.audio"
AUDIO_PATH = "audio/{audio_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.wav"
PCM_SAMPLE_WIDTH_BYTES = 2


class AudioCaptureError(RuntimeError):
    """The headset audio stream could not produce a valid episode."""


class PicoAudioRecorder:
    """Continuously drain XRoboToolkit audio and stage one PCM WAV per episode.

    Draining while idle prevents the SDK's bounded queue from overflowing and
    ensures a new episode never begins with audio left over from the wait state.
    """

    def __init__(self, xrt_getter: Callable[[], Any], root: Path) -> None:
        self._xrt_getter = xrt_getter
        self._root = Path(root)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recording = False
        self._chunks: list[bytes] = []
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._format: str | None = None
        self._first_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._sequence_gaps = 0
        self._error: BaseException | None = None
        self._prepared_path: Path | None = None
        self._prepared_info: dict[str, object] | None = None

    def start(self) -> None:
        xrt = self._xrt_getter()
        if xrt is None or not callable(getattr(xrt, "get_audio_frame", None)):
            raise AudioCaptureError(
                "The installed xrobotoolkit_sdk has no get_audio_frame API. "
                "Rebuild/install the audio-enabled SDK from external_dependencies."
            )
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-pico-audio",
            daemon=True,
        )
        self._thread.start()
        log.info("PICO audio capture ready; enable Audio in the headset app.")

    def begin_episode(self) -> None:
        with self._lock:
            self._raise_if_failed_locked()
            self._remove_prepared_locked()
            self._reset_episode_locked()
            self._recording = True

    def prepare_episode(self) -> dict[str, object]:
        """Stop the active episode and create a staged WAV without publishing it."""
        with self._lock:
            self._recording = False
            self._raise_if_failed_locked()
            chunks = list(self._chunks)
            sample_rate = self._sample_rate
            channels = self._channels
            audio_format = self._format
            first_timestamp_ns = self._first_timestamp_ns
            last_timestamp_ns = self._last_timestamp_ns
            first_sequence = self._first_sequence
            last_sequence = self._last_sequence
            sequence_gaps = self._sequence_gaps

        if not chunks or sample_rate is None or channels is None:
            raise AudioCaptureError(
                "No PICO audio arrived during the episode. Enable Audio in the headset app."
            )
        staging_dir = self._root / ".audio-staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix="episode-", suffix=".wav", dir=staging_dir)
        os.close(fd)
        staged = Path(raw_path)
        try:
            with wave.open(str(staged), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
                wav.setframerate(sample_rate)
                for chunk in chunks:
                    wav.writeframesraw(chunk)
        except Exception:
            staged.unlink(missing_ok=True)
            raise

        sample_frames = sum(len(chunk) for chunk in chunks) // (
            PCM_SAMPLE_WIDTH_BYTES * channels
        )
        info: dict[str, object] = {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": PCM_SAMPLE_WIDTH_BYTES,
            "format": audio_format or "pcm_s16le",
            "sample_frames": sample_frames,
            "duration_s": sample_frames / sample_rate,
            "first_capture_timestamp_ns": first_timestamp_ns,
            "last_capture_timestamp_ns": last_timestamp_ns,
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "sequence_gaps": sequence_gaps,
        }
        with self._lock:
            self._prepared_path = staged
            self._prepared_info = info
        if sequence_gaps:
            log.warning("PICO audio episode contains %d missing chunk(s).", sequence_gaps)
        return dict(info)

    def commit_episode(self, episode_index: int, *, chunks_size: int = 1000) -> Path:
        with self._lock:
            if self._prepared_path is None:
                raise AudioCaptureError("Audio episode was not prepared before commit.")
            staged = self._prepared_path
            self._prepared_path = None
            self._prepared_info = None
        chunk_index, file_index = divmod(int(episode_index), int(chunks_size))
        relative = Path(
            AUDIO_PATH.format(
                audio_key=AUDIO_KEY,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        )
        destination = self._root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            staged.unlink(missing_ok=True)
            raise AudioCaptureError(f"Refusing to overwrite existing audio: {destination}")
        os.replace(staged, destination)
        self._remove_empty_staging_dir()
        return destination

    def cancel_episode(self) -> None:
        with self._lock:
            self._recording = False
            self._remove_prepared_locked()
            self._reset_episode_locked()
        self._remove_empty_staging_dir()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.cancel_episode()

    def _run(self) -> None:
        while not self._stop.is_set():
            xrt = self._xrt_getter()
            if xrt is None:
                self._stop.wait(0.05)
                continue
            try:
                packet = xrt.get_audio_frame(timeout_ms=100, latest=False)
            except Exception as exc:  # noqa: BLE001 - SDK can fail during recovery.
                if self._stop.is_set():
                    return
                log.warning("PICO audio read failed; retrying: %s", exc)
                time.sleep(0.1)
                continue
            if packet is None:
                continue
            try:
                self._accept_packet(packet)
            except Exception as exc:  # noqa: BLE001 - validate all native payload failures.
                with self._lock:
                    self._error = exc
                    self._recording = False
                log.error("PICO audio capture failed: %s", exc)

    def _accept_packet(self, packet: Any) -> None:
        data = bytes(packet["data"])
        sample_rate = int(packet["sample_rate"])
        channels = int(packet["channels"])
        audio_format = str(packet.get("format") or "pcm_s16le").lower()
        timestamp_ns = int(packet.get("capture_timestamp_ns", 0))
        sequence = int(packet.get("sequence", 0))
        if sample_rate <= 0 or channels <= 0 or not data:
            raise AudioCaptureError("XRoboToolkit returned an invalid audio chunk.")
        if audio_format not in {"pcm16", "pcm_s16le", "s16le"}:
            raise AudioCaptureError(f"Unsupported XRoboToolkit audio format: {audio_format}")
        if len(data) % (PCM_SAMPLE_WIDTH_BYTES * channels):
            raise AudioCaptureError("XRoboToolkit audio chunk is not whole PCM16 frames.")
        with self._lock:
            if not self._recording:
                return
            if self._sample_rate is None:
                self._sample_rate = sample_rate
                self._channels = channels
                self._format = audio_format
                self._first_timestamp_ns = timestamp_ns
                self._first_sequence = sequence
            elif sample_rate != self._sample_rate or channels != self._channels:
                raise AudioCaptureError(
                    "PICO audio format changed within an episode "
                    f"({self._sample_rate} Hz/{self._channels} ch -> "
                    f"{sample_rate} Hz/{channels} ch)."
                )
            if self._last_sequence is not None and sequence > self._last_sequence + 1:
                self._sequence_gaps += sequence - self._last_sequence - 1
            self._last_sequence = sequence
            self._last_timestamp_ns = timestamp_ns
            self._chunks.append(data)

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            error = self._error
            self._error = None
            raise AudioCaptureError(str(error)) from error

    def _reset_episode_locked(self) -> None:
        self._chunks = []
        self._sample_rate = None
        self._channels = None
        self._format = None
        self._first_timestamp_ns = None
        self._last_timestamp_ns = None
        self._first_sequence = None
        self._last_sequence = None
        self._sequence_gaps = 0

    def _remove_prepared_locked(self) -> None:
        if self._prepared_path is not None:
            self._prepared_path.unlink(missing_ok=True)
        self._prepared_path = None
        self._prepared_info = None

    def _remove_empty_staging_dir(self) -> None:
        staging_dir = self._root / ".audio-staging"
        try:
            staging_dir.rmdir()
        except OSError:
            pass


def audio_metadata(enabled: bool) -> dict[str, object]:
    return {
        "enabled": bool(enabled),
        "key": AUDIO_KEY,
        "path": AUDIO_PATH,
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_width_bytes": PCM_SAMPLE_WIDTH_BYTES,
        "episode_aligned": True,
    }


def validate_audio_files(root: Path, total_episodes: int, *, chunks_size: int = 1000) -> None:
    """Require one readable non-empty WAV for every committed episode."""
    for episode_index in range(total_episodes):
        chunk_index, file_index = divmod(episode_index, chunks_size)
        path = Path(root) / AUDIO_PATH.format(
            audio_key=AUDIO_KEY,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        if not path.is_file():
            raise RuntimeError(f"Dataset is missing episode audio: {path}.")
        try:
            with wave.open(str(path), "rb") as wav:
                if wav.getnframes() <= 0 or wav.getframerate() <= 0 or wav.getnchannels() <= 0:
                    raise RuntimeError(f"Dataset has empty or invalid episode audio: {path}.")
        except (OSError, EOFError, wave.Error) as exc:
            raise RuntimeError(f"Dataset has invalid episode audio: {path}.") from exc
