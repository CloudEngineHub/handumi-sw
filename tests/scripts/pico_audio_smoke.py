#!/usr/bin/env python3
"""Manual PICO microphone smoke test using the teleop-record audio path.

This intentionally opens no robot, cameras, Feetech devices, or dataset. It
starts the same PICO XRoboToolkit provider used by ``handumi teleop-record``,
drains its audio with ``PicoAudioRecorder``, and saves one real WAV when audio
packets were received.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from handumi.audio import AudioCaptureError, PicoAudioRecorder
from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.tracking.pico import PicoTrackingProvider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read only PICO audio and save one diagnostic WAV."
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pico-audio-smoke"),
        help="Parent directory for the timestamped diagnostic run.",
    )
    parser.add_argument(
        "--pico-wifi",
        action="store_true",
        help="Use the same optional WiFi transport as teleop-record.",
    )
    parser.add_argument(
        "--skip-adb-check",
        action="store_true",
        help="Skip ADB device validation while retaining ADB transport.",
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be > 0")
    return args


def _identity_calibration() -> ControllerTcpCalibration:
    # Audio does not use controller poses, but PicoTrackingProvider requires
    # calibration because it is the same provider instantiated by teleop-record.
    identity = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    return ControllerTcpCalibration(left=identity.copy(), right=identity.copy())


def _audio_counts(recorder: PicoAudioRecorder) -> tuple[int, int]:
    # This is a manual diagnostic alongside the implementation, so inspecting
    # the recorder under its own lock is preferable to adding debug API to src/.
    with recorder._lock:
        chunks = list(recorder._chunks)
    return len(chunks), sum(len(chunk.data) for chunk in chunks)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    tracker = PicoTrackingProvider(
        calibration=_identity_calibration(),
        mode="mandos",
        transport="wifi" if args.pico_wifi else "adb",
        skip_adb_check=args.skip_adb_check,
    )
    recorder: PicoAudioRecorder | None = None

    print("Connecting to PICO with the teleop-record XRoboToolkit path ...", flush=True)
    try:
        tracker.start()
        recorder = PicoAudioRecorder(lambda: tracker.xrt, run_dir)
        recorder.start()
        recorder.begin_episode(time.monotonic_ns())
        print(
            f"Recording PICO audio for {args.duration_s:.1f}s. "
            "Enable Audio in the headset app.",
            flush=True,
        )

        deadline = time.monotonic() + args.duration_s
        previous_chunks = -1
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            time.sleep(min(1.0, remaining))
            chunks, byte_count = _audio_counts(recorder)
            if chunks > previous_chunks:
                print(
                    f"AUDIO RECEIVED: {chunks} chunk(s), {byte_count} PCM bytes",
                    flush=True,
                )
                previous_chunks = chunks

        chunks, byte_count = _audio_counts(recorder)
        if chunks == 0:
            print(
                "NO AUDIO RECEIVED: verify that Audio is enabled in the PICO app.",
                file=sys.stderr,
                flush=True,
            )
            recorder.cancel_episode()
            return 2

        info = recorder.prepare_episode()
        wav_path = recorder.commit_episode(0)
        print(
            "AUDIO OK: "
            f"{chunks} chunk(s), {byte_count} bytes, "
            f"{float(info['duration_s']):.2f}s saved to {wav_path}",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no diagnostic WAV was committed.", file=sys.stderr)
        return 130
    except AudioCaptureError as exc:
        print(f"AUDIO ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if recorder is not None:
            recorder.close()
        tracker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
