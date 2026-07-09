"""Camera overlay and laptop preview helpers."""

from __future__ import annotations

import logging
import shutil
import subprocess

import cv2
import numpy as np

log = logging.getLogger("handumi.record")
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (235, 235, 235),
    thick: int = 1,
) -> None:
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick)


def _panel(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], alpha: float = 0.55) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:04.1f}"


def draw_laptop_overlay(
    frame: np.ndarray,
    *,
    elapsed_s: float,
    n_frames: int,
    tracker_count: int,
    manual_control: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    _panel(out, (0, 0), (w, 48), 0.52)
    _text(out, "REC", (12, 30), 0.62, (235, 80, 80), 2)
    _text(out, _clock(elapsed_s), (w // 2 - 42, 31), 0.72, (70, 220, 245), 2)
    _text(out, f"{n_frames} fr", (w - 150, 20), 0.42, (220, 220, 220), 1)
    tr_color = (80, 220, 120) if tracker_count > 0 else (235, 130, 80)
    _text(out, f"TR:{tracker_count}", (w - 150, 41), 0.38, tr_color, 1)

    if manual_control:
        _panel(out, (0, h - 28), (w, h), 0.42)
        _text(out, "A stop/save   B repeat   Y finish", (12, h - 9), 0.42, (230, 230, 230), 1)
    return out


class LaptopPreview:
    """Low-latency ffplay window fed with the exact laptop frame saved to the dataset."""

    def __init__(self, *, width: int, height: int, fps: int, title: str) -> None:
        self.width = width
        self.height = height
        self.proc: subprocess.Popen | None = None
        ffplay = shutil.which("ffplay")
        if ffplay is None:
            log.warning("ffplay not found; laptop preview window disabled.")
            return

        cmd = [
            ffplay,
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framedrop",
            "-sync",
            "ext",
            "-window_title",
            title,
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            log.info("Laptop preview window opened.")
        except OSError as exc:
            log.warning("Could not open laptop preview window: %s", exc)

    def show(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.poll() is not None or self.proc.stdin is None:
            return
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            log.warning("Laptop preview window closed.")
            self.close()

    def close(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
