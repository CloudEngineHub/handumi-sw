"""Fixed-rate joint streamer infrastructure shared by all arm drivers."""

from __future__ import annotations

import threading

import numpy as np


def step_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Clamp each joint's delta to *max_step* per tick toward *target*.

    Returns a float64 array.  The caller is responsible for any further
    dtype casting (e.g. rounding to int64 for milli-degree arms).
    If *max_step* <= 0 the target is returned immediately.
    """
    current_f = np.asarray(current, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    if max_step <= 0.0:
        return target_f.copy()
    return current_f + np.clip(target_f - current_f, -max_step, max_step)


class JointStreamer:
    """Daemon thread + lock + error bookkeeping for fixed-rate arm streamers.

    Subclasses must implement :meth:`_run`.  All shared infrastructure
    (threading.Thread, threading.Lock, threading.Event, error propagation)
    lives here so individual arm drivers only contain arm-specific logic.

    Typical subclass pattern::

        class MyStreamer(JointStreamer):
            def __init__(self, arms, *, command_rate_hz, ...):
                super().__init__(command_rate_hz=command_rate_hz, thread_name="my-streamer")
                self.arms = arms
                ...

            def _run(self) -> None:
                period = 1.0 / self.command_rate_hz
                ...
    """

    def __init__(self, *, command_rate_hz: float, thread_name: str) -> None:
        if command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be > 0")
        self.command_rate_hz = float(command_rate_hz)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{type(self).__name__} failed") from self._error

    def _run(self) -> None:
        raise NotImplementedError


__all__ = ["JointStreamer", "step_toward"]
