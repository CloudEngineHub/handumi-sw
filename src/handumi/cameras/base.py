"""Backend-neutral camera contracts for HandUMI recording."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraDevice(ABC):
    """Minimal camera interface used by HandUMI recorders."""

    @abstractmethod
    def connect(self) -> None:
        """Open the camera stream."""

    @abstractmethod
    def async_read(self) -> np.ndarray:
        """Return the latest RGB frame."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the camera stream."""
