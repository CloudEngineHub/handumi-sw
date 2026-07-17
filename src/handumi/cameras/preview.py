"""Small OpenCV preview helpers for laptop/context camera streams."""

from __future__ import annotations

import numpy as np


def draw_laptop_overlay(
    image: np.ndarray,
    *,
    label: str = "laptop",
) -> np.ndarray:
    """Return an RGB image with a small source label overlay."""
    frame = np.asarray(image).copy()
    try:
        import cv2
    except ImportError:
        return frame

    text = str(label)
    cv2.rectangle(frame, (8, 8), (8 + max(88, 10 * len(text)), 36), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


class LaptopPreview:
    """Best-effort local preview window for camera setup workflows."""

    def __init__(self, title: str = "HandUMI camera preview") -> None:
        self.title = title
        self._open = False

    def show(self, image: np.ndarray, *, label: str = "laptop") -> None:
        try:
            import cv2
        except ImportError:
            return
        frame = draw_laptop_overlay(image, label=label)
        cv2.imshow(self.title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        try:
            import cv2
        except ImportError:
            return
        cv2.destroyWindow(self.title)
        self._open = False


__all__ = ["LaptopPreview", "draw_laptop_overlay"]
