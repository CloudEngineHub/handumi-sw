"""Put the model's edit back into the recorded frames through a mask.

The model returns a fully re-rendered clip: every pixel is new, including the
ones that had to stay. Compositing through a mask keeps the table, the objects
and the scene clutter as the pixels the camera actually recorded, so only the
region that had to change does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskConfig:
    """How the edit region is derived and grown before compositing."""

    difference_threshold: int = 28
    kernel: int = 21
    dilate_iterations: int = 2
    temporal_radius: int = 2
    feather: int = 31
    skin_threshold: int = 18
    anchor_dilate_iterations: int = 3


def operator_footprint(source: np.ndarray, config: MaskConfig | None = None) -> np.ndarray:
    """Where the operator and the hand-held rig actually are in the recording.

    Skin plus whatever moves: the arm, the shells and the cables are the only
    things in a fixed-camera tabletop recording that move with the hand.
    """
    config = config or MaskConfig()
    skin = np.array([
        cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb),
                    np.array([0, 133, 77], np.uint8),
                    np.array([255, 173, 127], np.uint8))
        for f in source
    ]) > 0
    motion = np.abs(source[1:].astype(np.int16) - source[:-1].astype(np.int16)).mean(axis=3)
    motion = np.concatenate([motion[:1], motion]) > config.skin_threshold
    kernel = np.ones((25, 25), np.uint8)
    footprint = np.array([
        cv2.dilate((s | m).astype(np.uint8), kernel, iterations=config.anchor_dilate_iterations)
        for s, m in zip(skin, motion, strict=True)
    ])
    radius = config.temporal_radius
    if radius:
        padded = np.pad(footprint, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        footprint = np.max(
            np.stack([padded[i:i + len(footprint)] for i in range(2 * radius + 1)]), axis=0
        )
    return footprint > 0


def edit_mask(
    source: np.ndarray,
    generated: np.ndarray,
    config: MaskConfig | None = None,
    *,
    anchored: bool = True,
) -> np.ndarray:
    """Return a feathered per-frame alpha for where the model changed the scene.

    Derived from the source/generated difference and then bounded by the region
    the operator ever reached. The model re-renders every frame, so the
    difference alone cannot tell the intended edit from global drift -- and most
    of it is drift. Bounding it keeps recorded pixels wherever the operator
    never was, and bounding it by a region that does not move keeps the seam
    from sweeping across the scene.
    """
    config = config or MaskConfig()
    diff = np.abs(source.astype(np.int16) - generated.astype(np.int16)).mean(axis=3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel, config.kernel))
    masks = []
    for frame in diff:
        mask = (cv2.GaussianBlur(frame, (9, 9), 0) > config.difference_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks.append(cv2.dilate(mask, kernel, iterations=config.dilate_iterations))
    masks = np.array(masks)

    if anchored:
        # Anchor to where the operator ever was, not to where it is each frame.
        # The camera is fixed, so that region is constant; a per-frame boundary
        # sweeps the scene instead, and every static pixel it crosses swaps
        # recorded for generated pixels, which reads as flicker.
        reachable = operator_footprint(source, config).any(axis=0)
        masks = np.minimum(masks, reachable.astype(np.uint8) * 255)

    # A mask that flickers makes the seam flicker, so widen it over time.
    radius = config.temporal_radius
    if radius:
        padded = np.pad(masks, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        masks = np.max(np.stack([padded[i:i + len(masks)] for i in range(2 * radius + 1)]), axis=0)
    masks = np.array([cv2.GaussianBlur(m, (config.feather, config.feather), 0) for m in masks])
    return masks


def composite(source: np.ndarray, generated: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Blend generated pixels inside the mask over the recorded frames."""
    alpha = (masks.astype(np.float32) / 255.0)[..., None]
    blended = generated.astype(np.float32) * alpha + source.astype(np.float32) * (1.0 - alpha)
    return blended.astype(np.uint8)
