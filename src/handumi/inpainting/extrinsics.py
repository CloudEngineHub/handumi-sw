"""Recover the context camera's pose from a recording that never calibrated it.

``handumi calibrate spatial workspace`` solves ``table_from_camera`` from ChArUco
views, but a session that skipped that stage leaves the field empty and the
camera unusable for anything geometric. The recording itself still holds the
answer: the operator's tracking controller is a bright, compact marker whose
position in the table frame is stored per frame in ``observation.state``, so its
image trajectory and its known 3-D trajectory are a PnP correspondence set.

The residual is reported in pixels. Treat it as the measurement it is: the
marker's centroid sits a few centimetres from the controller origin it is
matched against, so a couple of pixels of error is the floor, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from handumi.inpainting.gates import read_video


@dataclass(frozen=True)
class MarkerConfig:
    """How the controller marker is separated from a static, bright scene."""

    foreground_threshold: int = 25
    min_value: int = 200
    max_saturation: int = 60
    min_area_px: int = 60


@dataclass(frozen=True)
class CameraFromTable:
    """The solved pose, with the evidence that produced it."""

    rvec: np.ndarray
    tvec: np.ndarray
    correspondences: int
    inliers: int
    mean_error_px: float
    median_error_px: float
    max_error_px: float

    @property
    def camera_position_m(self) -> np.ndarray:
        """Where the camera sits in the table frame."""
        rotation, _ = cv2.Rodrigues(self.rvec)
        return (-rotation.T @ self.tvec).ravel()

    def to_dict(self) -> dict[str, Any]:
        rotation, _ = cv2.Rodrigues(self.rvec)
        return {
            "source": "recovered_from_recording",
            "frame_convention": "camera_from_table, OpenCV rvec/tvec, metres",
            "rvec": self.rvec.ravel().tolist(),
            "tvec": self.tvec.ravel().tolist(),
            "camera_position_m": self.camera_position_m.tolist(),
            "rotation_matrix": rotation.tolist(),
            "metrics": {
                "correspondences": self.correspondences,
                "inliers": self.inliers,
                "mean_error_px": round(self.mean_error_px, 3),
                "median_error_px": round(self.median_error_px, 3),
                "max_error_px": round(self.max_error_px, 3),
            },
        }


def detect_marker(video: Path | np.ndarray, config: MarkerConfig | None = None) -> np.ndarray:
    """Track the controller marker, returning ``(u, v, area)`` per frame.

    The camera is fixed, so the scene's median is its background and the marker
    is what is bright, unsaturated and moving. A frame with no detection gets
    ``nan``.
    """
    config = config or MarkerConfig()
    frames = read_video(video) if isinstance(video, Path) else video
    background = np.median(frames, axis=0).astype(np.uint8)
    opening = np.ones((3, 3), np.uint8)
    closing = np.ones((7, 7), np.uint8)

    points = []
    for frame in frames:
        difference = np.abs(frame.astype(np.int16) - background.astype(np.int16)).mean(axis=2)
        moving = cv2.morphologyEx(
            (difference > config.foreground_threshold).astype(np.uint8), cv2.MORPH_OPEN, opening
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        marker = (
            (hsv[:, :, 2] > config.min_value)
            & (hsv[:, :, 1] < config.max_saturation)
            & (moving > 0)
        ).astype(np.uint8)
        marker = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, closing)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(marker)
        if count > 1:
            index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            points.append((centroids[index][0], centroids[index][1], stats[index, cv2.CC_STAT_AREA]))
        else:
            points.append((np.nan, np.nan, 0))
    return np.array(points, dtype=np.float64)


def solve_camera_from_table(
    marker_uv: np.ndarray,
    marker_xyz: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    config: MarkerConfig | None = None,
    reprojection_error_px: float = 8.0,
) -> CameraFromTable:
    """Solve the fixed camera pose from the marker's image and table trajectories."""
    config = config or MarkerConfig()
    usable = (~np.isnan(marker_uv[:, 0])) & (marker_uv[:, 2] > config.min_area_px)
    image_points = marker_uv[usable, :2]
    object_points = marker_xyz[usable].astype(np.float64)
    if len(image_points) < 6:
        raise ValueError(
            f"Only {len(image_points)} usable marker detections; PnP needs at least 6. "
            "The marker may be occluded, or the scene too bright to separate it."
        )

    solved, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=reprojection_error_px,
        iterationsCount=5000, confidence=0.999,
    )
    if not solved or inliers is None or len(inliers) < 6:
        raise ValueError("PnP did not converge on the marker trajectory.")

    # RANSAC returns indices; make that explicit so they can index arrays.
    keep = np.asarray(inliers, dtype=np.intp).reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[keep], image_points[keep], camera_matrix, distortion, rvec, tvec
    )
    projected, _ = cv2.projectPoints(object_points[keep], rvec, tvec, camera_matrix, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points[keep], axis=1)
    return CameraFromTable(
        rvec=rvec, tvec=tvec,
        correspondences=len(image_points), inliers=len(keep),
        mean_error_px=float(errors.mean()),
        median_error_px=float(np.median(errors)),
        max_error_px=float(errors.max()),
    )


def retarget_offset_px(
    position_error_m: np.ndarray,
    camera: CameraFromTable,
    marker_xyz: np.ndarray,
    camera_matrix: np.ndarray,
) -> dict[str, float]:
    """How far the retargeted gripper lands from where the operator's was, in pixels.

    Screening already measured that gap in metres. Converting it with the solved
    camera says what it looks like on screen -- which is what decides whether a
    viewer, or a policy, would see the robot grasping the object or beside it.
    """
    rotation, _ = cv2.Rodrigues(camera.rvec)
    depths = (rotation @ marker_xyz.T + camera.tvec.reshape(3, 1))[2]
    focal = float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2)
    valid = np.isfinite(depths) & (depths > 1e-6) & np.isfinite(position_error_m)
    pixels = position_error_m[valid] * focal / depths[valid]
    if pixels.size == 0:
        return {"mean_px": 0.0, "median_px": 0.0, "max_px": 0.0}
    return {
        "mean_px": round(float(pixels.mean()), 2),
        "median_px": round(float(np.median(pixels)), 2),
        "max_px": round(float(pixels.max()), 2),
    }
