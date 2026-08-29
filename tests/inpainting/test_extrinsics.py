from __future__ import annotations

import cv2
import numpy as np
import pytest

from handumi.inpainting import (
    MarkerConfig,
    detect_marker,
    retarget_offset_px,
    solve_camera_from_table,
)

CAMERA = np.array([[290.6, 0.0, 336.0], [0.0, 288.5, 188.0], [0.0, 0.0, 1.0]])
DISTORTION = np.zeros(5)


def _scene(positions: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
           width: int = 672, height: int = 376) -> np.ndarray:
    """Render a bright marker moving over a static background, as the camera sees it."""
    projected, _ = cv2.projectPoints(positions, rvec, tvec, CAMERA, DISTORTION)
    frames = []
    for (u, v) in projected.reshape(-1, 2):
        frame = np.full((height, width, 3), 90, np.uint8)
        frame[::7, ::7] = 110                      # static texture
        cv2.circle(frame, (int(round(u)), int(round(v))), 9, (255, 255, 255), -1)
        frames.append(frame)
    return np.array(frames)


def _trajectory(n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.stack([0.25 * np.cos(2 * np.pi * t), 0.25 * np.sin(2 * np.pi * t),
                     0.05 + 0.02 * t], axis=1).astype(np.float64)


def test_a_known_camera_is_recovered_from_the_recording():
    """The marker's image path plus its table-frame path determine the pose."""
    truth_rvec = np.array([[2.2], [0.0], [0.0]])
    truth_tvec = np.array([[0.0], [0.05], [0.6]])
    positions = _trajectory()

    marker = detect_marker(_scene(positions, truth_rvec, truth_tvec))
    solved = solve_camera_from_table(marker, positions, CAMERA, DISTORTION)

    assert solved.inliers >= 30
    assert solved.mean_error_px < 2.0
    truth_position = (-cv2.Rodrigues(truth_rvec)[0].T @ truth_tvec).ravel()
    assert np.allclose(solved.camera_position_m, truth_position, atol=0.02)


def test_the_marker_is_found_in_most_frames():
    positions = _trajectory()
    marker = detect_marker(_scene(positions, np.array([[2.2], [0.0], [0.0]]),
                                  np.array([[0.0], [0.05], [0.6]])))
    found = (~np.isnan(marker[:, 0])) & (marker[:, 2] > MarkerConfig().min_area_px)
    assert found.mean() > 0.8


def test_too_few_detections_is_refused_with_a_reason():
    """Better to say the marker was not found than to return a made-up pose."""
    marker = np.full((40, 3), np.nan)
    with pytest.raises(ValueError, match="usable marker detections"):
        solve_camera_from_table(marker, _trajectory(40), CAMERA, DISTORTION)


def test_the_retarget_offset_is_reported_in_pixels():
    positions = _trajectory()
    solved = solve_camera_from_table(
        detect_marker(_scene(positions, np.array([[2.2], [0.0], [0.0]]),
                             np.array([[0.0], [0.05], [0.6]]))),
        positions, CAMERA, DISTORTION,
    )
    offset = retarget_offset_px(np.full(len(positions), 0.02), solved, positions, CAMERA)
    assert offset["mean_px"] > 0
    assert offset["max_px"] >= offset["mean_px"]


def test_a_perfect_retarget_shows_no_offset():
    positions = _trajectory()
    solved = solve_camera_from_table(
        detect_marker(_scene(positions, np.array([[2.2], [0.0], [0.0]]),
                             np.array([[0.0], [0.05], [0.6]]))),
        positions, CAMERA, DISTORTION,
    )
    assert retarget_offset_px(np.zeros(len(positions)), solved, positions, CAMERA)["mean_px"] == 0.0
