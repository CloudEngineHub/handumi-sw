from __future__ import annotations

import numpy as np

from handumi.visualize import OpenCVCameraViewer


def test_opencv_viewer_only_queues_selected_cameras_side_by_side() -> None:
    viewer = OpenCVCameraViewer(["left_wrist", "right_wrist"])
    left = np.full((3, 4, 3), 10, dtype=np.uint8)
    right = np.full((3, 5, 3), 20, dtype=np.uint8)
    workspace = np.full((3, 6, 3), 30, dtype=np.uint8)

    viewer.submit(
        {
            "observation.images.left_wrist": left,
            "observation.images.right_wrist": right,
            "observation.images.workspace": workspace,
        }
    )

    images = viewer._queue.get_nowait()
    assert isinstance(images, tuple)
    preview = viewer._side_by_side(None, images)
    assert preview.shape == (3, 9, 3)
    np.testing.assert_array_equal(preview[:, :4], left)
    np.testing.assert_array_equal(preview[:, 4:], right)
    assert not np.any(preview == 30)


def test_opencv_viewer_replaces_stale_preview_without_blocking() -> None:
    viewer = OpenCVCameraViewer(["left_wrist"])
    old = np.zeros((2, 2, 3), dtype=np.uint8)
    newest = np.ones((2, 2, 3), dtype=np.uint8)

    viewer.submit({"observation.images.left_wrist": old})
    viewer.submit({"observation.images.left_wrist": newest})

    images = viewer._queue.get_nowait()
    assert isinstance(images, tuple)
    np.testing.assert_array_equal(images[0], newest)
    assert viewer.dropped_batches == 1
