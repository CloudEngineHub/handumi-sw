import queue

import numpy as np

from handumi.visualize.cameras import RerunCameraViewer


def _frame(value: int) -> dict[str, np.ndarray]:
    return {
        "observation.images.left_wrist": np.full(
            (2, 3, 3), value, dtype=np.uint8
        )
    }


def test_submit_replaces_stale_preview_instead_of_blocking() -> None:
    viewer = RerunCameraViewer(
        ["left_wrist"], application_id="handumi_test_cameras"
    )

    viewer.submit(_frame(1), capture_time_ns=1)
    viewer.submit(_frame(2), capture_time_ns=2)

    capture_time_ns, images = viewer._queue.get_nowait()
    assert capture_time_ns == 2
    assert images["observation.images.left_wrist"][0, 0, 0] == 2
    assert viewer.dropped_batches == 1


def test_submit_copies_camera_memory_before_returning() -> None:
    viewer = RerunCameraViewer(
        ["left_wrist"], application_id="handumi_test_cameras"
    )
    frames = _frame(7)

    viewer.submit(frames, capture_time_ns=1)
    frames["observation.images.left_wrist"].fill(99)

    _, images = viewer._queue.get_nowait()
    assert images["observation.images.left_wrist"][0, 0, 0] == 7


def test_submit_is_ignored_after_close() -> None:
    viewer = RerunCameraViewer(
        ["left_wrist"], application_id="handumi_test_cameras"
    )

    viewer.close(timeout_s=0.0)
    viewer.submit(_frame(1))

    item = viewer._queue.get_nowait()
    assert item is not None
    with np.testing.assert_raises(queue.Empty):
        viewer._queue.get_nowait()
