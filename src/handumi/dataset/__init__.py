"""LeRobot dataset read/write boundary for handumi."""

from typing import Any

from handumi.dataset.reader import (
    DatasetDownloadResult,
    download_dataset,
    ensure_metadata,
    open_dataset,
)
from handumi.dataset.raw import (
    HANDUMI_RAW_IMAGE_KEYS,
    HANDUMI_RAW_STATE_NAMES,
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
    raw_state_feature,
    validate_raw_state_shape,
)
from handumi.dataset.reader import DatasetRef, dataset_root_from_repo_id


def __getattr__(name: str) -> Any:
    """Lazily expose writer symbols without importing pandas at package import time."""
    writer_symbols = {
        "CHUNKS_SIZE",
        "EpisodeResult",
        "chunk_and_file",
        "info_path",
        "load_info",
        "write_dataset",
    }
    if name in writer_symbols:
        from handumi.dataset.writer import (
            CHUNKS_SIZE,
            EpisodeResult,
            chunk_and_file,
            info_path,
            load_info,
            write_dataset,
        )

        return locals()[name]
    if name == "load_pico_body_poses":
        from handumi.devices.pico import load_pico_body_poses

        return load_pico_body_poses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CHUNKS_SIZE",
    "DatasetDownloadResult",
    "DatasetRef",
    "EpisodeResult",
    "HANDUMI_RAW_IMAGE_KEYS",
    "HANDUMI_RAW_STATE_NAMES",
    "HANDUMI_RAW_STATE_SIZE",
    "LEFT_GRIPPER_INDEX",
    "LEFT_POSE_SLICE",
    "RIGHT_GRIPPER_INDEX",
    "RIGHT_POSE_SLICE",
    "chunk_and_file",
    "dataset_root_from_repo_id",
    "download_dataset",
    "ensure_metadata",
    "info_path",
    "load_info",
    "load_pico_body_poses",
    "open_dataset",
    "raw_state_feature",
    "validate_raw_state_shape",
    "write_dataset",
]
