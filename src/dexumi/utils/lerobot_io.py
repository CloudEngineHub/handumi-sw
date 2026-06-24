"""Small wrappers around LeRobot dataset loading.

This module keeps LeRobot imports isolated so the base project can stay light.
Install the optional dependency with:

    GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetDownloadResult:
    """Summary of a downloaded or loaded LeRobot dataset."""

    repo_id: str
    root: Path
    num_episodes: int
    num_frames: int
    fps: int
    features: tuple[str, ...]


def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def download_lerobot_dataset(
    repo_id: str,
    output_dir: str | Path,
    *,
    revision: str | None = "main",
    episodes: str | None = None,
    download_videos: bool = True,
    force_cache_sync: bool = False,
) -> DatasetDownloadResult:
    """Download or load a LeRobot dataset using LeRobot's own dataset wrapper."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install it with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot"
        ) from exc

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(output_dir),
        revision=revision,
        episodes=_parse_episodes(episodes),
        download_videos=download_videos,
        force_cache_sync=force_cache_sync,
    )

    return DatasetDownloadResult(
        repo_id=dataset.repo_id,
        root=Path(dataset.root),
        num_episodes=dataset.num_episodes,
        num_frames=dataset.num_frames,
        fps=dataset.fps,
        features=tuple(dataset.features.keys()),
    )


def load_lerobot_dataset(
    repo_id: str,
    root: str | Path,
    *,
    episode: int | None = None,
    revision: str | None = "main",
    download_videos: bool = False,
) -> Any:
    """Load a local or remote LeRobot dataset with LeRobot's reader."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install it with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot"
        ) from exc

    episodes = None if episode is None else [episode]
    return LeRobotDataset(
        repo_id=repo_id,
        root=Path(root),
        revision=revision,
        episodes=episodes,
        download_videos=download_videos,
    )


def load_pico_body_poses(
    repo_id: str,
    root: str | Path,
    *,
    episode: int = 0,
    column: str = "observation.pico.body_joints_pose",
    revision: str | None = "main",
) -> tuple[Any, int]:
    """Load PICO body joint poses from a LeRobot dataset.

    Returns:
        A tuple ``(poses, fps)`` where ``poses`` has shape ``(T, 24, 7)``.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required. Install with: uv sync --extra viewer") from exc

    dataset = load_lerobot_dataset(
        repo_id=repo_id,
        root=root,
        episode=episode,
        revision=revision,
        download_videos=False,
    )

    if column not in dataset.features:
        available = "\n".join(f"  - {name}" for name in dataset.features)
        raise KeyError(f"Column {column!r} not found. Available features:\n{available}")

    frames = []
    for idx in range(len(dataset)):
        item = dataset.get_raw_item(idx)
        frames.append(np.asarray(item[column], dtype=np.float32))

    if not frames:
        raise ValueError(f"No frames found for episode {episode}.")

    poses = np.stack(frames, axis=0)
    if poses.ndim != 3 or poses.shape[1:] != (24, 7):
        raise ValueError(f"Expected poses with shape (T, 24, 7), got {poses.shape}.")

    return poses, int(dataset.fps)
