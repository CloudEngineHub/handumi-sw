"""Initial joint-state distribution for an already converted dataset.

One first ``observation.state`` is sampled per episode, so every episode has
the same weight regardless of its duration.  Results are expressed in
HandUMI's canonical physical layout: arm joints in radians and logical
gripper openings in meters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from handumi.dataset.canonical import (
    canonical_joint_layout,
    is_canonical_state_layout,
)
from handumi.dataset.external_layouts import detect_external_layout, to_canonical
from handumi.robots.registry import load_embodiment

INITIAL_DISTRIBUTION_SCHEMA_VERSION = 1
INITIAL_DISTRIBUTION_KIND = "handumi_initial_joint_distribution"
STATE_KEY = "observation.state"


def initial_distribution(
    root: str | Path,
    *,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Return first-state statistics for a converted joint-level dataset."""
    dataset_root = Path(root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Dataset is missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    robot = _target_robot(info)
    runtime = load_embodiment(robot)
    canonical_layout = canonical_joint_layout(runtime)
    external_layout = None
    if not is_canonical_state_layout(info):
        external_layout = detect_external_layout(info)
        if external_layout is None:
            raise ValueError(
                "Initial distribution requires an already converted joint-level "
                "dataset. Run `handumi convert` first."
            )
        if external_layout.robot != robot:
            raise ValueError(
                f"Dataset layout describes {external_layout.robot!r}, but its "
                f"target robot is {robot!r}."
            )

    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError("Dataset metadata has no feature mapping")
    feature = features.get(STATE_KEY)
    if not isinstance(feature, dict):
        raise TypeError(f"Dataset has no {STATE_KEY!r} feature")
    expected_names = (
        canonical_layout.names if external_layout is None else external_layout.names
    )
    recorded_names = feature.get("names")
    if list(recorded_names or []) != list(expected_names):
        raise ValueError(
            f"{STATE_KEY} names do not match the declared joint layout. "
            f"Expected {expected_names}, got {recorded_names}."
        )

    episode_indices, values = _first_states(dataset_root)
    declared_episodes = int(info.get("total_episodes", 0))
    expected_indices = np.arange(declared_episodes, dtype=np.int64)
    if declared_episodes <= 0 or not np.array_equal(episode_indices, expected_indices):
        raise ValueError(
            "Joint data is incomplete: expected continuous episode indices "
            f"0..{declared_episodes - 1}, got {episode_indices.tolist()}."
        )
    if external_layout is not None:
        values = to_canonical(values, layout=external_layout, runtime=runtime)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] != canonical_layout.size:
        raise ValueError(
            f"Expected {canonical_layout.size} canonical joint values, got "
            f"{values.shape[1]}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Initial joint states contain non-finite values")

    mean = values.mean(axis=0)
    median = np.median(values, axis=0)
    scale = np.ones(canonical_layout.size, dtype=np.float64)
    gripper_columns: list[int] = []
    for column, side in enumerate(canonical_layout.gripper_sides):
        if side is not None:
            gripper_columns.append(column)
            scale[column] = max(float(runtime.config.gripper_max_width_m), 1e-9)
    distances = np.sqrt(np.mean(np.square((values - mean) / scale), axis=1))
    nearest_row = int(np.argmin(distances))

    source_layout = (
        "handumi_canonical" if external_layout is None else external_layout.name
    )
    return {
        "schema_version": INITIAL_DISTRIBUTION_SCHEMA_VERSION,
        "kind": INITIAL_DISTRIBUTION_KIND,
        "source": {
            "dataset": repo_id or f"local/{dataset_root.name}",
            "robot": robot,
            "feature": STATE_KEY,
            "sampling": "first_observation_per_episode",
            "episode_count": int(len(values)),
            "source_layout": source_layout,
            "joint_data_sha256": _joint_data_sha256(dataset_root),
        },
        "layout": {
            "representation": "handumi_canonical",
            "names": list(canonical_layout.names),
            "units": [
                "m" if column in gripper_columns else "rad"
                for column in range(canonical_layout.size)
            ],
        },
        "distribution": {
            "count": int(len(values)),
            "mean": _float_list(mean),
            "std_population": _float_list(values.std(axis=0, ddof=0)),
            "min": _float_list(values.min(axis=0)),
            "q01": _float_list(np.percentile(values, 1, axis=0)),
            "q10": _float_list(np.percentile(values, 10, axis=0)),
            "median": _float_list(median),
            "q90": _float_list(np.percentile(values, 90, axis=0)),
            "q99": _float_list(np.percentile(values, 99, axis=0)),
            "max": _float_list(values.max(axis=0)),
        },
        "recommended": {
            "default": "nearest_to_mean",
            "mean": {"values": _float_list(mean)},
            "median": {"values": _float_list(median)},
            "nearest_to_mean": {
                "values": _float_list(values[nearest_row]),
                "episode_index": int(episode_indices[nearest_row]),
                "distance_to_mean": float(distances[nearest_row]),
                "distance_metric": (
                    "root_mean_square; arm joints in radians; grippers divided "
                    "by gripper_max_width_m"
                ),
            },
        },
    }


def write_initial_distribution(path: str | Path, report: dict[str, Any]) -> Path:
    """Write an initial-distribution report as stable, human-readable YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def _target_robot(info: dict[str, Any]) -> str:
    handumi = info.get("handumi")
    target = handumi.get("target_robot") if isinstance(handumi, dict) else None
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    external = detect_external_layout(info)
    if external is not None:
        return external.robot
    raise ValueError(
        "Dataset records no target robot and matches no known external joint layout"
    )


def _first_states(root: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError(f"Dataset has no data Parquet files under {root / 'data'}")
    first_by_episode: dict[int, tuple[int, np.ndarray]] = {}
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", STATE_KEY])
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        states = table[STATE_KEY].to_pylist()
        for episode, frame, state in zip(episodes, frames, states, strict=True):
            episode_index = int(episode)
            frame_index = int(frame)
            previous = first_by_episode.get(episode_index)
            if previous is None or frame_index < previous[0]:
                first_by_episode[episode_index] = (
                    frame_index,
                    np.asarray(state, dtype=np.float64),
                )
    if not first_by_episode:
        raise ValueError("Dataset contains no episode rows")
    indices = np.asarray(sorted(first_by_episode), dtype=np.int64)
    states = np.stack([first_by_episode[int(index)][1] for index in indices])
    if states.ndim != 2:
        raise ValueError(f"Initial states must form a 2-D array, got {states.shape}")
    return indices, states


def _joint_data_sha256(root: Path) -> str:
    """Fingerprint the joint payload without hashing unrelated video files."""
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        paths.append(info_path)
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]
