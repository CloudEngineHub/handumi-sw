from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from handumi.dataset.canonical import canonical_joint_layout
from handumi.dataset.external_layouts import BI_PIPER_FOLLOWER, from_canonical
from handumi.dataset.initial_distribution import (
    initial_distribution,
    write_initial_distribution,
)
from handumi.robots.registry import load_embodiment


def _joint_dataset(root: Path, starts: list[np.ndarray]) -> None:
    runtime = load_embodiment("piper")
    layout = canonical_joint_layout(runtime)
    info = {
        "robot_type": "piper",
        "total_episodes": len(starts),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [layout.size],
                "names": layout.names,
            }
        },
        "handumi": {
            "state_layout": "yaml_arm_joints_plus_logical_gripper_width_m",
            "target_robot": {"name": "piper"},
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    for episode, start in enumerate(starts):
        path = root / "data" / "chunk-000" / f"file-{episode:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        later = start + np.float32(100.0)
        pd.DataFrame(
            {
                "episode_index": [episode, episode],
                "frame_index": [0, 1],
                "observation.state": [start.tolist(), later.tolist()],
            }
        ).to_parquet(path, index=False)


def test_distribution_uses_one_first_state_per_episode(tmp_path: Path) -> None:
    runtime = load_embodiment("piper")
    width = canonical_joint_layout(runtime).size
    starts = [
        np.zeros(width, dtype=np.float32),
        np.full(width, 2.0, dtype=np.float32),
        np.full(width, 4.0, dtype=np.float32),
    ]
    root = tmp_path / "piper-joints"
    _joint_dataset(root, starts)

    report = initial_distribution(root)

    assert report["source"]["episode_count"] == 3
    assert report["source"]["sampling"] == "first_observation_per_episode"
    assert report["distribution"]["mean"] == pytest.approx([2.0] * width)
    assert report["distribution"]["median"] == pytest.approx([2.0] * width)
    assert report["recommended"]["nearest_to_mean"]["episode_index"] == 1
    assert report["recommended"]["nearest_to_mean"]["values"] == pytest.approx(
        [2.0] * width
    )


def test_distribution_writes_external_yaml(tmp_path: Path) -> None:
    runtime = load_embodiment("piper")
    width = canonical_joint_layout(runtime).size
    root = tmp_path / "piper-joints"
    _joint_dataset(root, [np.zeros(width, dtype=np.float32)])
    report = initial_distribution(root, repo_id="local/piper-joints")

    output = write_initial_distribution(tmp_path / "profiles" / "initial.yaml", report)
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert output.is_file()
    assert payload["kind"] == "handumi_initial_joint_distribution"
    assert payload["source"]["dataset"] == "local/piper-joints"
    assert payload["layout"]["units"].count("m") == 2


def test_distribution_decodes_external_follower_layout(tmp_path: Path) -> None:
    runtime = load_embodiment("piper")
    canonical_layout = canonical_joint_layout(runtime)
    # Build the canonical command explicitly because logical grippers have no
    # URDF qpos index in the dataset vector.
    canonical = np.zeros(canonical_layout.size, dtype=np.float32)
    for column, index in enumerate(canonical_layout.indices):
        if index is not None:
            canonical[column] = runtime.config.home_q[index]
    external = from_canonical(
        canonical[None], layout=BI_PIPER_FOLLOWER, runtime=runtime
    )[0]

    root = tmp_path / "piper-follower"
    _joint_dataset(root, [external])
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["robot_type"] = BI_PIPER_FOLLOWER.robot_type
    info["features"]["observation.state"]["names"] = BI_PIPER_FOLLOWER.names
    info["handumi"]["state_layout"] = BI_PIPER_FOLLOWER.name
    info_path.write_text(json.dumps(info), encoding="utf-8")

    report = initial_distribution(root)

    assert report["source"]["source_layout"] == BI_PIPER_FOLLOWER.name
    assert report["distribution"]["mean"] == pytest.approx(canonical, abs=1e-6)


def test_distribution_rejects_missing_episodes(tmp_path: Path) -> None:
    runtime = load_embodiment("piper")
    width = canonical_joint_layout(runtime).size
    root = tmp_path / "incomplete-piper-joints"
    _joint_dataset(root, [np.zeros(width, dtype=np.float32)])
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = 2
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="Joint data is incomplete"):
        initial_distribution(root)
