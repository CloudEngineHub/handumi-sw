"""``handumi dataset export`` must produce a drop-in for the external stack.

The export rewrites parquet, metadata and statistics at the file level, so
these tests build a tiny canonical dataset with the real writer and check the
result the way the external loader would: feature schema, column types,
statistics keys, and that the numbers still mean the same joints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from handumi.dataset.canonical import CANONICAL_STATE_LAYOUT
from handumi.dataset.external_layouts import BI_PIPER_FOLLOWER, to_canonical
from handumi.dataset.writer import EpisodeResult, write_dataset
from handumi.robots.registry import load_embodiment
from handumi.scripts.export_dataset import (
    DROPPED_COLUMNS,
    SCALAR_COLUMNS,
    export_dataset,
    parse_camera_map,
    schema_differences,
)

FPS = 30


@pytest.fixture(scope="module")
def runtime():
    return load_embodiment("piper")


def _episode(runtime, index: int, frames: int, *, seed: int) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    lower = np.array([e.lower_rad for e in BI_PIPER_FOLLOWER.arm_encodings])
    upper = np.array([e.upper_rad for e in BI_PIPER_FOLLOWER.arm_encodings])
    joints = np.zeros((frames + 1, 14), dtype=np.float32)
    for base in (0, 7):
        joints[:, base : base + 6] = rng.uniform(lower, upper, size=(frames + 1, 6))
        joints[:, base + 6] = rng.uniform(0.0, runtime.config.gripper_max_width_m, frames + 1)
    return EpisodeResult(
        episode_index=index,
        states=joints[:-1],
        actions=joints[1:],
        task="put the block in the puzzle",
        calibration_id=0,
        source_kind=0,
    )


@pytest.fixture
def canonical_dataset(tmp_path: Path, runtime) -> Path:
    root = tmp_path / "tblock-piper-joints"
    source = tmp_path / "raw"
    (source / "meta").mkdir(parents=True)
    write_dataset(
        output_root=root,
        source_root=source,
        source_info={"features": {}, "fps": FPS},
        episodes=[_episode(runtime, 0, 12, seed=0), _episode(runtime, 1, 9, seed=1)],
        robot_type="piper",
        joint_names=[
            *(f"left_joint{i}.pos" for i in range(1, 7)),
            "left_gripper.width_m",
            *(f"right_joint{i}.pos" for i in range(1, 7)),
            "right_gripper.width_m",
        ],
        fps=FPS,
        handumi_metadata={
            "state_layout": CANONICAL_STATE_LAYOUT,
            "target_robot": {"name": "piper"},
        },
    )
    return root


def _states(root: Path) -> np.ndarray:
    tables = [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]
    return np.concatenate(
        [np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)) for t in tables]
    ).astype(np.float32)


def test_export_is_a_drop_in(canonical_dataset: Path, runtime) -> None:
    output = canonical_dataset.parent / "exported"
    result = export_dataset(
        canonical_dataset,
        output,
        layout=BI_PIPER_FOLLOWER,
        camera_map={},
        source_repo_id="local/test",
        strict=True,
    )
    assert result["episodes"] == 2 and result["out_of_range"] == {}

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["robot_type"] == "bi_piper_follower"
    assert list(info["features"]) == ["action", "observation.state", *SCALAR_COLUMNS]
    for key in ("action", "observation.state"):
        assert info["features"][key]["names"] == BI_PIPER_FOLLOWER.names
        assert info["features"][key]["shape"] == [14]
    assert info["handumi"]["state_layout"] == "lerobot_bi_piper_follower"
    columns = info["handumi"]["export"]["columns"]
    assert [columns[n]["sign"] for n in BI_PIPER_FOLLOWER.names[:6]] == [-1, 1, 1, -1, 1, -1]
    assert columns["left_gripper.pos"]["mode"] == "range_0_100"

    data_file = sorted((output / "data").rglob("*.parquet"))[0]
    schema = pq.read_schema(data_file)
    assert schema.field("action").type == pa.list_(pa.float32(), 14)
    assert not any(name in schema.names for name in DROPPED_COLUMNS)

    stats = json.loads((output / "meta" / "stats.json").read_text())
    assert not any(name in stats for name in DROPPED_COLUMNS)
    assert set(stats["action"]) == {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert stats["action"]["count"] == [21]
    assert len(stats["action"]["q01"]) == 14

    episodes = pq.read_table(sorted((output / "meta" / "episodes").rglob("*.parquet"))[0])
    assert not any(c.startswith("stats/calibration_id/") for c in episodes.column_names)
    assert "stats/observation.state/q99" in episodes.column_names

    # The exported numbers still describe the same joints.
    restored = to_canonical(_states(output), layout=BI_PIPER_FOLLOWER, runtime=runtime)
    np.testing.assert_allclose(restored, _states(canonical_dataset), atol=1e-5)


def test_export_refuses_to_exist_twice_and_strict_aborts(canonical_dataset: Path, runtime) -> None:
    # Push one gripper past the full opening so the driver would reject it.
    data_file = sorted((canonical_dataset / "data").rglob("*.parquet"))[0]
    table = pq.read_table(data_file)
    states = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    states[0, 6] = 2.0 * runtime.config.gripper_max_width_m
    column = pa.array(list(states), type=pa.list_(pa.float32()))
    table = table.set_column(table.column_names.index("observation.state"), "observation.state", column)
    pq.write_table(table, data_file)

    output = canonical_dataset.parent / "strict"
    with pytest.raises(SystemExit, match="outside the range"):
        export_dataset(
            canonical_dataset, output, layout=BI_PIPER_FOLLOWER, camera_map={},
            source_repo_id="local/test", strict=True,
        )
    assert not output.exists()

    result = export_dataset(
        canonical_dataset, output, layout=BI_PIPER_FOLLOWER, camera_map={},
        source_repo_id="local/test", strict=False,
    )
    assert result["out_of_range"] == {"observation.state/left_gripper.pos": 1}


def test_camera_map_defaults_and_validation() -> None:
    keys = [
        "observation.images.left_wrist",
        "observation.images.workspace",
        "observation.images.right_wrist",
    ]
    default = parse_camera_map(None, layout=BI_PIPER_FOLLOWER, video_keys=keys)
    assert list(default.values()) == [
        "observation.images.left",
        "observation.images.top",
        "observation.images.right",
    ]
    custom = parse_camera_map("workspace=top", layout=BI_PIPER_FOLLOWER, video_keys=keys)
    assert custom["observation.images.workspace"] == "observation.images.top"
    assert custom["observation.images.left_wrist"] == "observation.images.left_wrist"
    with pytest.raises(SystemExit, match="no such camera"):
        parse_camera_map("nope=top", layout=BI_PIPER_FOLLOWER, video_keys=keys)
    with pytest.raises(SystemExit, match="same name"):
        parse_camera_map("workspace=top,left_wrist=top", layout=BI_PIPER_FOLLOWER, video_keys=keys)


def test_schema_differences_reports_what_breaks_a_drop_in() -> None:
    feature = {"dtype": "float32", "shape": [14], "names": BI_PIPER_FOLLOWER.names}
    video = {"dtype": "video", "shape": [376, 672, 3], "names": ["height", "width", "channels"]}
    reference = {
        "robot_type": "bi_piper_follower",
        "fps": 30,
        "features": {"action": feature, "observation.images.top": video},
    }
    assert schema_differences(reference, reference) == ([], [])
    changed = {**reference, "robot_type": "piper"}
    assert schema_differences(changed, reference)[0] == [
        "robot_type: exported='piper' reference='bi_piper_follower'"
    ]
    extra = {**reference, "features": {**reference["features"], "source_kind": {"dtype": "int64"}}}
    assert "feature only in export: source_kind" in schema_differences(extra, reference)[0]
    # A different camera resolution is a fact about the source, not a blocker;
    # 'channel' vs 'channels' is no difference at all.
    resized = {
        **reference,
        "features": {
            **reference["features"],
            "observation.images.top": {**video, "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
        },
    }
    blocking, notes = schema_differences(resized, reference)
    assert blocking == []
    assert len(notes) == 1 and "resolution" in notes[0]
