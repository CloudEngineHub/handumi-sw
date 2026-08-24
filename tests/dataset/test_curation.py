from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from handumi.audio import AUDIO_KEY, AUDIO_PATH, audio_metadata
from handumi.dataset.analysis import analyze_dataset, write_analysis_report
from handumi.dataset.curation import (
    curate_dataset,
    plan_dataset_curation,
)


def _create_dataset(root: Path) -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["episode", "frame"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["episode", "frame"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/source",
        fps=1,
        features=features,
        root=root,
        use_videos=False,
    )
    for episode_index, length in enumerate([2, 5, 6]):
        for frame_index in range(length):
            value = np.array([episode_index, frame_index], dtype=np.float32)
            dataset.add_frame(
                {
                    "observation.state": value,
                    "action": value,
                    "task": "test task",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["handumi"] = {
        "capture_schema": "test_capture",
        "calibration": {"sha256": "abc123"},
        "audio": audio_metadata(True),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for episode_index in range(3):
        audio_path = root / AUDIO_PATH.format(
            audio_key=AUDIO_KEY,
            chunk_index=0,
            file_index=episode_index,
        )
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(bytes([episode_index, 0]) * 8)


def _analysis_report(root: Path) -> Path:
    report = analyze_dataset(root, repo_id="local/source")
    return write_analysis_report(root / "meta" / "handumi_analysis.json", report)


def _create_video_dataset(root: Path) -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["episode", "frame"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["episode", "frame"],
        },
        "observation.images.test": {
            "dtype": "video",
            "shape": (16, 16, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/video-source",
        fps=5,
        features=features,
        root=root,
        use_videos=True,
        vcodec="h264",
    )
    for episode_index, length in enumerate([2, 5, 6]):
        for frame_index in range(length):
            value = np.array([episode_index, frame_index], dtype=np.float32)
            image = np.full(
                (16, 16, 3),
                episode_index * 50 + frame_index,
                dtype=np.uint8,
            )
            dataset.add_frame(
                {
                    "observation.state": value,
                    "action": value,
                    "observation.images.test": image,
                    "task": "video task",
                }
            )
        dataset.save_episode()
    dataset.finalize()


def test_curation_creates_reindexed_validated_derivative(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "clean"
    _create_dataset(source)
    report_path = _analysis_report(source)
    source_info_before = (source / "meta" / "info.json").read_bytes()

    plan = plan_dataset_curation(
        source,
        analysis_path=report_path,
        output_root=output,
        exclude_episode_indices=[0],
    )
    result = curate_dataset(plan)

    assert result.total_episodes == 2
    assert result.total_frames == 11
    assert result.excluded_source_episode_indices == (0,)
    assert source_info_before == (source / "meta" / "info.json").read_bytes()
    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["handumi"]["calibration"] == {"sha256": "abc123"}
    assert info["handumi"]["curation"]["removed_source_episode_indices"] == [0]
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 11

    data = pq.read_table(sorted((output / "data").glob("chunk-*/*.parquet")))
    assert data["episode_index"].to_pylist() == [0] * 5 + [1] * 6
    values = np.asarray(data["observation.state"].to_pylist())
    assert np.all(values[:5, 0] == 1)
    assert np.all(values[5:, 0] == 2)

    curation = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert curation["episode_index_mapping"] == {"1": 0, "2": 1}
    assert curation["audio_files_copied"] == 2
    assert curation["validation"]["lerobot_load"] == "ok"
    assert curation["validation"]["audio"] == {"enabled": True, "files": 2}
    first_audio = output / AUDIO_PATH.format(
        audio_key=AUDIO_KEY,
        chunk_index=0,
        file_index=0,
    )
    with wave.open(str(first_audio), "rb") as audio:
        assert audio.readframes(1) == bytes([1, 0])
    assert not any(output.parent.glob(f".{output.name}.curating-*"))


def test_plan_requires_human_exclusions_and_detects_stale_report(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _create_dataset(source)
    report_path = _analysis_report(source)

    with pytest.raises(ValueError, match="explicitly confirmed"):
        plan_dataset_curation(
            source,
            analysis_path=report_path,
            output_root=tmp_path / "kept",
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dataset"]["total_frames"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        plan_dataset_curation(
            source,
            analysis_path=report_path,
            output_root=tmp_path / "stale",
            exclude_episode_indices=[0],
        )


def test_plan_detects_payload_change_with_unchanged_totals(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _create_dataset(source)
    report_path = _analysis_report(source)
    stats_path = source / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["index"]["mean"] = [999.0]
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="payload fingerprint changed"):
        plan_dataset_curation(
            source,
            analysis_path=report_path,
            output_root=tmp_path / "changed",
            exclude_episode_indices=[0],
        )


def test_curation_reencodes_shared_video_with_source_codec(tmp_path: Path) -> None:
    source = tmp_path / "video-source"
    output = tmp_path / "video-clean"
    _create_video_dataset(source)
    report = analyze_dataset(source, repo_id="local/video-source")
    report_path = write_analysis_report(
        source / "meta" / "handumi_analysis.json",
        report,
    )

    result = curate_dataset(
        plan_dataset_curation(
            source,
            analysis_path=report_path,
            output_root=output,
            exclude_episode_indices=[0],
        )
    )

    video = result.validation["video"]["observation.images.test"]
    assert video["total_frames"] == 11
    assert len(video["files"]) == 1
    assert video["files"][0]["codec"] == "h264"
    assert video["files"][0]["pixel_format"] == "yuv420p"
