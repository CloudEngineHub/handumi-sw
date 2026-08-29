from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from handumi.dataset.analysis import (
    analyze_dataset,
    load_analysis_report,
    render_analysis_markdown,
    write_analysis_report,
)


def _create_dataset(root: Path, lengths: list[int], *, fps: int = 1) -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["position", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["position", "gripper"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/source",
        fps=fps,
        features=features,
        root=root,
        use_videos=False,
    )
    for episode_index, length in enumerate(lengths):
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


def test_analysis_detects_outliers_without_automatic_removal(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _create_dataset(root, [2, 5, 6, 20])

    report = analyze_dataset(root, repo_id="local/source")

    assert report["candidates_for_review"] == [3]
    assert report["method"]["automatic_removal"] is False
    assert report["episodes"][0]["status"] == "accepted"
    assert report["episodes"][3]["status"] == "outlier"
    assert sum(item["count"] for item in report["summary"]["histogram"]) == 4
    assert report["summary"]["duration_seconds"]["mean"] == pytest.approx(8.25)
    assert report["summary"]["task_distribution_episodes"] == {"test task": 4}
    assert report["summary"]["state_action_alignment"]["exact_equal_fraction"] == 1.0
    assert "| 3 | 20.00 | outlier |" in render_analysis_markdown(report)
    assert "| shortest | 0 | 2.00 | accepted |" in render_analysis_markdown(report)


def test_analysis_merges_quality_rejections_and_round_trips_report(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _create_dataset(root, [5, 5, 5])
    quality_path = root / "meta" / "handumi_quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_index": 1,
                        "status": "rejected",
                        "findings": [
                            {
                                "code": "tracking_quality_fraction",
                                "severity": "reject",
                                "message": "tracking failed",
                                "metrics": {"fraction": 0.2},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_dataset(root)
    report_path = write_analysis_report(root / "meta" / "analysis.json", report)

    assert report["candidates_for_review"] == [1]
    finding = report["episodes"][1]["findings"][0]
    # The producer is named so a merged review says which dimension objected.
    assert finding["source"] == "quality"
    assert load_analysis_report(report_path) == report
