"""Grading rules and the conversion gate for retargeting screening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from handumi.dataset.analysis import _load_quality_findings, dataset_payload_manifest
from handumi.dataset.screening import (
    RetargetScreeningConfig,
    _grade,
    _rotation_fences,
    evaluate_screening_gate,
    robot_fingerprint,
    screening_report_path,
    write_screening_report,
)

CFG = RetargetScreeningConfig()

# A dataset whose episodes come from two recording sessions: four of the ten
# were captured with a different tool orientation, the rest share one.
BIMODAL_ROTATIONS = {
    3: 17.3, 4: 17.7, 6: 22.1, 9: 15.9,
    20: 3.4, 23: 1.8, 25: 4.3, 29: 2.4, 34: 5.5, 36: 4.1,
}


def _metrics(**overrides) -> dict[str, float | int]:
    base = {
        "position_error_mean_m": 0.0002,
        "position_error_max_m": 0.0005,
        "position_error_p99_m": 0.0004,
        "position_error_frames_over_1cm": 0,
        "rotation_error_mean_deg": 2.0,
        "rotation_error_max_deg": 8.0,
        "initial_position_error_m": 0.003,
        "initial_solve_iterations": 1,
        "self_collision_min_clearance_m": 0.008,
        "self_collision_frames": 0,
        "table_min_clearance_m": 0.0,
        "table_penetration_frames": 0,
        "base_rotation_max_deg": 24.0,
    }
    base.update(overrides)
    return base


def _raw(index: int, **overrides) -> dict:
    return {
        "episode_index": index,
        "unreachable": None,
        "frame_count": 300,
        "metrics": _metrics(**overrides),
    }


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_sustained_position_error_is_a_rejection() -> None:
    """A miss the robot holds is a reachability failure."""
    report = _grade(
        _raw(0, position_error_max_m=0.05, position_error_p99_m=0.045),
        cfg=CFG,
        fps=30.0,
        fences={},
    )
    assert "retarget_position_error" in _codes(report)
    assert not report.accepted


def test_a_position_spike_is_only_a_warning() -> None:
    """Regression: one settling frame must not read as an unreachable episode.

    An episode can peak above the ceiling on the single frame after the start
    solve, while the joint-delta limit catches up, and then track the whole
    demonstration an order of magnitude tighter. Grading on the maximum
    discarded those alongside episodes that miss by the same margin for
    hundreds of consecutive frames, which is the real reachability failure.
    """
    strict = RetargetScreeningConfig(max_position_error_m=0.02)
    report = _grade(
        _raw(15, position_error_max_m=0.0299, position_error_p99_m=0.0024),
        cfg=strict,
        fps=30.0,
        fences={},
    )
    assert _codes(report) == {"retarget_position_spike"}
    assert report.accepted

    sustained = _grade(
        _raw(16, position_error_max_m=0.0306, position_error_p99_m=0.0296),
        cfg=strict,
        fps=30.0,
        fences={},
    )
    assert _codes(sustained) == {"retarget_position_error"}
    assert not sustained.accepted


def test_unreachable_start_pose_is_a_rejection() -> None:
    item = {"episode_index": 7, "unreachable": "start pose 6.5 cm", "frame_count": 0, "metrics": {}}
    report = _grade(item, cfg=CFG, fps=30.0, fences={})
    assert _codes(report) == {"retarget_start_pose_unreachable"}
    assert not report.accepted


def test_self_collision_is_a_warning_not_a_rejection() -> None:
    """A short fold-back is for a human to judge, not an automatic discard."""
    report = _grade(
        _raw(3, self_collision_frames=16, self_collision_min_clearance_m=-0.0157),
        cfg=CFG,
        fps=30.0,
        fences={},
    )
    assert "retarget_self_collision" in _codes(report)
    assert report.accepted  # warnings never reject on their own


def test_base_rotation_is_off_unless_a_limit_is_set() -> None:
    """The base swing is a mount and task property; the default only records it."""
    report = _grade(_raw(4, base_rotation_max_deg=96.4), cfg=CFG, fps=30.0, fences={})
    assert _codes(report) == set()
    assert report.metrics["base_rotation_max_deg"] == 96.4


def test_base_rotation_past_the_limit_is_a_rejection() -> None:
    """Setting the limit is the reviewer's decision, so exceeding it needs none."""
    cfg = RetargetScreeningConfig(max_base_rotation_deg=60.0)
    report = _grade(_raw(4, base_rotation_max_deg=96.4), cfg=cfg, fps=30.0, fences={})
    assert _codes(report) == {"retarget_base_rotation"}
    assert not report.accepted
    finding = report.findings[0]
    assert finding.metrics["limit_deg"] == 60.0
    assert _grade(_raw(5, base_rotation_max_deg=57.5), cfg=cfg, fps=30.0, fences={}).accepted


def test_base_rotation_metric_measures_the_swing_from_home() -> None:
    """Measured from home_q, not from zero, so a non-zero rest pose is not a swing."""
    from types import SimpleNamespace

    from handumi.dataset.screening import base_rotation_max_deg

    runtime = SimpleNamespace(
        arms={"left": None, "right": None},
        arm_joint_indices=lambda side: [0, 1, 2] if side == "left" else [3, 4, 5],
        config=SimpleNamespace(home_q=np.array([0.0, 0, 0, 0.5, 0, 0], dtype=np.float32)),
    )
    qpos = np.array(
        [[0.0, 1.0, 1.0, 0.5, 1.0, 1.0], [np.radians(-30.0), 0, 0, 0.5 + np.radians(45.0), 0, 0]],
        dtype=np.float32,
    )
    assert base_rotation_max_deg(qpos, runtime) == pytest.approx(45.0, abs=1e-3)


def test_table_contact_is_recorded_but_never_flagged() -> None:
    """Fingers touching the table during a grasp is the demonstration, not a fault."""
    report = _grade(
        _raw(9, table_min_clearance_m=-0.0037, table_penetration_frames=8),
        cfg=CFG,
        fps=30.0,
        fences={},
    )
    assert _codes(report) == set()
    assert report.metrics["table_penetration_frames"] == 8


def test_rotation_rule_catches_a_bimodal_session() -> None:
    """Regression: an IQR fence missed this because the cluster was 40% of the data.

    With four of ten episodes in the bad cluster, Q3 and the IQR both sit inside
    it, putting the fence at 38.6 deg and catching nothing. The median stays in
    the dominant cluster.
    """
    raw = [
        _raw(index, rotation_error_mean_deg=value)
        for index, value in BIMODAL_ROTATIONS.items()
    ]
    fences = _rotation_fences(raw, CFG)
    flagged = {
        item["episode_index"]
        for item in raw
        if "retarget_rotation_outlier"
        in _codes(_grade(item, cfg=CFG, fps=30.0, fences=fences))
    }
    assert flagged == {3, 4, 6, 9}


def test_rotation_rule_stays_quiet_on_a_uniform_dataset() -> None:
    """A tight distribution must not have its own best episodes flagged."""
    raw = [_raw(i, rotation_error_mean_deg=value) for i, value in enumerate([1.0, 1.2, 2.0, 3.0, 1.5, 2.4])]
    fences = _rotation_fences(raw, CFG)
    for item in raw:
        assert "retarget_rotation_outlier" not in _codes(
            _grade(item, cfg=CFG, fps=30.0, fences=fences)
        )


def _write_dataset(tmp_path: Path, *, episodes: int = 3) -> Path:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "total_frames": 300, "fps": 30}),
        encoding="utf-8",
    )
    return root


def _write_report(root: Path, *, robot: str, flagged: dict[int, list] | None = None) -> Path:
    flagged = flagged or {}
    payload = {
        "schema_version": 1,
        "kind": "handumi_retarget_screening",
        "robot": robot,
        "payload_manifest": dataset_payload_manifest(root),
        "deployment_calibration_path": None,
        "robot_fingerprint": robot_fingerprint(robot),
        "summary": {"total": 3, "accepted": 3, "rejected": 0},
        "episodes": [
            {
                "episode_index": index,
                "status": "accepted",
                "frame_count": 100,
                "duration_s": 3.3,
                "metrics": {},
                "findings": flagged.get(index, []),
            }
            for index in range(3)
        ],
    }
    return write_screening_report(screening_report_path(root, robot), payload)


def test_gate_blocks_when_no_report_exists(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path)
    gate = evaluate_screening_gate(root, robot="piper")
    assert gate.status == "missing"
    assert gate.blocks


def test_gate_blocks_when_the_dataset_changed_after_screening(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path)
    _write_report(root, robot="piper")
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 3, "total_frames": 999, "fps": 30}),
        encoding="utf-8",
    )
    gate = evaluate_screening_gate(root, robot="piper")
    assert gate.status == "stale"
    assert gate.blocks


def test_gate_blocks_rejections_but_not_warnings(tmp_path: Path) -> None:
    """Warnings are a reviewer's call, already made during curation.

    Blocking on them would make the override habitual, and that same override
    switches off the missing, stale and rejected checks.
    """
    root = _write_dataset(tmp_path)
    _write_report(
        root,
        robot="piper",
        flagged={1: [{"code": "retarget_self_collision", "severity": "warning"}]},
    )
    gate = evaluate_screening_gate(root, robot="piper")
    assert gate.status == "review"
    assert set(gate.flagged) == {1}
    assert not gate.blocks

    rejected = _write_dataset(tmp_path / "other")
    _write_report(
        rejected,
        robot="piper",
        flagged={
            1: [{"code": "retarget_position_error", "severity": "reject"}],
            2: [{"code": "retarget_self_collision", "severity": "warning"}],
        },
    )
    gate = evaluate_screening_gate(rejected, robot="piper")
    assert gate.status == "flagged"
    assert gate.blocks
    # Converting only the episodes without a rejection is allowed.
    assert not evaluate_screening_gate(
        rejected, robot="piper", episodes=[0, 2]
    ).blocks


def test_gate_blocks_when_the_robot_geometry_changed(tmp_path: Path) -> None:
    """Moving the arm bases invalidates every solution without touching the data.

    This is the hole the dataset manifest cannot see: the recorded episodes are
    untouched, so only the robot fingerprint catches it.
    """
    root = _write_dataset(tmp_path)
    path = _write_report(root, robot="piper")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["robot_fingerprint"]["urdf"] = "0" * 64  # as if the URDF was edited
    path.write_text(json.dumps(payload), encoding="utf-8")

    gate = evaluate_screening_gate(root, robot="piper")
    assert gate.status == "stale"
    assert "urdf" in gate.detail


def test_gate_blocks_a_report_written_before_fingerprinting(tmp_path: Path) -> None:
    """Reports from before this check must not be trusted by default."""
    root = _write_dataset(tmp_path)
    path = _write_report(root, robot="piper")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["robot_fingerprint"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert evaluate_screening_gate(root, robot="piper").status == "stale"


def test_gate_rejects_a_report_that_skipped_episodes(tmp_path: Path) -> None:
    """A report written with --episodes must not read as full coverage."""
    root = _write_dataset(tmp_path, episodes=5)  # report below only covers 0-2
    _write_report(root, robot="piper")
    gate = evaluate_screening_gate(root, robot="piper")
    assert gate.status == "stale"
    assert "3, 4" in gate.detail or "[3, 4]" in gate.detail


def test_gate_is_per_embodiment(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path)
    _write_report(root, robot="piper")
    assert not evaluate_screening_gate(root, robot="piper").blocks
    assert evaluate_screening_gate(root, robot="metal").status == "missing"


def test_report_feeds_dataset_analyze(tmp_path: Path) -> None:
    """The screening report must be consumable as an analyze quality report."""
    root = _write_dataset(tmp_path)
    path = _write_report(
        root,
        robot="piper",
        flagged={
            2: [
                {
                    "code": "retarget_rotation_outlier",
                    "severity": "warning",
                    "message": "inconsistent orientation",
                    "metrics": {},
                }
            ]
        },
    )
    findings, resolved = _load_quality_findings(root, quality_report=path)
    assert resolved == [path]
    assert findings[2][0]["code"] == "retarget_rotation_outlier"
    assert findings[2][0]["source"] == "screening_piper"


def test_analyze_merges_recording_and_retargeting_reports(tmp_path: Path) -> None:
    """Both dimensions must reach one review; neither may displace the other.

    Reading a single report is how a whole category of defect disappears
    between screening and curation.
    """
    root = _write_dataset(tmp_path)
    _write_report(
        root,
        robot="piper",
        flagged={
            1: [
                {
                    "code": "retarget_self_collision",
                    "severity": "warning",
                    "message": "folds into itself",
                    "metrics": {},
                }
            ]
        },
    )
    (root / "meta" / "handumi_quality.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_index": 2,
                        "status": "rejected",
                        "findings": [
                            {
                                "code": "sensor_sync_fraction",
                                "severity": "reject",
                                "message": "camera desynchronized",
                                "metrics": {},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    findings, resolved = _load_quality_findings(root)
    assert {path.name for path in resolved} == {
        "handumi_quality.json",
        "handumi_screening_piper.json",
    }
    assert findings[1][0]["source"] == "screening_piper"
    assert findings[2][0]["source"] == "quality"


def test_cached_solve_is_refused_when_solver_settings_differ(tmp_path: Path) -> None:
    """The cache must key on the settings, not just the dataset and robot.

    Reusing a trajectory solved under a different retarget mode or calibration
    would emit joints that silently disagree with the ones that were graded.
    """
    from argparse import Namespace

    from handumi.dataset.screening import (
        load_cached_solve,
        solve_cache_path,
        solver_signature,
    )

    root = _write_dataset(tmp_path)
    settings = Namespace(retarget_mode="absolute-table", translation_scale=1.0)
    signature = solver_signature(settings, deployment_path="table.yaml")
    path = _write_report(root, robot="piper")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["solver_signature"] = signature
    path.write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(
        solve_cache_path(root, "piper"),
        **{"0/qpos": np.zeros((4, 3), dtype=np.float32)},
    )

    assert load_cached_solve(root, robot="piper", episode=0, signature=signature)
    # A different retarget mode, and a different table, must both miss.
    other_mode = solver_signature(
        Namespace(retarget_mode="local-relative", translation_scale=1.0),
        deployment_path="table.yaml",
    )
    assert (
        load_cached_solve(root, robot="piper", episode=0, signature=other_mode) is None
    )
    other_table = solver_signature(settings, deployment_path="other.yaml")
    assert (
        load_cached_solve(root, robot="piper", episode=0, signature=other_table) is None
    )
    # An episode the screening never solved is not in the cache.
    assert load_cached_solve(root, robot="piper", episode=2, signature=signature) is None
