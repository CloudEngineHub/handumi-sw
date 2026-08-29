"""Dataset-level statistics and auditable episode outlier analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ANALYSIS_SCHEMA_VERSION = 2
ANALYSIS_KIND = "handumi_dataset_analysis"
IQR_MULTIPLIER = 1.5
HISTOGRAM_BIN_S = 10.0


def analyze_dataset(
    root: str | Path,
    *,
    repo_id: str | None = None,
    quality_report: str | Path | None = None,
    quality_reports: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Analyze episode lengths and quality findings without changing dataset data."""
    dataset_root = Path(root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Dataset is missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"Dataset reports invalid fps: {fps}")

    episode_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")
    )
    if not episode_files:
        raise ValueError("Dataset has no episode metadata Parquet files")
    episodes_table = pq.read_table(
        episode_files, columns=["episode_index", "length", "tasks"]
    )
    episode_indices = np.asarray(
        episodes_table["episode_index"].to_pylist(), dtype=np.int64
    )
    lengths = np.asarray(episodes_table["length"].to_pylist(), dtype=np.int64)
    if len(lengths) == 0:
        raise ValueError("Dataset contains no episodes")
    expected_indices = np.arange(len(lengths), dtype=np.int64)
    if not np.array_equal(episode_indices, expected_indices):
        raise ValueError("Episode metadata indices must be continuous and zero-based")
    if np.any(lengths <= 0):
        raise ValueError("Every episode must contain at least one frame")

    declared_episodes = int(info.get("total_episodes", -1))
    declared_frames = int(info.get("total_frames", -1))
    if declared_episodes != len(lengths):
        raise ValueError(
            f"Episode count mismatch: info.json={declared_episodes}, metadata={len(lengths)}"
        )
    if declared_frames != int(lengths.sum()):
        raise ValueError(
            f"Frame count mismatch: info.json={declared_frames}, metadata={int(lengths.sum())}"
        )

    durations = lengths.astype(np.float64) / fps
    q1, median, q3 = np.quantile(durations, [0.25, 0.5, 0.75])
    iqr = float(q3 - q1)
    lower_fence = float(q1 - IQR_MULTIPLIER * iqr)
    upper_fence = float(q3 + IQR_MULTIPLIER * iqr)
    quality_by_episode, resolved_quality_reports = _load_quality_findings(
        dataset_root,
        quality_report=quality_report,
        quality_reports=quality_reports,
    )

    episode_results: list[dict[str, Any]] = []
    for position, (episode_index, length, duration, tasks) in enumerate(
        zip(
            episode_indices.tolist(),
            lengths.tolist(),
            durations.tolist(),
            episodes_table["tasks"].to_pylist(),
            strict=True,
        )
    ):
        findings: list[dict[str, Any]] = []
        if duration < lower_fence or duration > upper_fence:
            findings.append(
                {
                    "code": "duration_iqr_outlier",
                    "severity": "warning",
                    "message": "Episode duration lies outside the automatic IQR fences.",
                    "metrics": {
                        "duration_s": duration,
                        "lower_fence_s": lower_fence,
                        "upper_fence_s": upper_fence,
                    },
                }
            )
        findings.extend(quality_by_episode.get(episode_index, ()))
        findings = _deduplicate_findings(findings)
        rejected = any(item.get("severity") == "reject" for item in findings)
        warned = any(item.get("severity") == "warning" for item in findings)
        status = "rejected" if rejected else "outlier" if warned else "accepted"
        episode_results.append(
            {
                "source_episode_index": int(episode_index),
                "position": position,
                "frame_count": int(length),
                "duration_s": float(duration),
                "tasks": list(tasks or []),
                "status": status,
                "findings": findings,
            }
        )

    histogram = _duration_histogram(durations, HISTOGRAM_BIN_S)
    task_distribution: dict[str, int] = {}
    for tasks in episodes_table["tasks"].to_pylist():
        for task in tasks or []:
            task_distribution[str(task)] = task_distribution.get(str(task), 0) + 1
    candidates_for_review = [
        item["source_episode_index"]
        for item in episode_results
        if item["status"] != "accepted"
    ]
    resolved_repo_id = repo_id or f"local/{dataset_root.name}"
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "repo_id": resolved_repo_id,
            "root": str(dataset_root.resolve()),
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": fps,
            "total_episodes": len(lengths),
            "total_frames": int(lengths.sum()),
            "payload_manifest": dataset_payload_manifest(dataset_root),
        },
        "method": {
            "duration_outlier": "tukey_iqr",
            "duration_iqr_multiplier": IQR_MULTIPLIER,
            "histogram_bin_s": HISTOGRAM_BIN_S,
            "quality_rejections_included": True,
            "automatic_removal": False,
        },
        "quality_reports": [
            str(path.resolve()) for path in resolved_quality_reports
        ],
        "quality_report": (
            str(resolved_quality_reports[0].resolve())
            if resolved_quality_reports
            else None
        ),
        "summary": {
            "duration_frames": {
                "mean": float(lengths.mean()),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "std_population": float(lengths.std(ddof=0)),
            },
            "duration_seconds": {
                "mean": float(durations.mean()),
                "min": float(durations.min()),
                "max": float(durations.max()),
                "std_population": float(durations.std(ddof=0)),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
            },
            "histogram": histogram,
            "accepted": sum(item["status"] == "accepted" for item in episode_results),
            "outliers_for_review": sum(
                item["status"] == "outlier" for item in episode_results
            ),
            "quality_rejected": sum(
                item["status"] == "rejected" for item in episode_results
            ),
            "task_distribution_episodes": task_distribution,
            "storage_bytes": _storage_bytes(dataset_root),
            "features": _feature_summary(info),
            "core_feature_statistics": _core_feature_statistics(dataset_root, info),
            "state_action_alignment": _state_action_alignment(dataset_root, info),
        },
        "candidates_for_review": candidates_for_review,
        "episodes": episode_results,
    }


def write_analysis_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_analysis_markdown(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_analysis_markdown(report), encoding="utf-8")
    return output


def render_analysis_markdown(report: dict[str, Any]) -> str:
    """Render the machine-readable analysis as a compact human review report."""
    dataset = report["dataset"]
    summary = report["summary"]
    duration = summary["duration_seconds"]
    lines = [
        "# HandUMI Dataset Analysis",
        "",
        f"- Dataset: `{dataset['repo_id']}`",
        f"- Episodes: **{dataset['total_episodes']}**",
        f"- Frames: **{dataset['total_frames']}**",
        f"- FPS: **{dataset['fps']:g}**",
        (
            f"- Duration mean/min/max: **{duration['mean']:.2f}s / "
            f"{duration['min']:.2f}s / {duration['max']:.2f}s**"
        ),
        f"- Accepted: **{summary['accepted']}**",
        f"- Review: **{summary['outliers_for_review']}**",
        f"- Rejected by quality checks: **{summary['quality_rejected']}**",
        "",
        "## Episode duration histogram",
        "",
        "| From (s) | To (s) | Episodes |",
        "|---:|---:|---:|",
    ]
    for item in summary["histogram"]:
        lines.append(f"| {item['from_s']:.1f} | {item['to_s']:.1f} | {item['count']} |")
    shortest = sorted(report["episodes"], key=lambda item: item["duration_s"])[:5]
    longest = sorted(
        report["episodes"], key=lambda item: item["duration_s"], reverse=True
    )[:5]
    lines.extend(
        [
            "",
            "## Duration extremes",
            "",
            "| Tail | Source episode | Duration (s) | Status |",
            "|---|---:|---:|---|",
        ]
    )
    for tail, items in (("shortest", shortest), ("longest", longest)):
        for item in items:
            lines.append(
                f"| {tail} | {item['source_episode_index']} | "
                f"{item['duration_s']:.2f} | {item['status']} |"
            )
    lines.extend(
        [
            "",
            "## Episodes requiring attention",
            "",
            "| Source episode | Duration (s) | Status | Findings |",
            "|---:|---:|---|---|",
        ]
    )
    attention = [item for item in report["episodes"] if item["status"] != "accepted"]
    if attention:
        for item in attention:
            codes = ", ".join(finding["code"] for finding in item["findings"])
            lines.append(
                f"| {item['source_episode_index']} | {item['duration_s']:.2f} | "
                f"{item['status']} | {codes} |"
            )
    else:
        lines.append("| — | — | — | No findings |")
    lines.extend(
        [
            "",
            "## Human confirmation required",
            "",
            f"- Automatic review candidates: `{report['candidates_for_review']}`",
            "",
            (
                "The report never removes episodes automatically. Review the histogram, "
                "episode findings, and duration extremes, then pass the confirmed source "
                "indices explicitly to `handumi dataset curate --exclude`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def load_analysis_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read analysis report {report_path}: {exc}") from exc
    if report.get("kind") != ANALYSIS_KIND:
        raise ValueError(f"Unsupported analysis report kind: {report.get('kind')!r}")
    if report.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported analysis schema version: {report.get('schema_version')!r}"
        )
    episodes = report.get("episodes")
    candidates = report.get("candidates_for_review")
    if not isinstance(episodes, list) or not isinstance(candidates, list):
        raise TypeError("Analysis report is missing episodes or candidates_for_review")
    return report


def discover_quality_reports(root: str | Path) -> list[Path]:
    """Find every findings report a dataset carries, in a stable order.

    Each report grades one dimension -- recording quality, retargeting for one
    embodiment -- and they are independent, so an analysis has to see all of
    them. Reading only one silently drops whole categories of defect from the
    review a human then curates from.
    """
    meta = Path(root) / "meta"
    if not meta.is_dir():
        return []
    found = [meta / "handumi_quality.json"]
    found.extend(sorted(meta.glob("handumi_screening_*.json")))
    return [path for path in found if path.is_file()]


def _load_quality_findings(
    root: Path,
    *,
    quality_report: str | Path | None = None,
    quality_reports: Sequence[str | Path] | None = None,
) -> tuple[dict[int, tuple[dict[str, Any], ...]], list[Path]]:
    if quality_report is not None and quality_reports is not None:
        raise ValueError("Use only one of quality_report or quality_reports.")
    if quality_report is not None:
        requested: list[Path] | None = [Path(quality_report)]
    elif quality_reports is not None:
        requested = [Path(item) for item in quality_reports]
    else:
        requested = None

    paths = discover_quality_reports(root) if requested is None else requested
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Quality report does not exist: {path}")

    merged: dict[int, list[dict[str, Any]]] = {}
    resolved: list[Path] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read quality report {path}: {exc}") from exc
        resolved.append(path)
        # Name the producer so a merged review says which dimension objected.
        source = path.stem.removeprefix("handumi_") or path.stem
        for item in payload.get("episodes", []):
            if not isinstance(item, dict) or "episode_index" not in item:
                continue
            findings: list[dict[str, Any]] = []
            for finding in item.get("findings", []):
                if not isinstance(finding, dict):
                    continue
                copied = dict(finding)
                copied["source"] = source
                findings.append(copied)
            if item.get("status") == "rejected" and not any(
                finding.get("severity") == "reject" for finding in findings
            ):
                findings.append(
                    {
                        "code": "quality_report_rejection",
                        "severity": "reject",
                        "message": f"Episode was rejected by {source}.",
                        "metrics": {},
                        "source": source,
                    }
                )
            merged.setdefault(int(item["episode_index"]), []).extend(findings)
    return {index: tuple(items) for index, items in merged.items()}, resolved


def dataset_payload_manifest(root: str | Path) -> dict[str, Any]:
    """Hash data-bearing files while excluding mutable audit sidecars."""
    dataset_root = Path(root)
    paths: set[Path] = set()
    for directory in ("data", "videos", "audio"):
        candidate = dataset_root / directory
        if candidate.is_dir():
            paths.update(path for path in candidate.rglob("*") if path.is_file())
    for relative in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet"):
        candidate = dataset_root / relative
        if candidate.is_file():
            paths.add(candidate)
    episodes = dataset_root / "meta" / "episodes"
    if episodes.is_dir():
        paths.update(path for path in episodes.rglob("*") if path.is_file())

    records: list[tuple[str, int, str]] = []
    total_bytes = 0
    for path in sorted(paths):
        relative = path.relative_to(dataset_root).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        size = path.stat().st_size
        total_bytes += size
        records.append((relative, size, digest.hexdigest()))
    manifest = hashlib.sha256()
    for relative, size, digest in records:
        manifest.update(f"{relative}\0{size}\0{digest}\n".encode())
    return {
        "files": len(records),
        "bytes": total_bytes,
        "sha256": manifest.hexdigest(),
    }


def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (
            str(finding.get("code", "unknown")),
            str(finding.get("severity", "warning")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def _duration_histogram(
    durations: np.ndarray, bin_width_s: float
) -> list[dict[str, Any]]:
    upper = math.ceil(float(durations.max()) / bin_width_s) * bin_width_s
    if upper <= float(durations.max()):
        upper += bin_width_s
    upper = max(upper, bin_width_s)
    edges = np.arange(0.0, upper + bin_width_s * 0.5, bin_width_s)
    counts, resolved_edges = np.histogram(durations, bins=edges)
    return [
        {
            "from_s": float(start),
            "to_s": float(end),
            "count": int(count),
        }
        for start, end, count in zip(
            resolved_edges[:-1], resolved_edges[1:], counts, strict=True
        )
    ]


def _storage_bytes(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    total = 0
    for name in ("data", "videos", "audio", "meta"):
        directory = root / name
        size = (
            sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            if directory.is_dir()
            else 0
        )
        result[name] = size
        total += size
    result["total"] = total
    return result


def _feature_summary(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {})
    video = [key for key, value in features.items() if value.get("dtype") == "video"]
    image = [key for key, value in features.items() if value.get("dtype") == "image"]
    return {
        "total": len(features),
        "video": video,
        "image": image,
        "observation_state_shape": features.get("observation.state", {}).get("shape"),
        "action_shape": features.get("action", {}).get("shape"),
    }


def _core_feature_statistics(root: Path, info: dict[str, Any]) -> dict[str, Any]:
    path = root / "meta" / "stats.json"
    if not path.is_file():
        return {}
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    result = {}
    for key in ("observation.state", "action"):
        if key in info.get("features", {}) and key in stats:
            result[key] = stats[key]
    return result


def _state_action_alignment(root: Path, info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {})
    state = features.get("observation.state")
    action = features.get("action")
    if not isinstance(state, dict) or not isinstance(action, dict):
        return {"comparable": False, "reason": "state or action feature missing"}
    if state.get("shape") != action.get("shape"):
        return {"comparable": False, "reason": "state and action shapes differ"}
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        return {"comparable": False, "reason": "data Parquet files missing"}
    rows = 0
    finite_rows = 0
    exact_rows = 0
    max_difference = 0.0
    finite_difference_found = False
    for path in data_files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        rows += len(states)
        finite = np.isfinite(states).all(axis=1) & np.isfinite(actions).all(axis=1)
        finite_rows += int(finite.sum())
        exact_rows += int(np.all(states == actions, axis=1).sum())
        finite_values = np.isfinite(states) & np.isfinite(actions)
        if np.any(finite_values):
            finite_difference_found = True
            max_difference = max(
                max_difference,
                float(np.max(np.abs(states[finite_values] - actions[finite_values]))),
            )
    return {
        "comparable": True,
        "rows": rows,
        "finite_rows": finite_rows,
        "exact_equal_rows": exact_rows,
        "exact_equal_fraction": exact_rows / rows if rows else 0.0,
        "max_abs_difference": max_difference if finite_difference_found else None,
    }
