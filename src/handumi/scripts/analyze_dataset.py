"""Analyze a LeRobot dataset and write an auditable episode report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.dataset.analysis import (
    analyze_dataset,
    write_analysis_markdown,
    write_analysis_report,
)
from handumi.dataset.reader import ensure_metadata
from handumi.dataset.selection import resolve_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics and identify episodes for human review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Local dataset path or Hugging Face repo id.")
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument(
        "--quality-report",
        type=Path,
        action="append",
        default=None,
        dest="quality_reports",
        help=(
            "Findings report to merge; repeatable. Defaults to every "
            "meta/handumi_quality.json and meta/handumi_screening_*.json the "
            "dataset carries, so no dimension is silently left out."
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Human-readable report; defaults beside the JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the result without writing a report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args(sys.argv[1:])
    try:
        selection = resolve_dataset_selection(args.dataset, revision=args.revision)
        ensure_metadata(
            repo_id=selection.repo_id,
            root=selection.root,
            revision=selection.revision,
        )
        report = analyze_dataset(
            selection.root,
            repo_id=selection.repo_id,
            quality_reports=args.quality_reports,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    summary = report["summary"]
    seconds = summary["duration_seconds"]
    print(
        "Dataset analysis\n"
        f"  Dataset: {selection.root}\n"
        f"  Episodes: {report['dataset']['total_episodes']}\n"
        f"  Frames: {report['dataset']['total_frames']}\n"
        f"  Duration mean/min/max: {seconds['mean']:.2f}s / "
        f"{seconds['min']:.2f}s / {seconds['max']:.2f}s\n"
        f"  Accepted: {summary['accepted']}\n"
        f"  Review: {summary['outliers_for_review']}\n"
        f"  Automatic review candidates: {report['candidates_for_review'] or 'none'}\n"
        "  Automatic removal: no"
    )
    if args.dry_run:
        print("Report: not written (--dry-run)")
        return
    report_path = args.report or selection.root / "meta" / "handumi_analysis.json"
    write_analysis_report(report_path, report)
    markdown_path = args.markdown or report_path.with_suffix(".md")
    write_analysis_markdown(markdown_path, report)
    print(f"Report: {report_path}\nReview: {markdown_path}")


if __name__ == "__main__":
    main()
