"""Create a new local dataset from an analysis report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.dataset.curation import curate_dataset, plan_dataset_curation
from handumi.dataset.reader import ensure_metadata
from handumi.dataset.selection import resolve_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a validated local dataset from human-confirmed exclusions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Local dataset path or Hugging Face repo id.")
    parser.add_argument(
        "--output", type=Path, required=True, help="New local dataset root."
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=None,
        help="Defaults to DATASET/meta/handumi_analysis.json.",
    )
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument(
        "--exclude",
        required=True,
        help="Human-confirmed, comma-separated source episode indices to remove.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the curation plan without writing output.",
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
        analysis_path = (
            args.analysis or selection.root / "meta" / "handumi_analysis.json"
        )
        plan = plan_dataset_curation(
            selection.root,
            analysis_path=analysis_path,
            output_root=args.output,
            source_repo_id=selection.repo_id,
            output_repo_id=args.output_repo_id,
            exclude_episode_indices=_parse_indices(args.exclude),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "Dataset curation plan\n"
        f"  Source: {plan.source_root}\n"
        f"  Analysis: {plan.analysis_path}\n"
        f"  Output: {plan.output_root}\n"
        f"  Remove source episodes: {list(plan.excluded_source_episode_indices)}\n"
        f"  Keep: {plan.output_total_episodes}/{plan.source_total_episodes}\n"
        "  Publish to Hub: no"
    )
    if args.dry_run:
        print("Output: not written (--dry-run)")
        return
    try:
        result = curate_dataset(plan)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Curated dataset complete\n"
        f"  Root: {result.root}\n"
        f"  Episodes: {result.total_episodes}\n"
        f"  Frames: {result.total_frames}\n"
        f"  Report: {result.report_path}\n"
        "  Published to Hub: no"
    )


def _parse_indices(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid episode index list: {value!r}") from exc


if __name__ == "__main__":
    main()
