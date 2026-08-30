"""Join separately recorded sessions into one local dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.dataset.merging import (
    merge_dataset,
    merge_summary_lines,
    plan_dataset_merge,
)
from handumi.dataset.reader import ensure_metadata
from handumi.dataset.selection import resolve_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join two or more recording sessions into one local dataset. Every "
            "episode survives, in the order the sources are given; only the "
            "episode numbering and the task table change."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Two or more local dataset paths or Hugging Face repo ids, in the "
        "order they should appear in the output.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New local dataset root."
    )
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument(
        "--task",
        default=None,
        help="Rewrite every episode to this one task. Sessions that recorded the "
        "same task under different wordings otherwise merge into several tasks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merge plan without writing output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args(sys.argv[1:])
    try:
        selections = [
            resolve_dataset_selection(dataset, revision=args.revision)
            for dataset in args.datasets
        ]
        for selection in selections:
            ensure_metadata(
                repo_id=selection.repo_id,
                root=selection.root,
                revision=selection.revision,
            )
        plan = plan_dataset_merge(
            [selection.root for selection in selections],
            output_root=args.output,
            source_repo_ids=[selection.repo_id for selection in selections],
            output_repo_id=args.output_repo_id,
            task=args.task,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print("\n".join(merge_summary_lines(plan)))
    print("  Publish to Hub: no")
    if args.dry_run:
        print("Output: not written (--dry-run)")
        return
    try:
        result = merge_dataset(plan)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Merged dataset complete\n"
        f"  Root: {result.root}\n"
        f"  Episodes: {result.total_episodes}\n"
        f"  Frames: {result.total_frames}\n"
        f"  Tasks: {result.total_tasks}\n"
        f"  Report: {result.report_path}\n"
        "  Published to Hub: no"
    )


if __name__ == "__main__":
    main()
