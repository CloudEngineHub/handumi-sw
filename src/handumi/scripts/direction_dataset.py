"""Flag episodes that demonstrate the task backwards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.dataset.direction import (
    DEFAULT_VIDEO_KEY,
    analyze_dataset_direction,
)
from handumi.dataset.quality import EpisodeQualityConfig, write_quality_report
from handumi.dataset.reader import ensure_metadata
from handumi.dataset.selection import resolve_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grade every episode's scene change against the dataset's own median "
            "change, and flag the ones running the other way. A reset take "
            "recorded as an episode passes every other check: it is a clean "
            "recording of the task being undone."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Local dataset path or Hugging Face repo id.")
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument(
        "--video-key",
        default=DEFAULT_VIDEO_KEY,
        help="Camera that sees the workspace. Wrist cameras move with the tool "
        "and cannot report what changed on the table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to DATASET/meta/handumi_direction.json.",
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
        reports, summary = analyze_dataset_direction(
            selection.root, video_key=args.video_key
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    output = args.output or selection.root / "meta" / "handumi_direction.json"
    write_quality_report(
        output,
        reports,
        config=EpisodeQualityConfig(),
        dataset=selection.repo_id,
    )

    flagged = summary["reversed_episode_indices"]
    forward = [r for r in reports if not r.findings]
    if forward:
        lowest = min(
            float(r.metrics["direction_similarity"]) for r in forward
        )
        print(
            f"[direction] {len(reports)} episodes | lowest forward similarity "
            f"{lowest:+.3f}"
        )
    if flagged:
        highest = max(
            float(r.metrics["direction_similarity"])
            for r in reports
            if r.findings
        )
        print(
            f"[direction] running backwards: {flagged}\n"
            f"[direction] highest flagged similarity {highest:+.3f}; watch these "
            "before curating -- they read as reset takes, not demonstrations"
        )
    else:
        print("[direction] no episode runs against the dataset.")
    print(f"[direction] report: {output}")


if __name__ == "__main__":
    main()
