#!/usr/bin/env python3
"""Run every dataset review a conversion depends on, in one pass.

The reviews stay separate commands because they answer separate questions and
are useful on their own: `handumi validate` grades the recording and is
embodiment-agnostic, `handumi dataset screen` grades how one robot retargets
it. This command only sequences them and merges the result, so an operator
cannot leave a dimension unexamined by forgetting a step.

Nothing here removes data. It produces the merged review that a human curates
from, which keeps the decision in one place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.analysis import discover_quality_reports
from handumi.dataset.selection import resolve_dataset_selection
from handumi.robots.registry import EMBODIMENT_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review a dataset end to end: recording quality, retargeting for "
            "each target robot, then one merged analysis to curate from."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Local dataset path or Hugging Face repo id.")
    parser.add_argument(
        "--robot",
        choices=EMBODIMENT_NAMES,
        action="append",
        default=None,
        help="Target robot to screen for; repeatable. Omit to skip screening.",
    )
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--deployment-profile",
        choices=("auto", "local", "sim"),
        default="auto",
    )
    parser.add_argument(
        "--max-position-error-m",
        type=float,
        default=None,
        help="Retargeting position ceiling passed through to screening.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Reuse the existing recording-quality report instead of recomputing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the review plan without running it.",
    )
    return parser


# Invoke through the interpreter rather than the `handumi` console script: the
# entry point is only on PATH when the environment is activated, and a review
# command must not fail for that reason.
_CLI = [sys.executable, "-m", "handumi.scripts.cli"]


def _run(step: str, command: list[str]) -> None:
    print(f"\n=== {step} ===\n$ handumi {' '.join(command)}", flush=True)
    result = subprocess.run(_CLI + command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"[qa] {step} failed with exit code {result.returncode}.")


def main() -> None:
    args = build_parser().parse_args()
    try:
        selection = resolve_dataset_selection(args.dataset, revision=args.revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root = str(selection.root)
    robots = args.robot or []
    steps: list[tuple[str, list[str]]] = []
    if not args.skip_validate:
        steps.append(("recording quality", ["validate", root]))
    for robot in robots:
        command = [
            "dataset", "screen", root,
            "--robot", robot,
            "--deployment-profile", args.deployment_profile,
            "--rig-config", str(args.rig_config),
        ]
        if args.max_position_error_m is not None:
            command += ["--max-position-error-m", str(args.max_position_error_m)]
        steps.append((f"retargeting: {robot}", command))
    steps.append(("merged analysis", ["dataset", "analyze", root]))

    print(
        "Review plan\n"
        f"  Dataset: {root}\n"
        f"  Robots: {', '.join(robots) if robots else 'none (recording only)'}\n"
        f"  Steps: {len(steps)}"
    )
    if args.dry_run:
        for step, command in steps:
            print(f"  - {step}: handumi {' '.join(command)}")
        return

    for step, command in steps:
        _run(step, command)

    reports = discover_quality_reports(selection.root)
    print("\n[qa] findings reports merged into the analysis:")
    for path in reports:
        print(f"  - {path}")

    analysis_path = selection.root / "meta" / "handumi_analysis.json"
    rejected: list[int] = []
    review: list[int] = []
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        for item in analysis.get("episodes", []):
            index = int(item["source_episode_index"])
            if item.get("status") == "rejected":
                rejected.append(index)
            elif item.get("status") != "accepted":
                review.append(index)

    if rejected:
        print(f"[qa] rejected, removable without review: {rejected}")
    if review:
        print(f"[qa] flagged for your judgement: {review}")
        print("     read meta/handumi_analysis.md before deciding on these")
    if not rejected and not review:
        print("[qa] no episode was flagged.")
        return
    command = f"  handumi dataset curate {root} --output <new_root>"
    if rejected:
        command += " --exclude-rejected"
    if review:
        command += " --exclude <the ones you reject>"
    print(f"[qa] curate with:\n{command}")


if __name__ == "__main__":
    sys.exit(main())
