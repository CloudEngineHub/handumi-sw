#!/usr/bin/env python3
"""Grade a dataset's episodes by how well one embodiment can retarget them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.screening import (
    RetargetScreeningConfig,
    render_screening_markdown,
    screen_dataset,
    screening_report_path,
    write_screening_report,
)
from handumi.dataset.selection import resolve_dataset_selection
from handumi.robots.registry import EMBODIMENT_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget every episode through the conversion solver and grade the "
            "result, so unusable episodes are removed before joint conversion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Local dataset path or Hugging Face repo id.")
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, required=True)
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices (default: every episode).",
    )
    parser.add_argument(
        "--deployment-profile",
        choices=("auto", "local", "sim"),
        default="auto",
        help="Table placement source, matching replay and convert.",
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Processes solving episodes at once. Episodes are independent, so "
        "this changes wall-clock only. The default leaves room for the threads "
        "the solver already uses inside one episode; 1 solves them in order.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to DATASET/meta/handumi_screening_<robot>.json.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Human-readable report; defaults beside the JSON report.",
    )
    parser.add_argument(
        "--max-position-error-m",
        type=float,
        default=RetargetScreeningConfig.max_position_error_m,
    )
    parser.add_argument(
        "--max-rotation-error-deg",
        type=float,
        default=RetargetScreeningConfig.max_rotation_error_deg,
    )
    parser.add_argument(
        "--max-base-rotation-deg",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Reject episodes whose arm base (first joint) swings farther than "
            "this from home. Off by default; 60 matches what XHUMAN's real "
            "Piper teleoperation stays under."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the screening plan without solving episodes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        selection = resolve_dataset_selection(args.dataset, revision=args.revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    episodes = None
    if args.episodes:
        try:
            episodes = [int(part) for part in args.episodes.split(",") if part.strip()]
        except ValueError as exc:
            raise SystemExit(f"Invalid --episodes list: {exc}") from exc

    report_path = args.report or screening_report_path(selection.root, args.robot)
    print(
        "Screening plan\n"
        f"  Dataset: {selection.root}\n"
        f"  Robot: {args.robot}\n"
        f"  Episodes: {'all' if episodes is None else episodes}\n"
        f"  Report: {report_path}"
    )
    if args.dry_run:
        return

    config = RetargetScreeningConfig(
        max_position_error_m=args.max_position_error_m,
        max_rotation_error_deg=args.max_rotation_error_deg,
        max_base_rotation_deg=args.max_base_rotation_deg,
    )
    payload = screen_dataset(
        selection.root,
        robot=args.robot,
        repo_id=selection.repo_id,
        revision=args.revision,
        episodes=episodes,
        deployment_profile=args.deployment_profile,
        rig_config=args.rig_config,
        jobs=args.jobs,
        config=config,
    )
    write_screening_report(report_path, payload)
    markdown_path = args.markdown or report_path.with_suffix(".md")
    markdown_path.write_text(render_screening_markdown(payload), encoding="utf-8")

    summary = payload["summary"]
    print(
        f"\n[screen] {summary['total']} episodes: {summary['accepted']} accepted, "
        f"{summary['rejected']} rejected"
    )
    print(f"[screen] report: {report_path}")
    print(f"[screen] markdown: {markdown_path}")
    if summary["review_episode_indices"]:
        print(
            "[screen] warnings needing your review: "
            + ",".join(str(i) for i in summary["review_episode_indices"])
        )
    if summary["reject_episode_indices"]:
        excluded = ",".join(str(i) for i in summary["reject_episode_indices"])
        print(
            f"[screen] rejected: {excluded}\n"
            "[screen] to remove them, feed this report to analyze and curate:\n"
            f"  handumi dataset analyze {selection.root} "
            f"--quality-report {report_path}\n"
            f"  handumi dataset curate {selection.root} "
            f"--output <new_root> --exclude {excluded}"
        )
    else:
        print("[screen] no episode is unusable for this embodiment.")


if __name__ == "__main__":
    sys.exit(main())
