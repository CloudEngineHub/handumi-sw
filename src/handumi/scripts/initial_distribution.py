#!/usr/bin/env python3
"""Extract the initial joint distribution from a converted dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.dataset.initial_distribution import (
    initial_distribution,
    write_initial_distribution,
)
from handumi.dataset.selection import DatasetSelection, resolve_dataset_selection

DEFAULT_OUTPUT_DIR = Path("outputs/initial-distributions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one initial joint state per episode from an already "
            "converted robot dataset and write its distribution as YAML."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Local path or Hugging Face repo id of a converted joints dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "External YAML file to create. Defaults to "
            "outputs/initial-distributions/<dataset>-initial-distribution.yaml."
        ),
    )
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the output YAML when it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the summary without writing a YAML file.",
    )
    return parser


def default_output_path(dataset_root: Path) -> Path:
    """Return the external profile path without modifying the input dataset."""
    return DEFAULT_OUTPUT_DIR / f"{dataset_root.name}-initial-distribution.yaml"


def ensure_joint_data(selection: DatasetSelection) -> None:
    """Fetch only metadata and Parquet joints; this analysis never needs video."""
    info = selection.root / "meta" / "info.json"
    data_files = list((selection.root / "data").glob("chunk-*/*.parquet"))
    if selection.local:
        if not info.is_file() or not data_files:
            raise ValueError(
                f"Local dataset {selection.root} is missing meta/info.json or data Parquet files"
            )
        return

    from huggingface_hub import snapshot_download

    selection.root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=selection.repo_id,
        repo_type="dataset",
        revision=selection.revision,
        local_dir=selection.root,
        allow_patterns=["meta/**", "data/**"],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])

    selection = resolve_dataset_selection(args.dataset, revision=args.revision)
    output = args.output or default_output_path(selection.root)
    if not args.dry_run and output.exists() and not args.replace:
        parser.error(f"output already exists: {output}; pass --replace to overwrite it.")
    try:
        ensure_joint_data(selection)
        report = initial_distribution(selection.root, repo_id=selection.repo_id)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    nearest = report["recommended"]["nearest_to_mean"]
    print(
        "Initial joint distribution\n"
        f"  Dataset: {selection.root}\n"
        f"  Robot: {report['source']['robot']}\n"
        f"  Episodes: {report['source']['episode_count']}\n"
        "  Sampling: first observation per episode\n"
        "  Recommended: nearest_to_mean "
        f"(episode {nearest['episode_index']})"
    )
    if args.dry_run:
        print(f"Output: {output} (not written: --dry-run)")
        return
    output = write_initial_distribution(output, report)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
