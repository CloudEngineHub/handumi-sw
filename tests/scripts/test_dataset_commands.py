from pathlib import Path

from handumi.scripts.analyze_dataset import build_parser as analysis_parser
from handumi.scripts.curate_dataset import build_parser as curation_parser


def test_analysis_cli_is_automatic() -> None:
    args = analysis_parser().parse_args(["outputs/demo", "--dry-run"])
    assert args.dataset == "outputs/demo"
    assert args.dry_run


def test_curation_cli_requires_explicit_output() -> None:
    args = curation_parser().parse_args(
        [
            "outputs/demo",
            "--output",
            "outputs/demo_clean",
            "--analysis",
            "analysis.json",
            "--exclude",
            "6,75",
        ]
    )
    assert args.output == Path("outputs/demo_clean")
    assert args.analysis == Path("analysis.json")
    assert args.exclude == "6,75"
