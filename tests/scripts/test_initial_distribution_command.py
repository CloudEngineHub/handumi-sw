from pathlib import Path

from handumi.scripts.initial_distribution import build_parser, default_output_path


def test_initial_distribution_command_is_readable_and_explicit() -> None:
    args = build_parser().parse_args(
        [
            "outputs/datasets/demo-piper-joints",
            "--output",
            "outputs/initial-distributions/demo-piper.yaml",
        ]
    )
    assert args.dataset == "outputs/datasets/demo-piper-joints"
    assert args.output == Path("outputs/initial-distributions/demo-piper.yaml")
    assert not args.dry_run


def test_initial_distribution_dry_run_needs_no_output() -> None:
    args = build_parser().parse_args(
        ["outputs/datasets/demo-piper-joints", "--dry-run"]
    )
    assert args.output is None
    assert args.dry_run


def test_initial_distribution_has_a_clear_default_output() -> None:
    assert default_output_path(Path("outputs/datasets/demo-piper-joints")) == Path(
        "outputs/initial-distributions/demo-piper-joints-initial-distribution.yaml"
    )
