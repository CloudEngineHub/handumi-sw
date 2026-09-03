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


def test_joint_dataset_names_state_their_robot_and_kind() -> None:
    """A joint dataset is specific to one embodiment, so its name must say so.

    Otherwise the same capture converted for two robots yields names that
    differ only by a suffix nobody can interpret, and neither is
    distinguishable from the robot-agnostic source.
    """
    from handumi.scripts.conversion import _default_output_repo_id

    assert (
        _default_output_repo_id("local/handumi-demo-clean", "piper")
        == "local/handumi-demo-clean-piper-joints"
    )
    assert (
        _default_output_repo_id("local/handumi-demo-clean", "metal")
        == "local/handumi-demo-clean-metal-joints"
    )
    # A bare name keeps its namespace absent rather than inventing one.
    assert _default_output_repo_id("handumi-demo", "yam") == "handumi-demo-yam-joints"


def test_output_layout_names_the_plugin_vector() -> None:
    from handumi.dataset.external_layouts import BI_PIPER_FOLLOWER
    from handumi.scripts.conversion import _resolve_conversion_output
    from handumi.scripts.export_dataset import default_output_name

    repo_id, root = _resolve_conversion_output(
        None, source_repo_id="local/handumi-demo-clean", embodiment="piper",
        layout_suffix="bi_piper_follower",
    )
    assert repo_id == "local/handumi-demo-clean-bi_piper_follower"
    assert root == Path("outputs/datasets/handumi-demo-clean-bi_piper_follower")
    # Exporting an existing canonical dataset lands on the same name.
    assert default_output_name("handumi-demo-clean-piper-joints", BI_PIPER_FOLLOWER) == (
        "handumi-demo-clean-bi_piper_follower"
    )
    assert default_output_name("handumi-demo-clean", BI_PIPER_FOLLOWER) == (
        "handumi-demo-clean-bi_piper_follower"
    )


def test_output_layout_must_describe_the_chosen_robot() -> None:
    import pytest

    from handumi.scripts.conversion import _resolve_output_layout, build_parser

    parser = build_parser()
    args = parser.parse_args(["raw", "--robot", "piper", "--output-layout", "bi_openarm_follower"])
    with pytest.raises(SystemExit):
        _resolve_output_layout(parser, args)
    args = parser.parse_args(["raw", "--robot", "piper", "--output-layout", "bi_piper_follower", "--use-degrees"])
    with pytest.raises(SystemExit):  # XHUMAN's Piper plugin has no use_degrees
        _resolve_output_layout(parser, args)
    args = parser.parse_args(["raw", "--robot", "piper", "--output-layout", "bi_piper_follower"])
    assert _resolve_output_layout(parser, args).robot_type == "bi_piper_follower"
    args = parser.parse_args(["raw", "--robot", "piper"])
    assert _resolve_output_layout(parser, args) is None
