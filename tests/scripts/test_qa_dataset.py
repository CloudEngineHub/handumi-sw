"""The review sequence a conversion depends on."""

from __future__ import annotations

from pathlib import Path

from handumi.scripts.qa_dataset import build_parser, review_steps


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def test_every_dimension_is_reviewed_once_per_target() -> None:
    args = _args(["ds", "--robot", "piper", "--robot", "metal"])
    steps = review_steps(args, "ds")

    names = [name for name, _ in steps]
    assert names == [
        "recording quality",
        "demonstration direction",
        "retargeting: piper",
        "retargeting: metal",
        "merged analysis",
    ]
    # Recording quality and direction are embodiment-agnostic and run once each;
    # the analysis is last because it merges whatever the earlier steps wrote.
    assert names.count("recording quality") == 1
    assert names.count("demonstration direction") == 1
    assert names[-1] == "merged analysis"


def test_screening_is_skipped_without_a_target_robot() -> None:
    """A dataset can be reviewed as a recording before any robot is chosen."""
    steps = review_steps(_args(["ds"]), "ds")
    assert [name for name, _ in steps] == [
        "recording quality",
        "demonstration direction",
        "merged analysis",
    ]


def test_screening_thresholds_reach_the_screen_command() -> None:
    args = _args(["ds", "--robot", "piper", "--max-position-error-m", "0.02"])
    screen = next(cmd for name, cmd in review_steps(args, "ds") if "screen" in cmd)
    assert "--max-position-error-m" in screen
    assert screen[screen.index("--max-position-error-m") + 1] == "0.02"


def test_base_rotation_limit_reaches_the_screen_command() -> None:
    args = _args(["ds", "--robot", "piper", "--max-base-rotation-deg", "60"])
    screen = next(cmd for name, cmd in review_steps(args, "ds") if "screen" in cmd)
    assert screen[screen.index("--max-base-rotation-deg") + 1] == "60.0"
    without = _args(["ds", "--robot", "piper"])
    screen = next(cmd for name, cmd in review_steps(without, "ds") if "screen" in cmd)
    assert "--max-base-rotation-deg" not in screen


def test_validation_can_reuse_an_existing_report() -> None:
    args = _args(["ds", "--robot", "piper", "--skip-validate"])
    assert "recording quality" not in [name for name, _ in review_steps(args, "ds")]


def test_rig_config_is_passed_through(tmp_path: Path) -> None:
    """Screening must resolve the same table placement conversion will use."""
    rig = tmp_path / "rig.yaml"
    args = _args(["ds", "--robot", "piper", "--rig-config", str(rig)])
    screen = next(cmd for name, cmd in review_steps(args, "ds") if "screen" in cmd)
    assert str(rig) in screen


def test_direction_can_be_left_out() -> None:
    """It needs a workspace camera and a majority of episodes to compare against."""
    args = _args(["ds", "--robot", "piper", "--skip-direction"])
    assert "demonstration direction" not in [
        name for name, _ in review_steps(args, "ds")
    ]
