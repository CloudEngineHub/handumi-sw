from __future__ import annotations

import pytest

from handumi.inpainting import load_prompt, prompt_path
from handumi.inpainting.prompt import strip_markdown


def test_editorial_above_the_rule_never_reaches_the_model(tmp_path):
    (tmp_path / "piper.md").write_text(
        "# Title\n\nNotes for the maintainer.\n\n<!-- iteration history -->\n\n"
        "---\n\nReplace the human arm with the robot arm.\n",
        encoding="utf-8",
    )
    prompt = load_prompt("piper", tmp_path)
    assert prompt == "Replace the human arm with the robot arm."


def test_html_comments_are_dropped_even_below_the_rule():
    text = "---\nDo the thing.\n<!-- but not this -->\nAnd this.\n"
    assert strip_markdown(text) == "Do the thing.\n\nAnd this."


def test_a_file_without_a_rule_is_all_prompt():
    assert strip_markdown("# Heading\nDo the thing.") == "Do the thing."


def test_missing_prompt_names_the_embodiment(tmp_path):
    with pytest.raises(FileNotFoundError, match="openarmv1"):
        load_prompt("openarmv1", tmp_path)


def test_a_prompt_with_no_body_is_refused(tmp_path):
    (tmp_path / "piper.md").write_text("# Only a heading\n\n<!-- and a note -->\n---\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no instruction body"):
        load_prompt("piper", tmp_path)


def test_prompt_path_is_per_embodiment(tmp_path):
    assert prompt_path("piper", tmp_path).name == "piper.md"
    assert prompt_path("openarmv1", tmp_path).name == "openarmv1.md"


def test_the_shipped_piper_prompt_asks_for_one_arm():
    """A HandUMI episode is performed by one arm; two-arm phrasing confused the edit."""
    prompt = load_prompt("piper")
    assert "ONE robot arm" in prompt
    assert "second arm" in prompt
    assert "Keep everything else" in prompt


def test_the_shipped_piper_prompt_does_not_hard_code_a_side():
    """The operator enters from a different edge in different episodes."""
    prompt = load_prompt("piper").lower()
    for side in ("upper left", "upper right", "on the left", "on the right"):
        assert side not in prompt, f"{side!r} would only hold for some episodes"
    assert "from any edge" in prompt
    assert "same edge of the frame the forearm entered" in prompt


def test_the_shipped_prompt_does_not_name_one_task():
    """The same prompt has to serve any HandUMI recording, not one dataset."""
    prompt = load_prompt("piper").lower()
    for word in ("t-block", "t-shaped", "coloured block", "colored block",
                 "blue plate", "red block", "white cloth", "puzzle", "screw"):
        assert word not in prompt, f"{word!r} only holds for one task"


def test_a_second_embodiment_works_from_its_own_file(tmp_path):
    """Adding an arm is adding a file: no code change, no shared prompt."""
    (tmp_path / "openarmv1.md").write_text(
        "Replace the operator with an OpenArm.\n", encoding="utf-8"
    )
    assert load_prompt("openarmv1", tmp_path) == "Replace the operator with an OpenArm."
    with pytest.raises(FileNotFoundError, match="piper"):
        load_prompt("piper", tmp_path)
