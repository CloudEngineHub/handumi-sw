"""Load a per-embodiment inpainting prompt.

Prompts live in ``configs/inpainting-prompts/<robot>.md`` so each embodiment
carries its own and can be refined in place. The file is the prompt: it is sent
verbatim. A file that wants editorial notes alongside it can put them above a
``---`` rule or inside HTML comments, and only the instruction is sent.
"""

from __future__ import annotations

import re
from pathlib import Path

_PACKAGE_PROMPTS = Path(__file__).resolve().parent.parent / "configs" / "inpainting-prompts"
# Prefer the working copy so a prompt can be edited and re-run without a
# reinstall; fall back to the packaged one for an installed handumi.
PROMPTS_DIR = (
    Path("configs/inpainting-prompts")
    if Path("configs/inpainting-prompts").exists()
    else _PACKAGE_PROMPTS
)

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def prompt_path(robot: str, directory: Path | None = None) -> Path:
    return (directory or PROMPTS_DIR) / f"{robot}.md"


def strip_markdown(text: str) -> str:
    """Return the instruction body: the text below the first ``---`` rule.

    Without a rule the whole file is the instruction, minus headings and HTML
    comments, so a bare prompt file still works.
    """
    without_comments = _COMMENT.sub("", text)
    lines = without_comments.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "---":
            body = "\n".join(lines[index + 1:])
            break
    else:
        body = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return re.sub(r"\n{3,}", "\n\n", body.strip())


def load_prompt(robot: str, directory: Path | None = None) -> str:
    path = prompt_path(robot, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"No inpainting prompt for {robot!r}: {path}. Add one before inpainting this embodiment."
        )
    prompt = strip_markdown(path.read_text(encoding="utf-8"))
    if not prompt:
        raise ValueError(f"{path} has no instruction body outside its headings and comments.")
    return prompt
