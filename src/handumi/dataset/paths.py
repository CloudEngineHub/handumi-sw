"""Write paths into reports in a form that survives leaving this machine.

Provenance records name the dataset a derivative came from and the calibration
a solve used. Recording those as absolute paths makes the record true only on
the machine that produced it: a published dataset then carries someone's home
directory and username into a public artifact, and a reader on another machine
gets a path that does not exist.

What a reader actually needs is *which* file, not where it sat. A path inside
the checkout keeps its repo-relative form, which another checkout can resolve;
anything else keeps its final component, which still identifies it. Nothing is
lost that a reader could have used.
"""

from __future__ import annotations

from pathlib import Path

# Walk up from this module: handumi/dataset/paths.py -> handumi -> src -> repo.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def portable_path(value: str | Path | None) -> str | None:
    """Express a path relative to the checkout, or by name when outside it."""
    if value is None:
        return None
    path = Path(value)
    try:
        resolved = path.resolve()
    except OSError:
        return path.name or str(path)
    if resolved.is_relative_to(_REPO_ROOT):
        relative = resolved.relative_to(_REPO_ROOT)
        # The repo root itself has no relative form worth writing.
        return str(relative) if str(relative) != "." else resolved.name
    return resolved.name


def repo_path(value: str | Path | None) -> Path | None:
    """Resolve what :func:`portable_path` wrote, from anywhere in the checkout.

    A stored path is repo-relative, so a report written on one machine still
    names a file another checkout can find regardless of its working directory.
    Absolute values are honoured as given, for reports written before this.
    """
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path
