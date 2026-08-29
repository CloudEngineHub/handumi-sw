"""Append-only run ledger and API budget accounting for context inpainting.

A generation call costs money, so the ledger is the loop's memory: intents are
recorded *before* the call is issued (a crash cannot lose the count) and a call
the API refused never generated video, so it must not consume the budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MAX_CALLS = 8


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Budget:
    """How many generation calls this run has spent, and how many it may."""

    used: int
    limit: int

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def __str__(self) -> str:
        return f"{self.used}/{self.limit}"


class Ledger:
    """One JSON object per line, appended, never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def budget(self, limit: int = DEFAULT_MAX_CALLS) -> Budget:
        rows = self.rows()
        intents = sum(1 for r in rows if r.get("phase") == "intent" and r.get("spent_call"))
        refused = sum(1 for r in rows if r.get("phase") == "refused")
        return Budget(used=max(intents - refused, 0), limit=limit)

    def already_tried(self, prompt: str, reference_sha: str, clip: str) -> bool:
        """True when this exact (clip, prompt, reference) triple already generated.

        Repeating one is a bug, not an experiment: it spends a call to learn
        nothing. An intent the API never accepted produced no video, so it does
        not count as tried and must not block the retry.
        """
        rows = self.rows()
        refused = {r.get("run_id") for r in rows if r.get("phase") == "refused"}
        for row in rows:
            if row.get("phase") != "intent" or row.get("run_id") in refused:
                continue
            shas = {ref.get("sha256") for ref in row.get("references", [])}
            if row.get("prompt") == prompt and reference_sha in shas and row.get("clip") == clip:
                return True
        return False
