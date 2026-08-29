from __future__ import annotations

import json

from handumi.inpainting import Ledger


def _intent(prompt: str = "p", sha: str = "abc", clip: str = "ep000_f000-299") -> dict:
    return {
        "phase": "intent",
        "spent_call": True,
        "prompt": prompt,
        "clip": clip,
        "references": [{"path": "ref.png", "sha256": sha}],
    }


def test_budget_counts_only_issued_calls(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.budget(8).used == 0

    ledger.append(_intent())
    assert ledger.budget(8).used == 1

    ledger.append({"phase": "result", "spent_call": True})
    assert ledger.budget(8).used == 1, "a result must not be counted twice"


def test_refused_call_does_not_consume_budget(tmp_path):
    """A call the API rejected never generated video and was never billed."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_intent())
    ledger.append({"phase": "refused", "spent_call": False})
    assert ledger.budget(8).used == 0


def test_budget_exhaustion(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for index in range(2):
        ledger.append(_intent(prompt=f"p{index}"))
    assert ledger.budget(2).exhausted
    assert not ledger.budget(3).exhausted


def test_already_tried_matches_the_whole_triple(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_intent())

    assert ledger.already_tried("p", "abc", "ep000_f000-299")
    assert not ledger.already_tried("other prompt", "abc", "ep000_f000-299")
    assert not ledger.already_tried("p", "different-sha", "ep000_f000-299")
    assert not ledger.already_tried("p", "abc", "ep001_f000-299")


def test_ledger_is_append_only(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"phase": "intent", "spent_call": True, "episode": 0})
    ledger.append({"phase": "result", "episode": 0})
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["phase"] for line in lines] == ["intent", "result"]


def test_a_refused_intent_does_not_block_the_retry(tmp_path):
    """Nothing was generated, so the same triple is still worth one call."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({**_intent(), "run_id": "2026-08-29T10:00:00Z"})
    assert ledger.already_tried("p", "abc", "ep000_f000-299")

    ledger.append({"phase": "refused", "run_id": "2026-08-29T10:00:00Z", "spent_call": False})
    assert not ledger.already_tried("p", "abc", "ep000_f000-299")
    assert ledger.budget(8).used == 0
