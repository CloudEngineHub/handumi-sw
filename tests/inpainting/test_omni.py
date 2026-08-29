from __future__ import annotations

import base64
import sys
from types import ModuleType, SimpleNamespace

from handumi.inpainting.omni import edit_clip


def test_edit_clip_explicitly_requests_a_non_streaming_interaction(monkeypatch, tmp_path):
    calls = []
    payload = b"generated video"
    active = SimpleNamespace(name="ACTIVE")

    class FakeFiles:
        def upload(self, *, file):
            return SimpleNamespace(state=active, uri=f"files/{file}")

    class FakeInteractions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                id="interaction-1",
                status="completed",
                errors=None,
                output_video=SimpleNamespace(
                    data=base64.b64encode(payload).decode("ascii"),
                    uri=None,
                ),
            )

    client = SimpleNamespace(files=FakeFiles(), interactions=FakeInteractions())
    fake_genai = ModuleType("google.genai")
    setattr(fake_genai, "Client", lambda **_kwargs: client)
    fake_google = ModuleType("google")
    setattr(fake_google, "genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    output = tmp_path / "raw" / "result.mp4"
    result = edit_clip(
        tmp_path / "input.mp4",
        [],
        "replace the arm",
        output,
        api_key="test-key",
    )

    assert calls[0]["stream"] is False
    assert output.read_bytes() == payload
    assert result.interaction_id == "interaction-1"
