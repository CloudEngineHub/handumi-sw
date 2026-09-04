"""Hub snapshot helpers for LeRobot datasets."""

from __future__ import annotations

import json
from pathlib import Path

from handumi.dataset.reader import ensure_dataset, ensure_metadata


def _write_info(root: Path, *, videos: bool = True) -> None:
    features = {"observation.state": {"dtype": "float32"}}
    if videos:
        features["observation.images.cam"] = {"dtype": "video"}
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "features": features})
    )


def _write_parquet(root: Path) -> None:
    dest = root / "data" / "chunk-000"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "episode_000000.parquet").write_bytes(b"parquet")


def _write_video(root: Path) -> None:
    dest = root / "videos" / "observation.images.cam" / "chunk-000"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "episode_000000.mp4").write_bytes(b"video")


def test_ensure_dataset_downloads_the_full_hub_tree(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "tblock-all-piper-clean-bi_piper_follower"
    seen: list[dict[str, object]] = []

    def fake_download(resolved, *, allow_patterns=None):
        seen.append({"repo_id": resolved.repo_id, "allow_patterns": allow_patterns})
        _write_info(resolved.root)
        _write_parquet(resolved.root)
        _write_video(resolved.root)
        (resolved.root / "README.md").write_text("# dataset\n")

    monkeypatch.setattr("handumi.dataset.reader._download_hub_snapshot", fake_download)
    info = ensure_dataset(repo_id="murobotics/tblock-all-piper-clean-bi_piper_follower", root=dest)
    assert info["total_episodes"] == 1
    assert seen == [
        {"repo_id": "murobotics/tblock-all-piper-clean-bi_piper_follower", "allow_patterns": None}
    ]
    assert (dest / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (dest / "videos" / "observation.images.cam" / "chunk-000" / "episode_000000.mp4").is_file()
    assert (dest / "README.md").is_file()


def test_ensure_dataset_skips_a_complete_local_tree(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "already-local"
    _write_info(dest)
    _write_parquet(dest)
    _write_video(dest)
    monkeypatch.setattr(
        "handumi.dataset.reader._download_hub_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("complete trees must not fetch")),
    )
    info = ensure_dataset(repo_id="murobotics/already-local", root=dest)
    assert info["total_episodes"] == 1


def test_ensure_dataset_refetches_when_only_metadata_is_present(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "meta-only"
    _write_info(dest)
    fetched = []

    def fake_download(resolved, *, allow_patterns=None):
        fetched.append(allow_patterns)
        _write_parquet(resolved.root)
        _write_video(resolved.root)

    monkeypatch.setattr("handumi.dataset.reader._download_hub_snapshot", fake_download)
    ensure_dataset(repo_id="murobotics/meta-only", root=dest)
    assert fetched == [None]


def test_ensure_metadata_still_limits_the_download_to_meta(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "meta-fetch"
    seen: list[list[str] | None] = []

    def fake_download(resolved, *, allow_patterns=None):
        seen.append(allow_patterns)
        _write_info(resolved.root, videos=False)

    monkeypatch.setattr("handumi.dataset.reader._download_hub_snapshot", fake_download)
    ensure_metadata(repo_id="murobotics/meta-fetch", root=dest)
    assert seen == [["meta/**"]]
