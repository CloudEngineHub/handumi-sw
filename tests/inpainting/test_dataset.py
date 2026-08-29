from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from handumi.inpainting import write_inpainted_dataset
from handumi.inpainting.clip import probe_frames

KEY = "observation.images.workspace"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _video(path: Path, frames: int, fps: int = 30, width: int = 64, height: int = 36) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps}",
         "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _dataset(root: Path, lengths: list[int], fps: int = 30) -> Path:
    """A miniature LeRobot layout: episodes back to back in one stream file."""
    root.mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    _video(root / VIDEO_PATH.format(video_key=KEY, chunk_index=0, file_index=0), sum(lengths), fps)

    starts, total = [], 0
    for length in lengths:
        starts.append(total)
        total += length
    columns = {
        "episode_index": list(range(len(lengths))),
        "length": lengths,
        f"videos/{KEY}/chunk_index": [0] * len(lengths),
        f"videos/{KEY}/file_index": [0] * len(lengths),
        f"videos/{KEY}/from_timestamp": [s / fps for s in starts],
        f"videos/{KEY}/to_timestamp": [(s + n) / fps for s, n in zip(starts, lengths, strict=True)],
    }
    meta = root / "meta" / "episodes" / "chunk-000"
    meta.mkdir(parents=True)
    pq.write_table(pa.table(columns), meta / "file-000.parquet")
    (root / "meta" / "info.json").write_text(
        json.dumps({
            "fps": fps,
            "total_episodes": len(lengths),
            "video_path": VIDEO_PATH,
            "features": {KEY: {"dtype": "video", "shape": [36, 64, 3]}},
        }),
        encoding="utf-8",
    )
    return root


def test_the_rewritten_stream_keeps_every_frame(tmp_path):
    source = _dataset(tmp_path / "src", [12, 20, 8])
    replacement = _video(tmp_path / "ep1.mp4", 20)

    report = write_inpainted_dataset(source, tmp_path / "out", {1: replacement})

    assert report.frames_match
    assert report.frames_expected == 40
    written = tmp_path / "out" / VIDEO_PATH.format(video_key=KEY, chunk_index=0, file_index=0)
    assert probe_frames(written) == 40


def test_episodes_without_a_clip_keep_their_recording(tmp_path):
    source = _dataset(tmp_path / "src", [12, 20, 8])
    report = write_inpainted_dataset(source, tmp_path / "out", {1: _video(tmp_path / "ep1.mp4", 20)})

    assert report.inpainted_episodes == [1]
    assert report.passthrough_episodes == [0, 2]


def test_a_clip_that_disagrees_with_the_rows_is_refused(tmp_path):
    """Pictures and labels that disagree are worse than no dataset at all."""
    source = _dataset(tmp_path / "src", [12, 20, 8])
    wrong = _video(tmp_path / "ep1.mp4", 17)

    with pytest.raises(ValueError, match="pictures and labels that disagree"):
        write_inpainted_dataset(source, tmp_path / "out", {1: wrong})
    assert not (tmp_path / "out").exists(), "a refused write must leave nothing behind"


def test_provenance_is_recorded(tmp_path):
    source = _dataset(tmp_path / "src", [12, 20])
    write_inpainted_dataset(
        source, tmp_path / "out", {0: _video(tmp_path / "ep0.mp4", 12)},
        provenance={"model": "gemini-omni-1.1-flash", "prompt_sha256": "abc"},
    )
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    record = info["handumi"]["context_inpainting"]
    assert record["model"] == "gemini-omni-1.1-flash"
    assert record["prompt_sha256"] == "abc"
    assert record["inpainted_episodes"] == [0]
    assert record["video_key"] == KEY


def test_the_other_streams_and_rows_are_carried_over(tmp_path):
    source = _dataset(tmp_path / "src", [12, 20])
    write_inpainted_dataset(source, tmp_path / "out", {0: _video(tmp_path / "ep0.mp4", 12)})
    assert (tmp_path / "out" / "data" / "chunk-000" / "file-000.parquet").exists()
    original = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    copied = pq.read_table(tmp_path / "out" / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    assert original == copied, "action rows must not change"


def test_it_refuses_to_overwrite_an_existing_dataset(tmp_path):
    source = _dataset(tmp_path / "src", [12])
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        write_inpainted_dataset(source, tmp_path / "out", {0: _video(tmp_path / "ep0.mp4", 12)})
