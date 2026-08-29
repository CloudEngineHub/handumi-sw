from __future__ import annotations

import json
import subprocess

import pytest

from handumi.inpainting import (
    ClipSpec,
    align_to_source,
    extract_clip,
    extract_reference,
    resolve_episode_clip,
)
from handumi.inpainting.clip import MAX_UPLOAD_SECONDS, probe_frames


def _spec(frames: int, first_frame: int = 0, episode_frames: int | None = None) -> ClipSpec:
    return ClipSpec(
        episode=0,
        first_frame=first_frame,
        frames=frames,
        fps=30,
        width=64,
        height=36,
        episode_frames=episode_frames if episode_frames is not None else frames,
    )


def _source(path, frames: int, fps: int = 30, width: int = 64, height: int = 36):
    """A synthetic recording standing in for the context camera."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps}",
         "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_a_short_episode_is_not_time_compressed():
    assert _spec(146).time_scale == 1.0


def test_a_long_episode_is_compressed_to_the_upload_limit():
    spec = _spec(467)
    assert spec.seconds > MAX_UPLOAD_SECONDS
    assert spec.time_scale == pytest.approx(467 / 30 / MAX_UPLOAD_SECONDS)


def test_extraction_fits_the_upload_limit(tmp_path):
    """The API caps duration, so a long episode is sped up rather than cut."""
    source = _source(tmp_path / "src.mp4", 467)
    clip = extract_clip(source, _spec(467), tmp_path / "clip.mp4")
    assert probe_frames(clip) == int(MAX_UPLOAD_SECONDS * 30)


def test_a_short_episode_is_extracted_whole(tmp_path):
    source = _source(tmp_path / "src.mp4", 146)
    clip = extract_clip(source, _spec(146), tmp_path / "clip.mp4")
    assert probe_frames(clip) == 146


@pytest.mark.parametrize("episode_frames", [146, 320, 467])
def test_round_trip_restores_the_episode_frame_count(tmp_path, episode_frames):
    """Whatever the model returns must land back on the dataset frame grid."""
    spec = _spec(episode_frames)
    source = _source(tmp_path / "src.mp4", episode_frames)
    clip = extract_clip(source, spec, tmp_path / "clip.mp4")

    # The model answers at its own frame rate and resolution.
    generated = tmp_path / "generated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", "fps=24,scale=640:360",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(generated)],
        check=True,
    )
    assert probe_frames(generated) != episode_frames

    aligned = align_to_source(generated, spec, tmp_path / "aligned.mp4")
    assert probe_frames(aligned) == episode_frames


def test_a_generated_clip_far_too_short_is_still_refused(tmp_path):
    """Scaling corrects a small drift; a clip a sixth the length is not a drift."""
    spec = _spec(300)
    truncated = _source(tmp_path / "short.mp4", 40, fps=24)
    with pytest.raises(RuntimeError, match="distort"):
        align_to_source(truncated, spec, tmp_path / "aligned.mp4")


def test_a_dataset_without_a_context_camera_says_so(tmp_path):
    """A recording with only wrist cameras has nothing for this stage to repaint."""
    meta = tmp_path / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({
            "fps": 30,
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"observation.images.left_wrist": {"shape": [480, 640, 3]}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="left_wrist"):
        resolve_episode_clip(tmp_path, 0, video_key="observation.images.workspace")


def test_the_reference_keeps_the_episode_frame_rate(tmp_path):
    """A compressed upload cannot be composited against; the reference can."""
    spec = _spec(467)
    source = _source(tmp_path / "src.mp4", 467)

    upload = extract_clip(source, spec, tmp_path / "upload.mp4")
    reference = extract_reference(source, spec, tmp_path / "reference.mp4")

    assert probe_frames(upload) == int(MAX_UPLOAD_SECONDS * 30)
    assert probe_frames(reference) == 467


@pytest.mark.parametrize("returned_seconds", [9.7, 10.0, 10.4])
def test_a_generated_clip_slightly_off_duration_still_lands_on_the_grid(
    tmp_path, returned_seconds
):
    """The model answers a little short or long; that must not cost a paid call."""
    spec = _spec(300)
    generated = tmp_path / "generated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size=640x360:rate=24:duration={returned_seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(generated)],
        check=True,
    )
    aligned = align_to_source(generated, spec, tmp_path / "aligned.mp4")
    assert probe_frames(aligned) == 300
