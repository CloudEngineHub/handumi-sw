from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from handumi.dataset.videocopy import (
    EpisodeSegment,
    copy_segments,
    keyframe_aligned,
)

FPS = 10


def _write_video(path: Path, episode_lengths: list[int], *, keyframe_every: int) -> None:
    """A video whose frames are individually identifiable by their brightness."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(episode_lengths)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0", "g": str(keyframe_every)}
        for index in range(total):
            # Spaced so the RGB -> YUV -> RGB round trip cannot make two
            # frames indistinguishable.
            array = np.full((32, 32, 3), 10 + index * 5, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def _segments(episode_lengths: list[int], keep: list[int]) -> list[EpisodeSegment]:
    starts = np.concatenate(([0], np.cumsum(episode_lengths)[:-1])).astype(int)
    return [
        EpisodeSegment(int(starts[index]), int(episode_lengths[index]))
        for index in keep
    ]


def _decoded_values(path: Path) -> list[int]:
    import av

    values: list[int] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            values.append(int(np.median(frame.to_ndarray(format="rgb24"))))
    return values


def test_copy_keeps_exactly_the_frames_of_the_kept_episodes(tmp_path: Path) -> None:
    lengths = [10, 10, 10, 10]
    source = tmp_path / "source.mp4"
    _write_video(source, lengths, keyframe_every=10)
    segments = _segments(lengths, [0, 2, 3])

    written = copy_segments(source, tmp_path / "out.mp4", segments)

    assert written == 30
    values = _decoded_values(tmp_path / "out.mp4")
    assert len(values) == 30
    # Frame content, in order: episode 0 then 2 then 3, with episode 1 gone.
    expected = [10 + i * 5 for i in list(range(0, 10)) + list(range(20, 40))]
    assert max(abs(a - b) for a, b in zip(values, expected)) <= 2


def test_copy_reports_the_frame_count_ffprobe_sees(tmp_path: Path) -> None:
    """The count the caller records has to be the count a reader decodes.

    A muxed stream whose timestamps confuse the decoder yields fewer frames than
    packets written, which would put the video one frame out of step with the
    action rows for every later episode.
    """
    lengths = [12, 8, 12]
    source = tmp_path / "source.mp4"
    _write_video(source, lengths, keyframe_every=4)
    output = tmp_path / "out.mp4"

    written = copy_segments(source, output, _segments(lengths, [0, 2]))

    probed = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output),
        ],
        text=True,
    ).strip()
    assert written == int(probed) == 24


def _write_textured_video(path: Path, frames: int, *, keyframe_every: int) -> None:
    """Noise, so the encoder produces real GOPs.

    A flat colour compresses to a keyframe every frame whatever the GOP size is
    asked to be, which cannot exercise an unaligned cut.
    """
    import av

    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = 64, 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": str(keyframe_every)}
        for _ in range(frames):
            array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())


def test_keyframe_alignment_is_checked_before_copying(tmp_path: Path) -> None:
    """A segment starting mid-GOP references a keyframe the output would lack."""
    import av

    source = tmp_path / "source.mp4"
    _write_textured_video(source, 24, keyframe_every=8)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        ticks = round((1 / stream.time_base) / rate)
        keyframes = {
            packet.pts // ticks
            for packet in container.demux(stream)
            if packet.pts is not None and packet.is_keyframe
        }
    inside = next(index for index in range(1, 24) if index not in keyframes)

    assert not keyframe_aligned(source, [EpisodeSegment(inside, 4)])


def test_aligned_source_reports_alignment(tmp_path: Path) -> None:
    lengths = [6, 6, 6]
    source = tmp_path / "source.mp4"
    _write_video(source, lengths, keyframe_every=6)

    assert keyframe_aligned(source, _segments(lengths, [0, 1, 2]))


def test_copy_refuses_an_empty_selection(tmp_path: Path) -> None:
    lengths = [5, 5]
    source = tmp_path / "source.mp4"
    _write_video(source, lengths, keyframe_every=5)

    with pytest.raises(ValueError, match="no episodes were kept"):
        copy_segments(source, tmp_path / "out.mp4", [])
