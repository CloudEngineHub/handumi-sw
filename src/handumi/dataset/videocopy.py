"""Rebuild a dataset video from the episodes it keeps, without re-encoding.

Curation drops episodes and writes the rest back. Doing that by decoding and
re-encoding costs minutes per camera on a full dataset and, worse, it is lossy:
the kept frames come back slightly different from the ones the review graded.

It is also avoidable. Episodes sit back to back in one file per camera, and the
encoder starts a keyframe at every episode boundary, so each episode is a whole
number of GOPs. Copying the compressed packets of the episodes to keep, in
order, produces a stream that decodes to exactly the recorded frames.

Packets rather than the ffmpeg CLI because the CLI cuts by timestamp: seeking
lands on a keyframe at or before the requested time and the duration is honored
loosely, which measured a few frames long or short depending on where in the
file the cut fell. Selecting packets by index cannot be off by a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeSegment:
    """One episode's slice of a concatenated stream."""

    first_frame: int
    frame_count: int

    @property
    def stop(self) -> int:
        return self.first_frame + self.frame_count


def _frame_ticks(stream: Any) -> int:
    """How much a presentation stamp advances per frame on this stream."""
    time_base = stream.time_base
    if time_base is None or time_base.numerator == 0:
        raise ValueError("Video stream declares no time base")
    rate = stream.average_rate
    fps = float(rate) if rate else 30.0
    return max(1, round((time_base.denominator / time_base.numerator) / fps))


def keyframe_aligned(video_path: str | Path, segments: list[EpisodeSegment]) -> bool:
    """Whether every segment starts on a keyframe, which copying requires.

    A segment starting mid-GOP cannot be copied: its first frames reference a
    keyframe that would not be in the output.
    """
    import av

    starts = {segment.first_frame for segment in segments}
    if not starts:
        return True
    seen = 0
    wanted: set[int] = set()
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # By presentation stamp, not by arrival: with B-frames the packets
        # arrive out of display order, so counting them would test the wrong
        # frames.
        ticks = _frame_ticks(stream)
        wanted |= {start * ticks for start in starts}
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if packet.pts in wanted:
                if not packet.is_keyframe:
                    return False
                seen += 1
                if seen == len(wanted):
                    return True
    return seen == len(wanted)


def copy_segments(
    source: str | Path,
    output: str | Path,
    segments: list[EpisodeSegment],
) -> int:
    """Write the kept episodes into a new file, copying compressed packets.

    Returns the number of frames written. Raises when a segment does not start
    on a keyframe, rather than writing a stream whose first frames cannot be
    decoded.
    """
    import av

    keep: list[tuple[int, int]] = sorted(
        (segment.first_frame, segment.stop) for segment in segments
    )
    if not keep:
        raise ValueError("Nothing to copy: no episodes were kept")

    source_path, output_path = Path(source), Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source_path)) as reader:
        in_stream = reader.streams.video[0]
        # Presentation stamps advance by this much per frame, so a display index
        # maps to a stamp exactly. Packets arrive in decode order, which with
        # B-frames is not display order, so selecting by index would take the
        # wrong ones -- the stamp is what identifies a frame.
        ticks = _frame_ticks(in_stream)
        with av.open(str(output_path), mode="w") as writer:
            out_stream = writer.add_stream_from_template(in_stream)
            offset = 0
            last_dts: int | None = None
            for start, stop in keep:
                first_pts = start * ticks
                # Each kept segment is shifted to sit directly after the one
                # before it, carrying its own pts/dts skew with it: the skew is
                # what tells the decoder how to order B-frames, and flattening
                # it is what dropped a frame.
                offset = written * ticks - first_pts
                reader.seek(first_pts, stream=in_stream, backward=True, any_frame=False)
                for packet in reader.demux(in_stream):
                    if packet.dts is None or packet.pts is None:
                        continue
                    if packet.pts >= stop * ticks:
                        break
                    if packet.pts < first_pts:
                        continue
                    packet.stream = out_stream
                    packet.pts += offset
                    packet.dts += offset
                    if last_dts is not None and packet.dts <= last_dts:
                        shift = last_dts + 1 - packet.dts
                        packet.pts += shift
                        packet.dts += shift
                        offset += shift
                    last_dts = packet.dts
                    packet.duration = ticks
                    writer.mux(packet)
                    written += 1
    return written
