"""Extract the source clip and put a generated one back on the dataset clock.

The API guarantees neither frame rate nor frame count, so a returned clip is
resampled to the source timing and verified frame-for-frame. A video whose
frames no longer line up with the action rows silently mislabels the dataset,
which is worse than no video at all.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

MAX_UPLOAD_SECONDS = 10.0
# How far the returned duration may drift from what was asked before scaling it
# stops being a correction and starts being a distortion.
MAX_TIME_CORRECTION = 0.10


@dataclass(frozen=True)
class ClipSpec:
    """A frame range of one episode, and where it sits in the video file.

    ``first_frame`` counts from the start of the *episode*; ``source_offset`` is
    where that episode begins inside the shared per-stream mp4. Episodes share
    one file in a raw capture, so cutting from frame zero of the file would
    silently return the first episode's footage under another episode's name.
    """

    episode: int
    first_frame: int
    frames: int
    fps: int
    width: int
    height: int
    source_offset: int = 0
    episode_frames: int = 0

    @property
    def covers_episode(self) -> bool:
        """Whether the clip spans the whole episode.

        A video shorter than the episode it belongs to cannot be written beside
        that episode's action rows: every frame after the clip would have a label
        and no picture.
        """
        return self.episode_frames == 0 or self.first_frame + self.frames >= self.episode_frames

    @property
    def missing_frames(self) -> int:
        return max(self.episode_frames - (self.first_frame + self.frames), 0)

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    @property
    def time_scale(self) -> float:
        """How much the clip is sped up to fit the upload limit.

        The API caps the *duration* it accepts, not the frame count, so an
        episode longer than the cap is sent compressed in time and expanded again
        on return. The frame grid is restored exactly; what drops is how many
        distinct frames the model got to edit.
        """
        return max(self.seconds / MAX_UPLOAD_SECONDS, 1.0)

    @property
    def file_first_frame(self) -> int:
        return self.source_offset + self.first_frame

    @property
    def name(self) -> str:
        last = self.first_frame + self.frames - 1
        return f"ep{self.episode:03d}_f{self.first_frame:03d}-{last:03d}"

    def validate_uploadable(self) -> None:
        """Nothing to refuse: a clip over the cap is time-compressed instead."""


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip().splitlines()[0])


def probe_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(next(line for line in out.splitlines() if line.strip().isdigit()))


def extract_clip(source_video: Path, spec: ClipSpec, output: Path) -> Path:
    """Cut ``spec`` out of the dataset's shared per-stream mp4."""
    output.parent.mkdir(parents=True, exist_ok=True)
    first = spec.file_first_frame
    last = first + spec.frames - 1
    scale = spec.time_scale
    upload_frames = spec.frames if scale == 1.0 else int(MAX_UPLOAD_SECONDS * spec.fps)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source_video),
         "-vf",
         f"select='between(n,{first},{last})',setpts=(N/{spec.fps}/TB)/{scale},fps={spec.fps}",
         "-frames:v", str(upload_frames), "-r", str(spec.fps), "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(output)],
        check=True,
    )
    got = probe_frames(output)
    if got != upload_frames:
        raise RuntimeError(f"extracted {got} frames, expected {upload_frames}")
    return output


def align_to_source(generated: Path, spec: ClipSpec, output: Path) -> Path:
    """Resample a generated clip back onto the dataset's frame grid.

    The returned duration is measured rather than assumed. The model answers at
    its own frame rate and a little short or long of what it was given, and a
    long episode was sped up on the way out, so the clip is scaled to exactly the
    episode's duration before it is sampled. Padding or truncating instead would
    either invent frames or drop recorded ones.

    Also drops the audio track: the model adds one by default and the dataset's
    video streams carry none.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    target = spec.frames / spec.fps
    actual = probe_duration(generated)
    scale = target / actual if actual > 0 else 1.0
    drift = scale / spec.time_scale
    if not (1 - MAX_TIME_CORRECTION) <= drift <= (1 + MAX_TIME_CORRECTION):
        raise RuntimeError(
            f"the generated video is {actual:.3f}s where {target / spec.time_scale:.3f}s "
            f"was asked ({drift:.2f}x off); stretching it to {target:.3f}s would distort "
            "the motion rather than correct it, so it cannot be put back on the dataset clock"
        )
    stretch = f"setpts=PTS*{scale:.9f}," if abs(scale - 1.0) > 1e-9 else ""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(generated),
         "-vf", f"{stretch}fps={spec.fps},scale={spec.width}:{spec.height}:flags=lanczos",
         "-frames:v", str(spec.frames), "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(output)],
        check=True,
    )
    got = probe_frames(output)
    if got != spec.frames:
        raise RuntimeError(
            f"aligned clip has {got} frames, expected {spec.frames}; "
            "the generated video cannot be put back on the dataset clock"
        )
    return output


def extract_reference(source_video: Path, spec: ClipSpec, output: Path) -> Path:
    """Cut the episode at its own frame rate, for compositing and grading.

    The upload clip may be time-compressed to fit the API; this one never is, so
    it has exactly the episode's frames and lines up with the aligned result.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    first = spec.file_first_frame
    last = first + spec.frames - 1
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source_video),
         "-vf", f"select='between(n,{first},{last})',setpts=N/{spec.fps}/TB",
         "-frames:v", str(spec.frames), "-r", str(spec.fps), "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(output)],
        check=True,
    )
    got = probe_frames(output)
    if got != spec.frames:
        raise RuntimeError(f"reference has {got} frames, expected {spec.frames}")
    return output


def _episode_row(dataset: Path, episode: int) -> dict:
    """Return one episode's metadata row from ``meta/episodes``."""
    files = sorted((dataset / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset / 'meta' / 'episodes'}")
    for path in files:
        table = pq.read_table(path).to_pydict()
        for index, value in enumerate(table["episode_index"]):
            if int(value) == episode:
                return {key: column[index] for key, column in table.items()}
    raise KeyError(f"Episode {episode} is not in {dataset}")


def resolve_episode_clip(
    dataset: Path,
    episode: int,
    *,
    video_key: str,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> tuple[Path, ClipSpec]:
    """Locate an episode's video and the frame range to edit.

    Everything comes from the dataset's own metadata -- the ``video_path``
    template, the episode's chunk and file indices, and where it starts inside
    that file -- so this works for a raw capture whose episodes share one mp4
    and for a converted dataset with one file per episode.

    The clip spans the whole episode unless ``max_frames`` caps it: an episode
    longer than the upload limit is time-compressed rather than cut short.
    """
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    if video_key not in features:
        cameras = sorted(k for k in features if k.startswith("observation.images."))
        raise KeyError(
            f"{dataset} has no {video_key}. It declares {cameras or 'no cameras'}; "
            "a recording without a context camera has nothing to repaint."
        )
    feature = features[video_key]
    height, width = int(feature["shape"][0]), int(feature["shape"][1])
    fps = int(info["fps"])

    row = _episode_row(dataset, episode)
    prefix = f"videos/{video_key}"
    video = dataset / info["video_path"].format(
        video_key=video_key,
        chunk_index=int(row[f"{prefix}/chunk_index"]),
        file_index=int(row[f"{prefix}/file_index"]),
    )

    length = int(row["length"])
    if first_frame >= length:
        raise ValueError(f"Episode {episode} has {length} frames; --first-frame {first_frame} is past its end.")
    spec = ClipSpec(
        episode=episode,
        first_frame=first_frame,
        frames=min(max_frames, length - first_frame) if max_frames else length - first_frame,
        fps=fps,
        width=width,
        height=height,
        source_offset=round(float(row[f"{prefix}/from_timestamp"]) * fps),
        episode_frames=length,
    )
    return video, spec
