"""Write a derivative dataset whose context camera shows the robot.

Everything except one video stream is carried over untouched: the same rows, the
same actions, the same wrist cameras. What changes is
``observation.images.workspace``, and the metadata records exactly what produced
it, so a dataset can be traced back to the model, prompt and reference that
painted its pixels.

Episodes sit back to back inside each per-stream file, so the rewritten stream is
rebuilt by concatenating one segment per episode -- the inpainted clip where
there is one, the recorded frames where there is not. The layout, the timestamps
and the chunk and file indices come out identical, which is what lets the action
rows stay untouched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from handumi.inpainting.clip import probe_frames
from handumi.inpainting.ledger import now_iso

WORKSPACE_VIDEO_KEY = "observation.images.workspace"


@dataclass
class InpaintedDatasetReport:
    """What the writer put in, and where each episode's pixels came from."""

    output: Path
    video_key: str
    total_episodes: int
    inpainted_episodes: list[int] = field(default_factory=list)
    passthrough_episodes: list[int] = field(default_factory=list)
    frames_expected: int = 0
    frames_written: int = 0

    @property
    def frames_match(self) -> bool:
        return self.frames_expected == self.frames_written

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "video_key": self.video_key,
            "total_episodes": self.total_episodes,
            "inpainted_episodes": self.inpainted_episodes,
            "passthrough_episodes": self.passthrough_episodes,
            "frames_expected": self.frames_expected,
            "frames_written": self.frames_written,
            "frames_match": self.frames_match,
        }


def _episode_rows(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path).to_pydict()
        for index in range(len(table["episode_index"])):
            rows.append({key: column[index] for key, column in table.items()})
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def _concat(segments: list[Path], output: Path, encoding: list[str]) -> None:
    """Join per-episode segments into one stream file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for segment in segments:
            handle.write(f"file '{segment.resolve()}'\n")
        listing = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-an", *encoding, str(output)],
            check=True,
        )
    finally:
        listing.unlink(missing_ok=True)


def _cut(source_video: Path, first_frame: int, frames: int, fps: int, output: Path,
         encoding: list[str]) -> Path:
    last = first_frame + frames - 1
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source_video),
         "-vf", f"select='between(n,{first_frame},{last})',setpts=N/{fps}/TB",
         "-frames:v", str(frames), "-r", str(fps), "-an", *encoding, str(output)],
        check=True,
    )
    return output


def write_inpainted_dataset(
    source: Path,
    output: Path,
    videos: dict[int, Path],
    *,
    video_key: str = WORKSPACE_VIDEO_KEY,
    provenance: dict[str, Any] | None = None,
) -> InpaintedDatasetReport:
    """Copy ``source`` to ``output`` with ``video_key`` repainted.

    ``videos`` maps an episode index to its inpainted clip. Episodes without one
    keep their recorded footage, so a partially inpainted dataset is still a
    complete, playable dataset rather than one with holes.
    """
    if not videos:
        raise ValueError("No inpainted videos to write.")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another output.")

    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    if video_key not in info["features"]:
        raise KeyError(f"{source} has no {video_key}.")
    fps = int(info["fps"])
    encoding = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16"]

    rows = _episode_rows(source)
    report = InpaintedDatasetReport(
        output=output, video_key=video_key, total_episodes=len(rows)
    )

    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=".inpaint-"))
    try:
        work = staging / output.name
        shutil.copytree(source, work)

        # One segment per episode, in episode order, so the rebuilt file lands on
        # exactly the timeline the action rows already describe.
        prefix = f"videos/{video_key}"
        segments_dir = staging / "segments"
        segments_dir.mkdir()
        by_file: dict[tuple[int, int], list[Path]] = {}
        for row in rows:
            episode = int(row["episode_index"])
            frames = int(row["length"])
            key = (int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]))
            report.frames_expected += frames

            if episode in videos:
                segment = videos[episode]
                got = probe_frames(segment)
                if got != frames:
                    raise ValueError(
                        f"Episode {episode} has {frames} rows but its video has {got} frames; "
                        "writing it would give the dataset pictures and labels that disagree."
                    )
                report.inpainted_episodes.append(episode)
            else:
                original = source / info["video_path"].format(
                    video_key=video_key, chunk_index=key[0], file_index=key[1]
                )
                segment = _cut(
                    original,
                    round(float(row[f"{prefix}/from_timestamp"]) * fps),
                    frames,
                    fps,
                    segments_dir / f"ep{episode:06d}.mp4",
                    encoding,
                )
                report.passthrough_episodes.append(episode)
            by_file.setdefault(key, []).append(segment)

        for (chunk_index, file_index), segments in by_file.items():
            target = work / info["video_path"].format(
                video_key=video_key, chunk_index=chunk_index, file_index=file_index
            )
            _concat(segments, target, encoding)
            report.frames_written += probe_frames(target)

        if not report.frames_match:
            raise ValueError(
                f"Rewrote {report.frames_written} frames for {report.frames_expected} rows; "
                "refusing to publish a dataset whose video and actions disagree."
            )

        info.setdefault("handumi", {})["context_inpainting"] = {
            "video_key": video_key,
            "source": str(source),
            "written_at": now_iso(),
            "inpainted_episodes": report.inpainted_episodes,
            "passthrough_episodes": report.passthrough_episodes,
            **(provenance or {}),
        }
        (work / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

        work.replace(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return report
