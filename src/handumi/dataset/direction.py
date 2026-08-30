"""Find episodes that demonstrate a task backwards.

A session usually holds more than the demonstrations. Resetting the scene
between takes -- putting the object back where it started -- is itself a
recorded episode when the operator forgets to stop the recorder, and it carries
the same task string as the demonstrations around it. Nothing else in the
pipeline sees this. Such an episode is a flawless recording: sensors healthy,
tracking valid, timestamps regular, and it retargets as well as any other. The
defect is in what it teaches, and a policy trained on it learns to undo the task.

The check learns the task from the dataset instead of being told it. Every
episode's scene change is the difference between how the workspace looks at its
end and at its start; a demonstration and its reset produce opposite changes.
Comparing each episode's change against the dataset's own median change
therefore separates them without knowing the task, the object, or the robot.

The comparison is weighted by how consistently each pixel changes across the
dataset. Without that weighting the operator dominates -- their arm is the
largest moving thing in frame and it moves differently every take -- and the
object, which is small but changes the same way every time, is drowned out.
Weighting by ``|median| / MAD`` inverts that: pixels that always change the same
way count, pixels that change at random do not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from handumi.dataset.quality import EpisodeQualityReport, QualityFinding

DIRECTION_SCHEMA_VERSION = 1

# Frames averaged at each end of an episode. A single frame is whatever the
# operator's hand happened to occlude; the median over a short window at each
# end removes them, because they move and the scene does not.
EDGE_FRAMES = 15

# The workspace camera is the only stream that sees the task. Wrist cameras
# move with the tool, so their first and last frames describe where the hand
# went, not what changed on the table.
DEFAULT_VIDEO_KEY = "observation.images.workspace"

# Downsampled frame size for the comparison. Large enough that a hand-sized
# object survives, small enough that lighting and sensor noise average out.
FRAME_WIDTH, FRAME_HEIGHT = 84, 48

# Cosine similarity against the dataset's median change. Zero is the natural
# boundary -- a negative value is literally the opposite change -- and the
# measured separation on real sessions is wide enough that the exact figure
# does not matter: reversed episodes sat at -0.24 or below while every forward
# episode sat at +0.12 or above, across 212 episodes of three sessions.
REVERSED_CEILING = 0.0

# Below this, an episode barely changes the scene at all and its direction is
# not a meaningful question -- the cosine of near-zero vectors is noise.
MIN_CHANGE_MAGNITUDE = 1e-3


@dataclass(frozen=True)
class EpisodeDirection:
    episode_index: int
    frame_count: int
    similarity: float
    change_magnitude: float

    @property
    def reversed_demonstration(self) -> bool:
        return (
            self.change_magnitude >= MIN_CHANGE_MAGNITUDE
            and self.similarity < REVERSED_CEILING
        )


def episode_edge_frames(
    video_path: str | Path,
    episode_bounds: list[tuple[int, int]],
    *,
    edge_frames: int = EDGE_FRAMES,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read one median start frame and one median end frame per episode.

    Episodes sit back to back in one video file, so this walks the stream once
    and collects only the frames it needs rather than seeking per episode.
    """
    import cv2

    wanted: dict[int, list[tuple[int, str]]] = {}
    for index, (start, length) in enumerate(episode_bounds):
        window = min(edge_frames, max(length // 3, 1))
        for offset in range(window):
            wanted.setdefault(start + offset, []).append((index, "start"))
            wanted.setdefault(start + length - 1 - offset, []).append((index, "end"))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    collected: dict[tuple[int, str], list[np.ndarray]] = {}
    position = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for key in wanted.get(position, ()):
                resized = cv2.resize(
                    frame,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)
                collected.setdefault(key, []).append(resized)
            position += 1
    finally:
        capture.release()

    edges: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in range(len(episode_bounds)):
        start_frames = collected.get((index, "start"))
        end_frames = collected.get((index, "end"))
        if not start_frames or not end_frames:
            raise ValueError(
                f"Video is shorter than the episode metadata claims: episode {index}"
            )
        edges[index] = (
            np.median(np.stack(start_frames), axis=0),
            np.median(np.stack(end_frames), axis=0),
        )
    return edges


def score_directions(
    edges: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_counts: dict[int, int],
) -> list[EpisodeDirection]:
    """Grade every episode's scene change against the dataset's median change."""
    order = sorted(edges)
    if len(order) < 3:
        raise ValueError(
            "Direction needs at least three episodes: the reference is the "
            "dataset's own median change, which two episodes cannot establish."
        )
    changes = np.stack([(edges[i][1] - edges[i][0]).ravel() for i in order])
    reference = np.median(changes, axis=0)
    deviation = np.median(np.abs(changes - reference), axis=0)
    weight = np.abs(reference) / (deviation + 1e-6)
    peak = float(weight.max())
    weight = weight / peak if peak > 0 else np.ones_like(weight)

    weighted_reference = reference * weight
    reference_norm = float(np.linalg.norm(weighted_reference))
    scale = float(np.median(np.linalg.norm(changes, axis=1))) or 1.0

    results: list[EpisodeDirection] = []
    for position, index in enumerate(order):
        weighted = changes[position] * weight
        norm = float(np.linalg.norm(weighted))
        similarity = (
            float(weighted @ weighted_reference / (norm * reference_norm))
            if norm > 0 and reference_norm > 0
            else 0.0
        )
        results.append(
            EpisodeDirection(
                episode_index=index,
                frame_count=frame_counts[index],
                similarity=similarity,
                change_magnitude=float(np.linalg.norm(changes[position])) / scale,
            )
        )
    return results


def direction_reports(
    directions: list[EpisodeDirection],
    *,
    fps: float,
) -> list[EpisodeQualityReport]:
    """Express the scores in the findings schema the rest of the review uses."""
    reports: list[EpisodeQualityReport] = []
    for item in directions:
        findings: tuple[QualityFinding, ...] = ()
        if item.reversed_demonstration:
            findings = (
                QualityFinding(
                    code="reversed_demonstration",
                    # A warning, not a rejection: the reference is the majority
                    # of this dataset, so a session that deliberately holds both
                    # directions would see half its episodes flagged. What the
                    # measurement establishes is that the episode runs against
                    # the others, and whether that disqualifies it is the
                    # reviewer's call.
                    severity="warning",
                    message=(
                        "The scene changes opposite to the rest of the dataset: "
                        "this episode appears to undo the task rather than "
                        "perform it, which is what a reset take between "
                        "demonstrations looks like."
                    ),
                    metrics={
                        "direction_similarity": round(item.similarity, 4),
                        "ceiling": REVERSED_CEILING,
                    },
                ),
            )
        reports.append(
            EpisodeQualityReport(
                episode_index=item.episode_index,
                frame_count=item.frame_count,
                duration_s=item.frame_count / fps if fps else 0.0,
                findings=findings,
                metrics={
                    "direction_similarity": round(item.similarity, 4),
                    "scene_change_magnitude": round(item.change_magnitude, 4),
                },
            )
        )
    return reports


def analyze_dataset_direction(
    root: str | Path,
    *,
    video_key: str = DEFAULT_VIDEO_KEY,
) -> tuple[list[EpisodeQualityReport], dict[str, Any]]:
    """Grade every episode of a local dataset, and report what it measured."""
    import pyarrow.parquet as pq

    dataset_root = Path(root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Not a dataset root: {dataset_root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if video_key not in info.get("features", {}):
        raise ValueError(f"Dataset has no {video_key}; nothing sees the workspace.")
    fps = float(info.get("fps", 30) or 30)

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise ValueError("Dataset is missing episode metadata")
    table = pq.read_table(episode_files, columns=["episode_index", "length"])
    lengths = [int(value) for value in table["length"].to_pylist()]
    indices = [int(value) for value in table["episode_index"].to_pylist()]

    video_files = sorted((dataset_root / "videos" / video_key).glob("chunk-*/*.mp4"))
    if len(video_files) != 1:
        raise ValueError(
            f"Direction expects one video file for {video_key}, found "
            f"{len(video_files)}"
        )

    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(int)
    bounds = [(int(start), length) for start, length in zip(starts, lengths)]
    edges = episode_edge_frames(video_files[0], bounds)
    frame_counts = {position: lengths[position] for position in range(len(lengths))}
    directions = score_directions(edges, frame_counts)
    # score_directions works in episode order; restore the recorded indices.
    directions = [
        EpisodeDirection(
            episode_index=indices[item.episode_index],
            frame_count=item.frame_count,
            similarity=item.similarity,
            change_magnitude=item.change_magnitude,
        )
        for item in directions
    ]
    summary = {
        "video_key": video_key,
        "edge_frames": EDGE_FRAMES,
        "reversed_ceiling": REVERSED_CEILING,
        "min_change_magnitude": MIN_CHANGE_MAGNITUDE,
        "reversed_episode_indices": [
            item.episode_index for item in directions if item.reversed_demonstration
        ],
    }
    return direction_reports(directions, fps=fps), summary
