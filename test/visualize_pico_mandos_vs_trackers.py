#!/usr/bin/env python3
"""Compare controller-only upper-body reconstruction with PICO object trackers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from handumi.retargeting.pico_upper_body import infer_elbow, parse_axis_map


UPPER_NAMES = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]
UPPER_LINES = np.asarray(
    sum(
        ([2, parent, child] for parent, child in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),
            (6, 7),
            (3, 8),
            (8, 9),
            (9, 10),
        ]),
        [],
    ),
    dtype=np.int_,
)

LEFT_WRIST_INDEX = UPPER_NAMES.index("left_wrist")
RIGHT_WRIST_INDEX = UPPER_NAMES.index("right_wrist")


def _load_fps(root: Path) -> float:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 30.0
    with info_path.open("r", encoding="utf-8") as f:
        return float(json.load(f).get("fps", 30.0))


def _stack_pose_column(values: np.ndarray, *, width: int = 7) -> np.ndarray:
    poses = [np.asarray(value, dtype=np.float32) for value in values]
    stacked = np.stack(poses)
    if stacked.ndim != 2 or stacked.shape[1] < width:
        raise ValueError(f"Expected pose column with shape (N, >={width}); got {stacked.shape}")
    return stacked[:, :width]


def _stack_tracker_column(values: np.ndarray) -> np.ndarray:
    frames: list[np.ndarray] = []
    max_count = 0
    for value in values:
        frame = np.asarray(value, dtype=object)
        if frame.size == 0:
            arr = np.zeros((0, 7), dtype=np.float32)
        else:
            arr = np.stack([np.asarray(item, dtype=np.float32) for item in frame])
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr[:, :7]
        frames.append(arr)
        max_count = max(max_count, len(arr))

    trackers = np.full((len(frames), max_count, 7), np.nan, dtype=np.float32)
    for index, frame in enumerate(frames):
        trackers[index, : len(frame), : frame.shape[1]] = frame
    return trackers


def load_episode(root: Path, episode: int) -> tuple[pd.DataFrame, float]:
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    df = df[df["episode_index"] == episode].copy()
    if df.empty:
        raise ValueError(f"Episode {episode} not found in {root}")

    sort_columns = [col for col in ("index", "frame_index") if col in df.columns]
    if sort_columns:
        df.sort_values(sort_columns, inplace=True)
    return df.reset_index(drop=True), _load_fps(root)


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _estimate_shoulders(
    anchors: np.ndarray,
    left_wrists: np.ndarray,
    right_wrists: np.ndarray,
    *,
    shoulder_width: float,
    anchor_to_shoulder: float,
    anchor_to_head: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    lateral_fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pelvises: list[np.ndarray] = []
    spines: list[np.ndarray] = []
    chests: list[np.ndarray] = []
    heads: list[np.ndarray] = []
    necks: list[np.ndarray] = []
    left_shoulders: list[np.ndarray] = []
    right_shoulders: list[np.ndarray] = []
    laterals: list[np.ndarray] = []

    for anchor, left_wrist, right_wrist in zip(anchors, left_wrists, right_wrists):
        lateral = left_wrist - right_wrist
        lateral = lateral - up * float(np.dot(lateral, up))
        lateral = _normalize(lateral, lateral_fallback)
        shoulder_center = anchor + up * anchor_to_shoulder

        pelvises.append(anchor - up * 0.34)
        spines.append(anchor - up * 0.17)
        chests.append(anchor)
        necks.append(anchor + up * (anchor_to_head * 0.68))
        heads.append(anchor + up * anchor_to_head)
        left_shoulders.append(shoulder_center + lateral * (shoulder_width * 0.5))
        right_shoulders.append(shoulder_center - lateral * (shoulder_width * 0.5))
        laterals.append(lateral)

    return (
        np.asarray(chests, dtype=np.float32),
        np.asarray(pelvises, dtype=np.float32),
        np.asarray(spines, dtype=np.float32),
        np.asarray(heads, dtype=np.float32),
        np.asarray(necks, dtype=np.float32),
        np.asarray(left_shoulders, dtype=np.float32),
        np.asarray(right_shoulders, dtype=np.float32),
        np.asarray(laterals, dtype=np.float32),
    )


def _estimate_arm_lengths(
    shoulders: np.ndarray,
    wrists: np.ndarray,
    *,
    upper_ratio: float,
    extension_ratio: float,
    percentile: float,
) -> tuple[float, float]:
    distances = np.linalg.norm(wrists - shoulders, axis=1)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return 0.28, 0.28
    arm_length = float(np.percentile(distances, percentile) / extension_ratio)
    return arm_length * upper_ratio, arm_length * (1.0 - upper_ratio)


def _constrain_wrist(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    *,
    arm_length: float,
    max_reach_ratio: float,
) -> np.ndarray:
    if not np.all(np.isfinite(wrist)):
        return np.full(3, np.nan, dtype=np.float32)

    offset = wrist - shoulder
    distance = float(np.linalg.norm(offset))
    if distance < 1e-6:
        return wrist.astype(np.float32)

    max_reach = arm_length * max_reach_ratio
    if distance <= max_reach:
        return wrist.astype(np.float32)
    return (shoulder + offset / distance * max_reach).astype(np.float32)


def reconstruct_upper_body(
    anchor_pose: np.ndarray,
    left_wrist_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    *,
    shoulder_width: float,
    anchor_to_shoulder: float,
    anchor_to_head: float,
    upper_ratio: float,
    extension_ratio: float,
    length_percentile: float,
    bend_forward: float,
    bend_down: float,
    bend_side: float,
    arm_length: float,
    max_reach_ratio: float,
    left_lengths: tuple[float, float] | None = None,
    right_lengths: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    anchors = anchor_pose[:, :3].astype(np.float32)
    left_wrists = left_wrist_pose[:, :3].astype(np.float32)
    right_wrists = right_wrist_pose[:, :3].astype(np.float32)
    chests, pelvises, spines, heads, necks, left_shoulders, right_shoulders, laterals = _estimate_shoulders(
        anchors,
        left_wrists,
        right_wrists,
        shoulder_width=shoulder_width,
        anchor_to_shoulder=anchor_to_shoulder,
        anchor_to_head=anchor_to_head,
    )
    if left_lengths is None:
        if arm_length > 0.0:
            left_lengths = (arm_length * upper_ratio, arm_length * (1.0 - upper_ratio))
        else:
            left_lengths = _estimate_arm_lengths(
                left_shoulders,
                left_wrists,
                upper_ratio=upper_ratio,
                extension_ratio=extension_ratio,
                percentile=length_percentile,
            )
    if right_lengths is None:
        if arm_length > 0.0:
            right_lengths = (arm_length * upper_ratio, arm_length * (1.0 - upper_ratio))
        else:
            right_lengths = _estimate_arm_lengths(
                right_shoulders,
                right_wrists,
                upper_ratio=upper_ratio,
                extension_ratio=extension_ratio,
                percentile=length_percentile,
            )

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    upper = np.empty((len(heads), len(UPPER_NAMES), 3), dtype=np.float32)
    for index in range(len(heads)):
        shoulder_center = 0.5 * (left_shoulders[index] + right_shoulders[index])
        left_wrist = _constrain_wrist(
            left_shoulders[index],
            left_wrists[index],
            arm_length=sum(left_lengths),
            max_reach_ratio=max_reach_ratio,
        )
        right_wrist = _constrain_wrist(
            right_shoulders[index],
            right_wrists[index],
            arm_length=sum(right_lengths),
            max_reach_ratio=max_reach_ratio,
        )
        wrist_center = np.nanmean(np.stack([left_wrist, right_wrist]), axis=0)
        forward = wrist_center - shoulder_center
        forward = forward - up * float(np.dot(forward, up))
        forward = _normalize(forward, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        left_hint = (
            forward * bend_forward
            + up * bend_down
            + laterals[index] * bend_side
        )
        right_hint = (
            forward * bend_forward
            + up * bend_down
            - laterals[index] * bend_side
        )

        left_elbow = infer_elbow(
            left_shoulders[index],
            left_wrist,
            upper_length=left_lengths[0],
            forearm_length=left_lengths[1],
            bend_hint=left_hint,
        )
        right_elbow = infer_elbow(
            right_shoulders[index],
            right_wrist,
            upper_length=right_lengths[0],
            forearm_length=right_lengths[1],
            bend_hint=right_hint,
        )
        upper[index] = np.stack(
            [
                pelvises[index],
                spines[index],
                chests[index],
                necks[index],
                heads[index],
                left_shoulders[index],
                left_elbow,
                left_wrist,
                right_shoulders[index],
                right_elbow,
                right_wrist,
            ]
        )
    return upper, left_lengths, right_lengths


def order_trackers_by_controller_side(
    trackers: np.ndarray,
    left_controller: np.ndarray,
    right_controller: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """Return tracker poses ordered as left/right using controller proximity."""

    if trackers.shape[1] < 2:
        raise ValueError("Need two object trackers to infer left/right wrists.")

    positions = trackers[:, :2, :3]
    left = left_controller[:, :3]
    right = right_controller[:, :3]

    identity_left = np.linalg.norm(positions[:, 0] - left, axis=1)
    identity_right = np.linalg.norm(positions[:, 1] - right, axis=1)
    swap_left = np.linalg.norm(positions[:, 1] - left, axis=1)
    swap_right = np.linalg.norm(positions[:, 0] - right, axis=1)

    identity_cost = float(np.nanmedian(identity_left + identity_right))
    swap_cost = float(np.nanmedian(swap_left + swap_right))
    if identity_cost <= swap_cost:
        ordered = trackers[:, [0, 1], :].copy()
        return ordered, (0, 1), (
            float(np.nanmedian(identity_left)),
            float(np.nanmedian(identity_right)),
        )

    ordered = trackers[:, [1, 0], :].copy()
    return ordered, (1, 0), (
        float(np.nanmedian(swap_left)),
        float(np.nanmedian(swap_right)),
    )


def apply_display_transform(
    values: np.ndarray,
    *,
    origin: np.ndarray,
    axis_map: str,
    scale: float,
    offset: np.ndarray,
) -> np.ndarray:
    transform = parse_axis_map(axis_map)
    flat = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    out = np.asarray(
        [transform(point - origin) * scale + offset for point in flat],
        dtype=np.float32,
    )
    return out.reshape(values.shape)


def polyline(points: np.ndarray) -> "pv.PolyData":
    import pyvista as pv

    valid_points: list[np.ndarray] = []
    lines: list[int] = []
    run: list[int] = []
    for point in np.asarray(points, dtype=np.float32):
        if np.all(np.isfinite(point)):
            run.append(len(valid_points))
            valid_points.append(point)
            continue
        if len(run) >= 2:
            lines.extend([len(run), *run])
        run = []
    if len(run) >= 2:
        lines.extend([len(run), *run])

    if not valid_points:
        return pv.PolyData(np.zeros((1, 3), dtype=np.float32))
    poly = pv.PolyData(np.asarray(valid_points, dtype=np.float32))
    poly.lines = np.asarray(lines, dtype=np.int_)
    return poly


def update_polyline(poly: "pv.PolyData", points: np.ndarray) -> None:
    valid_points: list[np.ndarray] = []
    lines: list[int] = []
    run: list[int] = []
    for point in np.asarray(points, dtype=np.float32):
        if np.all(np.isfinite(point)):
            run.append(len(valid_points))
            valid_points.append(point)
            continue
        if len(run) >= 2:
            lines.extend([len(run), *run])
        run = []
    if len(run) >= 2:
        lines.extend([len(run), *run])

    poly.points = (
        np.asarray(valid_points, dtype=np.float32)
        if valid_points
        else np.zeros((1, 3), dtype=np.float32)
    )
    poly.lines = np.asarray(lines, dtype=np.int_)


def valid_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = points[np.all(np.isfinite(points), axis=1)]
    if len(valid) == 0:
        return np.zeros((1, 3), dtype=np.float32)
    return valid


def trail(values: np.ndarray, frame: int, length: int) -> np.ndarray:
    start = 0 if length <= 0 else max(0, frame + 1 - length)
    return values[start : frame + 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("outputs/datasets/pico_object_test")
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--axis-map", default="z,x,y")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=1.35)
    parser.add_argument("--trail-length", type=int, default=180)
    parser.add_argument("--shoulder-width", type=float, default=0.38)
    parser.add_argument("--anchor-to-shoulder", type=float, default=0.18)
    parser.add_argument("--anchor-to-head", type=float, default=0.42)
    parser.add_argument("--upper-ratio", type=float, default=0.44)
    parser.add_argument(
        "--arm-length",
        type=float,
        default=0.62,
        help="Visual human arm length in meters. Use 0 to estimate from data.",
    )
    parser.add_argument("--max-reach-ratio", type=float, default=0.98)
    parser.add_argument("--extension-ratio", type=float, default=0.92)
    parser.add_argument("--length-percentile", type=float, default=95.0)
    parser.add_argument("--bend-forward", type=float, default=0.65)
    parser.add_argument("--bend-down", type=float, default=-1.0)
    parser.add_argument("--bend-side", type=float, default=0.25)
    parser.add_argument(
        "--max-tracker-controller-distance",
        type=float,
        default=0.0,
        help="Optional wrist-tracker filter in meters. 0 disables filtering.",
    )
    parser.add_argument("--background", default="black")
    parser.add_argument("--point-size", type=float, default=16.0)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    df, dataset_fps = load_episode(args.dataset_root, args.episode)
    left_controller = _stack_pose_column(df["observation.pico.left_controller_pose"].values)
    right_controller = _stack_pose_column(df["observation.pico.right_controller_pose"].values)
    headset = _stack_pose_column(df["observation.pico.headset_pose"].values)
    trackers = _stack_tracker_column(df["observation.pico.motion_tracker_pose"].values)
    counts = (
        np.asarray(df["observation.pico.motion_tracker_count"].values, dtype=np.int32)
        if "observation.pico.motion_tracker_count" in df
        else np.sum(np.isfinite(trackers[:, :, 0]), axis=1)
    )

    tracker_wrists, tracker_mapping, tracker_median_distances = (
        order_trackers_by_controller_side(trackers, left_controller, right_controller)
    )
    if args.max_tracker_controller_distance > 0:
        tracker_wrists[
            np.linalg.norm(tracker_wrists[:, 0, :3] - left_controller[:, :3], axis=1)
            > args.max_tracker_controller_distance,
            0,
            :,
        ] = np.nan
        tracker_wrists[
            np.linalg.norm(tracker_wrists[:, 1, :3] - right_controller[:, :3], axis=1)
            > args.max_tracker_controller_distance,
            1,
            :,
        ] = np.nan

    mandos_upper, left_lengths, right_lengths = reconstruct_upper_body(
        headset,
        left_controller,
        right_controller,
        shoulder_width=args.shoulder_width,
        anchor_to_shoulder=args.anchor_to_shoulder,
        anchor_to_head=args.anchor_to_head,
        upper_ratio=args.upper_ratio,
        extension_ratio=args.extension_ratio,
        length_percentile=args.length_percentile,
        bend_forward=args.bend_forward,
        bend_down=args.bend_down,
        bend_side=args.bend_side,
        arm_length=args.arm_length,
        max_reach_ratio=args.max_reach_ratio,
    )
    tracker_upper, _, _ = reconstruct_upper_body(
        headset,
        tracker_wrists[:, 0],
        tracker_wrists[:, 1],
        shoulder_width=args.shoulder_width,
        anchor_to_shoulder=args.anchor_to_shoulder,
        anchor_to_head=args.anchor_to_head,
        upper_ratio=args.upper_ratio,
        extension_ratio=args.extension_ratio,
        length_percentile=args.length_percentile,
        bend_forward=args.bend_forward,
        bend_down=args.bend_down,
        bend_side=args.bend_side,
        arm_length=args.arm_length,
        max_reach_ratio=args.max_reach_ratio,
        left_lengths=left_lengths,
        right_lengths=right_lengths,
    )

    frame_indices = list(range(args.start_frame, len(df), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected.")

    origin = headset[frame_indices[0], :3].copy()
    mandos_display_all = apply_display_transform(
        mandos_upper,
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
        offset=np.array([-args.spacing * 0.5, 0.0, 0.0], dtype=np.float32),
    )
    trackers_display_all = apply_display_transform(
        tracker_upper,
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
        offset=np.array([args.spacing * 0.5, 0.0, 0.0], dtype=np.float32),
    )
    raw_tracker_display_all = apply_display_transform(
        tracker_wrists[:, :, :3],
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
        offset=np.array([args.spacing * 0.5, 0.0, 0.0], dtype=np.float32),
    )
    upper_display = mandos_display_all[frame_indices].copy()
    trackers_display = trackers_display_all[frame_indices].copy()
    raw_tracker_display = raw_tracker_display_all[frame_indices].copy()

    floor_z = float(
        np.nanmin(
            [
                np.nanmin(upper_display[0, :, 2]),
                np.nanmin(trackers_display[0, :, 2]),
                np.nanmin(raw_tracker_display[0, :, 2]),
            ]
        )
    )
    upper_display[:, :, 2] -= floor_z
    trackers_display[:, :, 2] -= floor_z
    raw_tracker_display[:, :, 2] -= floor_z

    print(
        "Loaded episode "
        f"{args.episode}: frames={len(df)}, selected={len(frame_indices)}, "
        f"fps={dataset_fps:g}, tracker_count min/max={int(np.nanmin(counts))}/{int(np.nanmax(counts))}"
    )
    print(
        "Tracker side mapping: "
        f"left=tracker[{tracker_mapping[0]}] median_dist={tracker_median_distances[0]:.3f}m, "
        f"right=tracker[{tracker_mapping[1]}] median_dist={tracker_median_distances[1]:.3f}m"
    )
    print(
        "Inferred arm lengths "
        f"left upper/forearm={left_lengths[0]:.3f}/{left_lengths[1]:.3f} m, "
        f"right upper/forearm={right_lengths[0]:.3f}/{right_lengths[1]:.3f} m"
    )

    if args.smoke_test:
        return

    import pyvista as pv

    plotter = pv.Plotter(window_size=(1500, 850), off_screen=args.screenshot is not None)
    plotter.set_background(args.background)
    plotter.add_axes()
    plotter.add_floor("-z", color="gray", lighting=False, pad=1.0)

    upper_points = pv.PolyData(upper_display[0])
    upper_bones = pv.PolyData(upper_display[0])
    upper_bones.lines = UPPER_LINES
    tracker_points = pv.PolyData(trackers_display[0])
    tracker_bones = pv.PolyData(trackers_display[0])
    tracker_bones.lines = UPPER_LINES
    raw_tracker_points = pv.PolyData(valid_points(raw_tracker_display[0]))

    plotter.add_mesh(
        upper_points,
        color="crimson",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="Mandos joints",
    )
    plotter.add_mesh(
        upper_bones,
        color="white",
        line_width=args.line_width,
        render_lines_as_tubes=True,
        label="Mandos bones",
    )
    plotter.add_mesh(
        tracker_points,
        color="deepskyblue",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="Trackers inferred joints",
    )
    plotter.add_mesh(
        tracker_bones,
        color="dodgerblue",
        line_width=args.line_width,
        render_lines_as_tubes=True,
        label="Trackers inferred bones",
    )
    plotter.add_mesh(
        raw_tracker_points,
        color="yellow",
        point_size=args.point_size + 3.0,
        render_points_as_spheres=True,
        label="Raw tracker points",
    )

    left_wrist_trail = polyline(upper_display[:1, LEFT_WRIST_INDEX])
    right_wrist_trail = polyline(upper_display[:1, RIGHT_WRIST_INDEX])
    plotter.add_mesh(left_wrist_trail, color="lime", line_width=5, label="Mandos L")
    plotter.add_mesh(right_wrist_trail, color="orange", line_width=5, label="Mandos R")
    tracker_trails = []
    for tracker_index, wrist_index in enumerate((LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX)):
        tracker_trail = polyline(trackers_display[:1, wrist_index])
        tracker_trails.append(tracker_trail)
        color = "springgreen" if tracker_index == 0 else "gold"
        plotter.add_mesh(
            tracker_trail,
            color=color,
            line_width=5,
            label=f"Tracker {'L' if tracker_index == 0 else 'R'}",
        )
    plotter.add_text("Mandos", position=(0.13, 0.92), viewport=True)
    plotter.add_text("Object trackers", position=(0.62, 0.92), viewport=True)

    plotter.add_legend(size=(0.25, 0.28), loc="upper right")
    plotter.camera_position = [
        (0.0, -4.0, 1.7),
        (0.0, 0.0, 0.7),
        (0.0, 0.0, 1.0),
    ]

    if args.screenshot is not None:
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Screenshot saved to {args.screenshot}")
        return

    plotter.show(auto_close=False, interactive_update=True)
    playback_fps = float(args.fps or dataset_fps)
    dt = 0.0 if playback_fps <= 0 else 1.0 / playback_fps
    while True:
        next_time = time.perf_counter()
        for selected_index, frame in enumerate(frame_indices):
            upper_points.points = upper_display[selected_index]
            upper_bones.points = upper_display[selected_index]
            current_tracker_skeleton = trackers_display[selected_index]
            tracker_points.points = current_tracker_skeleton
            tracker_bones.points = current_tracker_skeleton
            raw_tracker_points.points = valid_points(raw_tracker_display[selected_index])

            update_polyline(
                left_wrist_trail,
                trail(
                    upper_display[:, LEFT_WRIST_INDEX],
                    selected_index,
                    args.trail_length,
                ),
            )
            update_polyline(
                right_wrist_trail,
                trail(
                    upper_display[:, RIGHT_WRIST_INDEX],
                    selected_index,
                    args.trail_length,
                ),
            )
            for tracker_index, tracker_trail in enumerate(tracker_trails):
                update_polyline(
                    tracker_trail,
                    trail(
                        trackers_display[
                            :,
                            LEFT_WRIST_INDEX
                            if tracker_index == 0
                            else RIGHT_WRIST_INDEX,
                        ],
                        selected_index,
                        args.trail_length,
                    ),
                )

            plotter.add_text(
                f"episode={args.episode} frame={frame}/{len(df) - 1}",
                position="lower_left",
                color="white",
                font_size=12,
                name="frame_label",
            )
            plotter.update()
            if dt > 0:
                next_time += dt
                time.sleep(max(0.0, next_time - time.perf_counter()))
            if selected_index == len(frame_indices) - 1 and not args.loop:
                plotter.show()
                return


if __name__ == "__main__":
    main()
