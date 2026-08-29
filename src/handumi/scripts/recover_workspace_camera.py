#!/usr/bin/env python3
"""Recover the context camera's pose from a recording that never calibrated it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from handumi.inpainting import (
    WORKSPACE_VIDEO_KEY,
    detect_marker,
    read_video,
    resolve_episode_clip,
    retarget_offset_px,
    solve_camera_from_table,
)
from handumi.inpainting.ledger import now_iso

REPORT_NAME = "handumi_camera_from_table.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the workspace camera's pose in the table frame from the tracking "
            "controller's own trajectory, for sessions where the ChArUco stage was "
            "never run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="Local dataset root.")
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode whose motion drives the solve."
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help=f"Defaults to DATASET/meta/{REPORT_NAME}.",
    )
    parser.add_argument(
        "--max-error-px", type=float, default=6.0,
        help="Refuse a solve whose mean reprojection error exceeds this.",
    )
    return parser


def _active_side(state: np.ndarray) -> tuple[str, slice]:
    """Whichever controller actually moved: an idle one carries no information."""
    sides = {"left": slice(0, 3), "right": slice(7, 10)}
    travelled = {
        name: float(np.linalg.norm(np.diff(state[:, span], axis=0), axis=1).sum())
        for name, span in sides.items()
    }
    name = max(travelled, key=travelled.__getitem__)
    return name, sides[name]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    info = json.loads((args.dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    try:
        cameras = info["handumi"]["spatial_session_calibration"]["spatial_calibration"]["cameras"]
        camera = cameras["workspace"]
    except KeyError as exc:
        raise SystemExit(
            f"{args.dataset} carries no workspace-camera intrinsics; nothing to solve against."
        ) from exc
    matrix = np.array(camera["matrix"], dtype=np.float64)
    distortion = np.array(camera["distortion"], dtype=np.float64)

    video, spec = resolve_episode_clip(
        args.dataset, args.episode, video_key=WORKSPACE_VIDEO_KEY
    )
    frames = read_video(video)[spec.source_offset:spec.source_offset + spec.frames]

    table = pq.read_table(args.dataset / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    rows = np.array(table["observation.state"], dtype=np.float32)
    episodes = pq.read_table(
        args.dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    index = episodes["episode_index"].index(args.episode)
    start = int(episodes["dataset_from_index"][index])
    state = rows[start:start + spec.frames]

    side, span = _active_side(state)
    positions = state[:, span].astype(np.float64)
    print(f"episode {args.episode}: {spec.frames} frames, active controller: {side}")

    try:
        solved = solve_camera_from_table(detect_marker(frames), positions, matrix, distortion)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    position = solved.camera_position_m
    print(f"  correspondences {solved.correspondences}, inliers {solved.inliers}")
    print(f"  reprojection: mean {solved.mean_error_px:.2f} px, "
          f"median {solved.median_error_px:.2f}, max {solved.max_error_px:.2f}")
    print(f"  camera at [{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}] m "
          f"in the table frame")

    payload = {
        "kind": "handumi_camera_from_table",
        "schema_version": 1,
        "created_at": now_iso(),
        "dataset": str(args.dataset),
        "episode": args.episode,
        "active_controller": side,
        "video_key": WORKSPACE_VIDEO_KEY,
        "camera_from_table": solved.to_dict(),
    }

    solves = args.dataset / "meta" / f"handumi_screening_{info.get('robot_type', '')}_solves.npz"
    for candidate in sorted((args.dataset / "meta").glob("handumi_screening_*_solves.npz")):
        solves = candidate
        break
    if solves.exists():
        cached = np.load(solves)
        key = f"{args.episode}/{side}_pos_error_m"
        if key in cached:
            payload["retarget_offset_px"] = retarget_offset_px(
                cached[key][:spec.frames], solved, positions, matrix
            )
            offset = payload["retarget_offset_px"]
            print(f"  retarget offset on screen: mean {offset['mean_px']} px, "
                  f"max {offset['max_px']} px")

    if solved.mean_error_px > args.max_error_px:
        raise SystemExit(
            f"Mean reprojection error {solved.mean_error_px:.2f} px exceeds "
            f"{args.max_error_px:.1f} px; not writing a pose this uncertain."
        )

    report = args.report or (args.dataset / "meta" / REPORT_NAME)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {report}")


if __name__ == "__main__":
    sys.exit(main())
