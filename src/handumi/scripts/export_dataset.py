#!/usr/bin/env python3
"""Export a converted HandUMI joints dataset as a drop-in for an external stack.

``handumi convert`` writes joints in the canonical HandUMI vector: radians for
the arms and meters for the gripper, which is what keeps the data physical
and comparable across robots. A training stack built around one follower
robot expects that robot's own vector instead, in the units its driver
produces. This command rewrites a converted dataset into such a layout at
the file level -- parquet columns, metadata and statistics -- without
re-encoding video, and records exactly which constants did it.

The canonical dataset is left untouched: the export is a derived view, and
``handumi replay-joints`` can play either one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from handumi.dataset.canonical import is_canonical_state_layout
from handumi.dataset.external_layouts import (
    EXTERNAL_LAYOUTS,
    ExternalJointLayout,
    check_layout_limits,
    clip_to_driver_range,
    external_layout_for_name,
    external_layouts_for_robot,
    from_canonical,
    out_of_range_counts,
)
from handumi.dataset.selection import resolve_dataset_selection
from handumi.robots.registry import load_embodiment

STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_PREFIX = "observation.images."
DROPPED_COLUMNS = ("calibration_id", "source_kind")
# Scalar columns every LeRobot v3 dataset carries, in the reference order.
SCALAR_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a converted HandUMI joints dataset into an external "
            "stack's joint layout (names, units, robot_type, camera names)."
        )
    )
    parser.add_argument("dataset", help="Local path of a converted joints dataset.")
    parser.add_argument(
        "--layout",
        choices=sorted(EXTERNAL_LAYOUTS),
        default=None,
        help=(
            "Target LeRobot robot_type. Defaults to the one layout that "
            "describes the dataset's embodiment."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root. Defaults to a sibling named <source>-<layout suffix>.",
    )
    parser.add_argument(
        "--camera-map",
        default=None,
        help=(
            "Comma-separated old=new camera names, e.g. "
            "'left_wrist=left,workspace=top,right_wrist=right'. Cameras not "
            "listed keep their name. Defaults to the layout's own map."
        ),
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help=(
            "Record arm joints in degrees instead of the normalized range, as "
            "LeRobot's use_degrees config does; only plugins that expose that "
            "option accept it. The gripper keeps its own mode."
        ),
    )
    parser.add_argument(
        "--clip-tolerance-rad",
        type=float,
        default=2e-3,
        help=(
            "Overshoots past a joint limit up to this many radians are clipped "
            "to the driver's range; the IK solver's soft limit constraint "
            "settles a millidegree past the limit, which the driver would "
            "reject outright. Larger overshoots are left and reported."
        ),
    )
    parser.add_argument(
        "--no-clip", action="store_true", help="Export the raw values, never clip."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort when any exported value falls outside the range the "
            "external driver accepts, instead of warning."
        ),
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help=(
            "Reference dataset whose schema the export must match. Exit "
            "non-zero on any difference in features or robot_type."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the export plan without writing anything.",
    )
    return parser


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise SystemExit(f"Not a LeRobot dataset root: missing {path}.")
    return json.loads(path.read_text())


def require_canonical_dataset(info: dict[str, Any]) -> dict[str, Any]:
    if not is_canonical_state_layout(info):
        recorded = info.get("handumi", {}).get("state_layout") if isinstance(
            info.get("handumi"), dict
        ) else None
        raise SystemExit(
            "Export needs a converted HandUMI joints dataset (canonical "
            f"layout), but handumi.state_layout is {recorded or 'absent'!r}. "
            "Run `handumi convert` first."
        )
    return dict(info["handumi"])


def layout_for(
    robot: str, requested: str | None, *, use_degrees: bool = False
) -> ExternalJointLayout:
    """The layout to export to: the requested one, or the embodiment's only one."""
    if requested is not None:
        layout = external_layout_for_name(requested)
        if layout.robot != robot:
            raise SystemExit(
                f"Layout {layout.robot_type} describes {layout.robot!r}, but the "
                f"dataset was converted for {robot!r}."
            )
        return apply_use_degrees(layout, use_degrees)
    candidates = external_layouts_for_robot(robot)
    if len(candidates) == 1:
        return apply_use_degrees(candidates[0], use_degrees)
    known = ", ".join(f"{k} ({v.robot})" for k, v in sorted(EXTERNAL_LAYOUTS.items()))
    if not candidates:
        raise SystemExit(
            f"No LeRobot layout is defined for embodiment {robot!r}. Known: {known}. "
            "Add one to handumi/dataset/external_layouts.py."
        )
    raise SystemExit(
        f"Several layouts describe {robot!r}; pass --layout. Known: {known}."
    )


def apply_use_degrees(layout: ExternalJointLayout, use_degrees: bool) -> ExternalJointLayout:
    """Honour --use-degrees the way the plugin would, or explain why it cannot."""
    if not use_degrees:
        return layout
    try:
        variant = layout.with_use_degrees(True)
    except ValueError as exc:
        raise SystemExit(f"--use-degrees: {exc}") from exc
    if variant is layout:
        print(f"[export] note: {layout.robot_type} already records degrees; --use-degrees changes nothing.")
    return variant


def default_output_name(source_name: str, layout: ExternalJointLayout) -> str:
    """``<source>-<robot_type>``, the same name ``convert --output-layout`` gives.

    A canonical dataset is named ``<source>-<robot>-joints``; the export
    replaces that suffix rather than stacking a second one on top of it.
    """
    canonical_suffix = f"-{layout.robot}-joints"
    base = source_name.removesuffix(canonical_suffix)
    return f"{base}-{layout.output_suffix}"


def target_robot(meta: dict[str, Any]) -> str:
    target = meta.get("target_robot")
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    raise SystemExit("Dataset records no handumi.target_robot; cannot pick an embodiment.")


def parse_camera_map(
    value: str | None,
    *,
    layout: ExternalJointLayout,
    video_keys: list[str],
) -> dict[str, str]:
    """Return {old video key: new video key} in the order the output should list them."""
    if value is None:
        mapping = {
            old: new for old, new in layout.default_camera_map.items() if old in video_keys
        }
    else:
        mapping = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"--camera-map entry {item!r} is not old=new.")
            old, new = (part.strip() for part in item.split("=", 1))
            old_key = old if old.startswith(VIDEO_PREFIX) else VIDEO_PREFIX + old
            new_key = new if new.startswith(VIDEO_PREFIX) else VIDEO_PREFIX + new
            if old_key not in video_keys:
                raise SystemExit(
                    f"--camera-map names {old!r}, but the dataset has no such "
                    f"camera. Available: {', '.join(k[len(VIDEO_PREFIX):] for k in video_keys)}."
                )
            mapping[old_key] = new_key
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        raise SystemExit("--camera-map maps two cameras to the same name.")
    for old in video_keys:
        mapping.setdefault(old, old)
    return mapping


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _vector_column(table: pa.Table, key: str) -> np.ndarray:
    values = table.column(key).to_numpy(zero_copy_only=False)
    return np.stack(values).astype(np.float32)


def _fixed_list(array: np.ndarray) -> pa.Array:
    """Store a (T, D) float32 array as fixed_size_list<float>[D], as LeRobot does."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    flat = pa.array(array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, array.shape[1])


def _feature_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    from lerobot.datasets.compute_stats import get_feature_stats

    return get_feature_stats(np.asarray(array, dtype=np.float32), axis=0, keepdims=False)


def transform_data_files(
    source_root: Path,
    output_root: Path,
    *,
    layout: ExternalJointLayout,
    runtime,
    clip_tolerance_rad: float | None,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    """Rewrite every data parquet; return per-episode stats and a range report.

    ``clip_tolerance_rad`` of ``None`` exports raw values. The report holds
    ``out_of_range`` (values the driver would still reject), ``clipped``
    (per column, how many overshoots were folded back) and ``max_overshoot_rad``.
    """
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    report: dict[str, Any] = {"out_of_range": {}, "clipped": {}, "max_overshoot_rad": 0.0}
    files = sorted((source_root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No data parquet files under {source_root / 'data'}.")
    for path in files:
        table = pq.read_table(path)
        exported: dict[str, np.ndarray] = {}
        for key in (STATE_KEY, ACTION_KEY):
            canonical = _vector_column(table, key)
            values = from_canonical(canonical, layout=layout, runtime=runtime)
            if clip_tolerance_rad is not None:
                values, clipped, worst = clip_to_driver_range(
                    values, layout=layout, tolerance_rad=clip_tolerance_rad
                )
                report["max_overshoot_rad"] = max(report["max_overshoot_rad"], worst)
                for name, count in clipped.items():
                    report["clipped"][f"{key}/{name}"] = (
                        report["clipped"].get(f"{key}/{name}", 0) + count
                    )
            exported[key] = values
            for name, count in out_of_range_counts(values, layout=layout).items():
                report["out_of_range"][f"{key}/{name}"] = (
                    report["out_of_range"].get(f"{key}/{name}", 0) + count
                )

        episodes = table.column("episode_index").to_numpy()
        for episode in np.unique(episodes):
            rows = episodes == episode
            episode_stats[int(episode)] = {
                key: _feature_stats(exported[key][rows]) for key in (STATE_KEY, ACTION_KEY)
            }

        columns: dict[str, pa.Array] = {
            ACTION_KEY: _fixed_list(exported[ACTION_KEY]),
            STATE_KEY: _fixed_list(exported[STATE_KEY]),
        }
        for key in SCALAR_COLUMNS:
            columns[key] = table.column(key)
        out_path = output_root / path.relative_to(source_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out_path)
    return episode_stats, report


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _stats_to_lists(stats: dict[str, np.ndarray]) -> dict[str, list]:
    return {name: np.asarray(value).tolist() for name, value in stats.items()}


def _channels_first(value: Any) -> Any:
    """Video statistics as (channels, 1, 1), the shape LeRobot itself writes."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.size not in (1, 3, 4):
        return value
    return array.reshape(-1, 1, 1).tolist()


def _rename_video_stats(stats: dict[str, Any], camera_map: dict[str, str]) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for old, new in camera_map.items():
        block = stats.get(old)
        if not isinstance(block, dict):
            continue
        renamed[new] = {
            name: (value if name == "count" else _channels_first(value))
            for name, value in block.items()
        }
    return renamed


def write_stats_json(
    source_stats: dict[str, Any],
    output_root: Path,
    *,
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    camera_map: dict[str, str],
) -> None:
    from lerobot.datasets.compute_stats import aggregate_stats

    aggregated = aggregate_stats([episode_stats[e] for e in sorted(episode_stats)])
    stats: dict[str, Any] = {
        ACTION_KEY: _stats_to_lists(aggregated[ACTION_KEY]),
        STATE_KEY: _stats_to_lists(aggregated[STATE_KEY]),
    }
    stats.update(_rename_video_stats(source_stats, camera_map))
    for key in SCALAR_COLUMNS:
        if key in source_stats:
            stats[key] = source_stats[key]
    (output_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")


def _video_stat_array(column: pa.ChunkedArray) -> pa.Array:
    values = [_channels_first(v) if v is not None else None for v in column.to_pylist()]
    return pa.array(values, type=pa.list_(pa.list_(pa.list_(pa.float64()))))


def transform_episode_files(
    source_root: Path,
    output_root: Path,
    *,
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    camera_map: dict[str, str],
) -> None:
    files = sorted((source_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No episode metadata under {source_root / 'meta' / 'episodes'}.")
    for path in files:
        table = pq.read_table(path)
        drop = [
            name
            for name in table.column_names
            if any(name.startswith(f"stats/{key}/") for key in DROPPED_COLUMNS)
        ]
        table = table.drop_columns(drop)

        episodes = [int(e) for e in table.column("episode_index").to_pylist()]
        for key in (STATE_KEY, ACTION_KEY):
            sample = episode_stats[episodes[0]][key]
            for stat_name in sample:
                column = f"stats/{key}/{stat_name}"
                values = [np.asarray(episode_stats[e][key][stat_name]).tolist() for e in episodes]
                dtype = pa.int64() if stat_name == "count" else pa.float64()
                array = pa.array(values, type=pa.list_(dtype))
                if column in table.column_names:
                    table = table.set_column(table.column_names.index(column), column, array)
                else:
                    table = table.append_column(column, array)

        names = []
        for name in table.column_names:
            renamed = name
            for old, new in camera_map.items():
                if old != new and (name.startswith(f"videos/{old}/") or name.startswith(f"stats/{old}/")):
                    renamed = name.replace(old, new, 1)
                    break
            names.append(renamed)
        table = table.rename_columns(names)
        for name in table.column_names:
            if name.startswith("stats/" + VIDEO_PREFIX) and name.split("/")[-1] != "count":
                index = table.column_names.index(name)
                table = table.set_column(index, name, _video_stat_array(table.column(name)))

        out_path = output_root / path.relative_to(source_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)


def build_info(
    source_info: dict[str, Any],
    *,
    layout: ExternalJointLayout,
    runtime,
    camera_map: dict[str, str],
    source_repo_id: str,
    range_report: dict[str, Any] | None = None,
    clip_tolerance_rad: float | None = None,
) -> dict[str, Any]:
    source_features = source_info["features"]
    vector = {"dtype": "float32", "shape": [layout.size], "names": list(layout.names)}
    features: dict[str, Any] = {
        ACTION_KEY: dict(vector),
        STATE_KEY: {**vector, "names": list(layout.names)},
    }
    for old, new in camera_map.items():
        features[new] = dict(source_features[old])
    for key in SCALAR_COLUMNS:
        features[key] = source_features[key]

    info = {k: v for k, v in source_info.items() if k not in ("features", "handumi")}
    info["robot_type"] = layout.robot_type
    info["features"] = features
    handumi = dict(source_info.get("handumi", {}))
    handumi["state_layout"] = layout.name
    handumi["joint_names"] = list(layout.names)
    handumi["export"] = {
        "schema_version": 2,
        **layout.as_metadata(),
        "source_repo_id": source_repo_id,
        "source_state_layout": str(source_info.get("handumi", {}).get("state_layout", "")),
        "source_robot_type": source_info.get("robot_type"),
        "gripper_fraction": "width_m / gripper_max_width_m",
        "gripper_max_width_m": float(runtime.config.gripper_max_width_m),
        "camera_map": {
            old[len(VIDEO_PREFIX):]: new[len(VIDEO_PREFIX):] for old, new in camera_map.items()
        },
        "dropped_columns": list(DROPPED_COLUMNS),
        "clipping": {
            "tolerance_rad": clip_tolerance_rad,
            "clipped_values": dict(sorted((range_report or {}).get("clipped", {}).items())),
            "max_overshoot_rad": float((range_report or {}).get("max_overshoot_rad", 0.0)),
            "remaining_out_of_range": dict(
                sorted((range_report or {}).get("out_of_range", {}).items())
            ),
            "why": (
                "The IK solver's soft joint-limit constraint settles microradians "
                "past a limit; a driver with a hard accepted range rejects any value "
                "outside it instead of clipping, which would freeze the robot."
            ),
        },
        "state_semantics": (
            "IK command replayed from the robot-agnostic capture; not measured "
            "follower feedback."
        ),
    }
    info["handumi"] = handumi
    return info


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def copy_videos(source_root: Path, output_root: Path, camera_map: dict[str, str]) -> dict[str, int]:
    summary = {"linked": 0, "copied": 0}
    for old, new in camera_map.items():
        old_dir = source_root / "videos" / old
        if not old_dir.is_dir():
            raise SystemExit(f"Missing video directory {old_dir}.")
        for path in sorted(old_dir.rglob("*.mp4")):
            target = output_root / "videos" / new / path.relative_to(old_dir)
            summary[_link_or_copy(path, target)] += 1
    return summary


def copy_side_files(source_root: Path, output_root: Path) -> None:
    for relative in ("meta/tasks.parquet", "meta/source_quality.json"):
        source = source_root / relative
        if source.is_file():
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    calibrations = source_root / "meta" / "calibrations"
    if calibrations.is_dir():
        shutil.copytree(calibrations, output_root / "meta" / "calibrations", dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def schema_differences(
    exported: dict[str, Any], reference: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Differences from a reference schema, split by whether they break a drop-in.

    Blocking: robot_type, fps, a feature present on one side only, and any
    dtype/shape/names mismatch on non-video features -- these change what a
    policy trains on or what the driver receives. Informational: a video
    feature whose resolution differs -- policies resize their inputs, so the
    data still trains; it is reported because it is a fact about the source
    cameras, not something an export can or should hide.
    """
    blocking: list[str] = []
    notes: list[str] = []
    if exported.get("robot_type") != reference.get("robot_type"):
        blocking.append(
            f"robot_type: exported={exported.get('robot_type')!r} "
            f"reference={reference.get('robot_type')!r}"
        )
    if exported.get("fps") != reference.get("fps"):
        blocking.append(f"fps: exported={exported.get('fps')} reference={reference.get('fps')}")
    ours, theirs = exported["features"], reference["features"]
    for key in sorted(set(ours) | set(theirs)):
        if key not in ours:
            blocking.append(f"feature only in reference: {key}")
            continue
        if key not in theirs:
            blocking.append(f"feature only in export: {key}")
            continue
        is_video = key.startswith(VIDEO_PREFIX)
        for attribute in ("dtype", "shape", "names"):
            mine, other = ours[key].get(attribute), theirs[key].get(attribute)
            if attribute == "shape":
                mine, other = list(mine or []), list(other or [])
            if attribute == "names" and is_video:
                # 'channel' and 'channels' are both accepted by LeRobot.
                mine = [n.rstrip("s") for n in (mine or [])]
                other = [n.rstrip("s") for n in (other or [])]
            if mine != other:
                line = f"{key}.{attribute}: exported={mine!r} reference={other!r}"
                if is_video and attribute == "shape":
                    notes.append(line + " (resolution; policies resize, not blocking)")
                else:
                    blocking.append(line)
    if list(ours) != list(theirs) and not any(d.startswith("feature only") for d in blocking):
        blocking.append(f"feature order: exported={list(ours)} reference={list(theirs)}")
    return blocking, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def export_dataset(
    source_root: Path,
    output_root: Path,
    *,
    layout: ExternalJointLayout,
    camera_map: dict[str, str],
    source_repo_id: str,
    strict: bool,
    clip_tolerance_rad: float | None = 2e-3,
) -> dict[str, Any]:
    info = load_info(source_root)
    meta = require_canonical_dataset(info)
    robot = target_robot(meta)
    if robot != layout.robot:
        raise SystemExit(
            f"Dataset was converted for {robot!r}, but layout {layout.robot_type} "
            f"describes {layout.robot!r}."
        )
    runtime = load_embodiment(robot)
    check_layout_limits(layout, runtime)

    output_root.mkdir(parents=True, exist_ok=False)
    episode_stats, report = transform_data_files(
        source_root,
        output_root,
        layout=layout,
        runtime=runtime,
        clip_tolerance_rad=clip_tolerance_rad,
    )
    clipped_total = sum(report["clipped"].values())
    if clipped_total:
        print(
            f"[export] clipped {clipped_total} value(s) that overshot a joint limit "
            f"by at most {report['max_overshoot_rad']:.2e} rad "
            f"(tolerance {clip_tolerance_rad:g} rad); recorded in handumi.export.clipping."
        )
    bad = report["out_of_range"]
    if bad:
        total = sum(bad.values())
        detail = ", ".join(f"{name}={count}" for name, count in sorted(bad.items()))
        message = (
            f"{total} exported value(s) fall outside the range the external "
            f"driver accepts ({detail}). It rejects such commands instead of "
            "clipping, which freezes the robot at deployment."
        )
        if strict:
            shutil.rmtree(output_root)
            raise SystemExit("[export] " + message)
        print("[export] warning: " + message)

    transform_episode_files(
        source_root, output_root, episode_stats=episode_stats, camera_map=camera_map
    )
    source_stats = json.loads((source_root / "meta" / "stats.json").read_text())
    write_stats_json(
        source_stats, output_root, episode_stats=episode_stats, camera_map=camera_map
    )
    new_info = build_info(
        info,
        layout=layout,
        runtime=runtime,
        camera_map=camera_map,
        source_repo_id=source_repo_id,
        range_report=report,
        clip_tolerance_rad=clip_tolerance_rad,
    )
    (output_root / "meta" / "info.json").write_text(json.dumps(new_info, indent=4) + "\n")
    copy_side_files(source_root, output_root)
    videos = copy_videos(source_root, output_root, camera_map)
    return {
        "info": new_info,
        "episodes": len(episode_stats),
        "out_of_range": bad,
        "clipped": report["clipped"],
        "max_overshoot_rad": report["max_overshoot_rad"],
        "videos": videos,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    try:
        selection = resolve_dataset_selection(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    source_root = Path(selection.root)
    if not selection.local:
        parser.error("Export works on a local dataset directory.")
    if args.clip_tolerance_rad < 0.0:
        parser.error("--clip-tolerance-rad must be >= 0.")
    info = load_info(source_root)
    meta = require_canonical_dataset(info)
    layout = layout_for(target_robot(meta), args.layout, use_degrees=args.use_degrees)
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    camera_map = parse_camera_map(args.camera_map, layout=layout, video_keys=video_keys)
    output_root = args.output or source_root.parent / default_output_name(source_root.name, layout)

    print(
        "Export plan\n"
        f"  Source:     {source_root}\n"
        f"  Layout:     {layout.robot_type} <- {layout.robot}: {layout.describe()}\n"
        f"  Output:     {output_root}\n"
        f"  Cameras:    "
        + ", ".join(
            f"{o[len(VIDEO_PREFIX):]}->{n[len(VIDEO_PREFIX):]}" for o, n in camera_map.items()
        )
        + f"\n  Dropped:    {', '.join(DROPPED_COLUMNS)}"
    )
    if args.dry_run:
        return
    if output_root.exists():
        if not args.force:
            raise SystemExit(f"{output_root} exists; pass --force to replace it.")
        shutil.rmtree(output_root)

    result = export_dataset(
        source_root,
        output_root,
        layout=layout,
        camera_map=camera_map,
        source_repo_id=selection.repo_id,
        strict=args.strict,
        clip_tolerance_rad=None if args.no_clip else args.clip_tolerance_rad,
    )
    videos = result["videos"]
    print(
        f"[export] wrote {result['episodes']} episodes to {output_root} "
        f"(videos linked={videos['linked']} copied={videos['copied']})"
    )

    if args.compare_with is not None:
        reference = load_info(Path(args.compare_with))
        blocking, notes = schema_differences(result["info"], reference)
        for line in notes:
            print(f"[export] note: {line}")
        if blocking:
            print(f"[export] schema differs from {args.compare_with}:")
            for line in blocking:
                print(f"  - {line}")
            raise SystemExit(1)
        print(
            f"[export] schema matches {args.compare_with}: features, names, "
            "shapes, robot_type."
            + (f" ({len(notes)} resolution note(s) above)" if notes else "")
        )


if __name__ == "__main__":
    main()
