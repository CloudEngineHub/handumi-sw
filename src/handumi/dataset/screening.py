"""Embodiment-specific retargeting screening for recorded HandUMI datasets.

``handumi validate`` and ``handumi dataset analyze`` grade the *recording*:
tracking dropouts, gripper freezes, duration outliers. None of that knows
whether a given robot can actually follow the demonstration. This module runs
every episode through the exact solver ``handumi convert`` uses and grades the
result, so low-quality episodes are removed before they become joint targets.

Findings use the ``handumi validate`` quality-report schema, so the output
drops straight into ``handumi dataset analyze --quality-report`` and from there
into ``handumi dataset curate --exclude``.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk

from handumi.dataset.analysis import dataset_payload_manifest
from handumi.dataset.quality import EpisodeQualityReport, QualityFinding

SCREENING_SCHEMA_VERSION = 1
SCREENING_KIND = "handumi_retarget_screening"


def screening_report_path(root: str | Path, robot: str) -> Path:
    """Canonical per-embodiment screening report location."""
    return Path(root) / "meta" / f"handumi_screening_{robot}.json"


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robot_fingerprint(
    robot: str, *, deployment_path: str | Path | None = None
) -> dict[str, str]:
    """Hash everything that changes where the solver puts the robot.

    Screening numbers are only valid for one geometry: moving the arm bases or
    the table invalidates every joint solution in the report while leaving the
    dataset untouched, so the dataset manifest alone cannot detect it.

    Meshes are deliberately excluded -- they are large, and swapping one
    without editing the URDF is not a workflow this repo has.
    """
    from handumi.robots.registry import CONFIG_DIR, load_robot_config

    entries = {"robot_config": _file_digest(CONFIG_DIR / f"{robot}.yaml")}
    try:
        entries["urdf"] = _file_digest(Path(load_robot_config(robot).urdf))
    except (ValueError, TypeError, KeyError):
        entries["urdf"] = "unresolved"
    if deployment_path is not None:
        entries["deployment_calibration"] = _file_digest(Path(deployment_path))
    return entries


def solve_cache_path(root: str | Path, robot: str) -> Path:
    """Sidecar holding the trajectories screening already solved."""
    return Path(root) / "meta" / f"handumi_screening_{robot}_solves.npz"


# Everything the replay solver reads that can change the joints it produces.
# A cached trajectory is only reusable when all of it matches, so conversion
# falls back to solving rather than emitting joints from different settings.
_SOLVER_ARGS = (
    "source",
    "retarget_mode",
    "compose_source",
    "translation_scale",
    "controller_device",
    "controller_tcp_calibration",
    "use_dataset_tcp_calibration",
    "raw_controller_debug",
    "absolute_orientation",
    "initial_solve_iterations",
    "initial_position_tolerance_m",
    "gripper_max_width_m",
    "only_manipulation",
    "start_frame",
    "max_frames",
    "stride",
)

# Cached solves are keyed by these fields plus the robot fingerprint and the
# dataset manifest, both already recorded in the screening report.
CACHED_ROLLOUT_FIELDS = (
    "qpos",
    "gripper_normalized",
    "left_pos_error_m",
    "right_pos_error_m",
    "left_rot_error_deg",
    "right_rot_error_deg",
    "initial_solve_iterations",
    "retarget_mode",
    "gripper_source",
)


def solver_signature(args, *, deployment_path: str | Path | None = None) -> str:
    """Digest the solver settings a cached trajectory was produced under."""
    payload = {name: str(getattr(args, name, None)) for name in _SOLVER_ARGS}
    payload["deployment_calibration"] = str(
        deployment_path
        if deployment_path is not None
        else getattr(args, "deployment_calibration", None)
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RetargetScreeningConfig:
    """Thresholds for grading a retargeted episode.

    The position and rotation ceilings mirror ``handumi replay --strict-ik``.
    The rotation *outlier* rule is relative to the dataset's own distribution:
    a session recorded with a different wrist pose can stay far below the
    absolute ceiling while still being inconsistent with the rest of the
    dataset, which is exactly what would teach a policy two behaviors for one
    task. ``rotation_outlier_floor_deg`` keeps that rule quiet on datasets
    whose spread is uniformly small.

    The threshold is a multiple of the median rather than an IQR fence. A
    second recording session is not a sprinkle of outliers but a whole cluster,
    and once such a cluster is a sizeable minority it sits inside Q3, inflating
    the interquartile range until the fence catches nothing. The median stays
    inside the dominant cluster until contamination passes half the dataset,
    which is the point where no automatic rule can help anyway.
    """

    max_position_error_m: float = 0.03
    max_rotation_error_deg: float = 45.0
    rotation_median_multiplier: float = 3.0
    rotation_outlier_floor_deg: float = 10.0
    self_collision_margin_m: float = 0.01
    self_collision_min_frames: int = 1


def _replay_args(
    *,
    root: Path,
    repo_id: str,
    revision: str,
    robot: str,
    episode: int,
    deployment_profile: str,
    rig_config: Path,
):
    from handumi.scripts.replay.replay_in_sim import build_parser

    args = build_parser().parse_args([str(root)])
    args.repo_id = repo_id
    args.dataset_root = root
    args.revision = revision
    args.episode = episode
    args.robot = robot
    args.retarget_mode = "auto"
    args.deployment_profile = deployment_profile
    args.rig_config = rig_config
    # Grade every episode instead of aborting on the first violation, and skip
    # the playback lead-in: it is render-only and not part of the trajectory.
    args.strict_ik = False
    args.approach_seconds = 0.0
    return args


def _pedestal_columns(coll, robot, home_q: np.ndarray, plane) -> set[int]:
    """Capsules already below the tabletop at rest (mounts, pedestals)."""
    distances = np.asarray(
        coll.compute_world_collision_distance(robot, jnp.asarray(home_q), plane)
    ).reshape(-1)
    return set(np.flatnonzero(distances < 0.0).tolist())


def collision_meshes(runtime) -> dict[str, Any]:
    """Load each link's collision geometry as a convex hull in its link frame.

    Capsules enclose these meshes generously: a capsule fitted to a long or
    curved link bulges several centimeters past the geometry, so it can report
    deep interpenetration between parts that are comfortably apart. Capsule
    overlap alone therefore cannot decide whether a trajectory is executable,
    and is used only to select candidate frames for this check.

    Hulls rather than raw meshes because collision geometry is often not
    watertight, which makes signed distance -- the only way to measure
    penetration rather than surface distance -- unreliable on it. A hull is
    watertight by construction and contains its mesh, so it can over-report a
    contact but never miss one, which is the safe direction for a filter that
    discards data.
    """
    import trimesh

    urdf = runtime.load_urdf(load_meshes=True)
    pkg_root = Path(runtime.config.pkg_root)
    meshes: dict[str, Any] = {}
    for link in urdf.robot.links:
        candidates = [
            c
            for c in (link.collisions or [])
            if c.geometry is not None and c.geometry.mesh is not None
        ]
        if not candidates:
            continue
        collision = candidates[0]
        raw = collision.geometry.mesh.filename
        relative = (
            raw.split("://", 1)[1].split("/", 1)[1]
            if raw.startswith("package://")
            else raw
        )
        try:
            mesh = trimesh.load(pkg_root / relative, force="mesh")
        except (OSError, ValueError):
            continue
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if collision.origin is not None:
            mesh = mesh.copy()
            mesh.apply_transform(np.asarray(collision.origin))
        meshes[link.name] = mesh.convex_hull
    return meshes


def _link_transforms(runtime, q: np.ndarray) -> dict[str, np.ndarray]:
    import trimesh

    poses = np.asarray(runtime.robot.forward_kinematics(jnp.asarray(q)))
    transforms = {}
    for index, name in enumerate(runtime.robot.links.names):
        matrix = trimesh.transformations.quaternion_matrix(poses[index][:4])
        matrix[:3, 3] = poses[index][4:7]
        transforms[name] = matrix
    return transforms


def _mesh_separation(mesh_a, transform_a, mesh_b, transform_b, *, stride: int = 1) -> float:
    """Signed separation in meters: positive is clear, negative is penetration.

    ``signed_distance`` is positive inside a watertight mesh, so negating the
    deepest reading over both directions yields one number that is the surface
    gap when apart and the penetration depth when overlapping.
    """
    import trimesh

    a = mesh_a.copy()
    a.apply_transform(transform_a)
    b = mesh_b.copy()
    b.apply_transform(transform_b)
    deepest = max(
        float(trimesh.proximity.signed_distance(a, b.vertices[::stride]).max()),
        float(trimesh.proximity.signed_distance(b, a.vertices[::stride]).max()),
    )
    return -deepest


COLLISION_CHUNK_FRAMES = 128


def _chunked(fn, qpos: np.ndarray, chunk: int = COLLISION_CHUNK_FRAMES) -> np.ndarray:
    """Evaluate ``fn`` over frames in fixed-size blocks.

    Episodes have different lengths, so passing a whole episode makes XLA
    recompile per episode -- measured at ~3 s each against 5 ms once the shape
    is cached, which dominated the screening run. Padding to a constant block
    means one compilation for the entire dataset.
    """
    count = len(qpos)
    padding = (-count) % chunk
    if padding:
        qpos = np.concatenate([qpos, np.repeat(qpos[-1:], padding, axis=0)])
    blocks = [
        np.asarray(fn(jnp.asarray(qpos[start : start + chunk])))
        for start in range(0, len(qpos), chunk)
    ]
    return np.concatenate(blocks, axis=0)[:count]


def _collision_metrics(
    qpos: np.ndarray,
    *,
    runtime,
    coll,
    plane,
    pedestal: set[int],
    world_fn,
    self_fn,
    meshes: dict[str, Any],
    narrow_phase_frames: int = 8,
    vertex_stride: int = 1,
) -> dict[str, float | int]:
    """Capsule broad phase, then a convex-hull check on the frames it flags."""
    self_distances = _chunked(self_fn, qpos)
    world = _chunked(world_fn, qpos)
    columns = [c for c in range(world.shape[1]) if c not in pedestal]
    world = world[:, columns]

    link_names = list(coll.link_names)
    idx_i = np.asarray(coll.active_idx_i)
    idx_j = np.asarray(coll.active_idx_j)
    confirmed_frames: set[int] = set()
    mesh_min = float("inf")
    unverifiable = 0
    for pair in np.flatnonzero((self_distances < 0.0).any(axis=0)):
        a, b = link_names[idx_i[pair]], link_names[idx_j[pair]]
        candidates = np.flatnonzero(self_distances[:, pair] < 0.0)
        if a not in meshes or b not in meshes:
            # No collision geometry to confirm against: keep the capsule verdict
            # rather than silently clearing the pair.
            confirmed_frames.update(candidates.tolist())
            unverifiable += 1
            continue
        # Deepest capsule overlaps first: if those clear, the pair is clear.
        ordered = candidates[np.argsort(self_distances[candidates, pair])]
        probe = ordered[:narrow_phase_frames]
        separations = {}
        for frame in probe:
            transforms = _link_transforms(runtime, qpos[frame])
            separations[int(frame)] = _mesh_separation(
                meshes[a], transforms[a], meshes[b], transforms[b], stride=vertex_stride
            )
        mesh_min = min(mesh_min, min(separations.values()))
        touching = [f for f, d in separations.items() if d <= 0.0]
        if not touching:
            continue
        # Real contact exists, so price the whole pair exactly.
        for frame in ordered:
            if int(frame) in separations:
                if separations[int(frame)] <= 0.0:
                    confirmed_frames.add(int(frame))
                continue
            transforms = _link_transforms(runtime, qpos[frame])
            if (
                _mesh_separation(
                    meshes[a],
                    transforms[a],
                    meshes[b],
                    transforms[b],
                    stride=vertex_stride,
                )
                <= 0.0
            ):
                confirmed_frames.add(int(frame))

    return {
        "self_collision_min_clearance_m": (
            float(mesh_min) if np.isfinite(mesh_min) else float(self_distances.min())
        ),
        "self_collision_frames": len(confirmed_frames),
        "self_collision_capsule_min_m": float(self_distances.min()),
        "self_collision_capsule_frames": int((self_distances.min(axis=1) < 0.0).sum()),
        "self_collision_unverifiable_pairs": unverifiable,
        "table_min_clearance_m": float(world.min()),
        "table_penetration_frames": int((world.min(axis=1) < 0.0).sum()),
    }


def screen_dataset(
    root: str | Path,
    *,
    robot: str,
    repo_id: str | None = None,
    revision: str = "main",
    episodes: list[int] | tuple[int, ...] | None = None,
    deployment_profile: str = "auto",
    rig_config: Path | None = None,
    config: RetargetScreeningConfig | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Retarget every episode and grade it. Never modifies the dataset."""
    from handumi.calibration.deployment import resolve_deployment_calibration
    from handumi.config import DEFAULT_RIG_CONFIG
    from handumi.robots.registry import build_pruned_collision_model, load_embodiment
    from handumi.scripts.replay.replay_in_sim import solve_episode

    cfg = config or RetargetScreeningConfig()
    dataset_root = Path(root).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Dataset is missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info.get("total_episodes", 0))
    fps = float(info.get("fps", 0))
    if total_episodes <= 0 or fps <= 0:
        raise ValueError("Dataset info contains invalid episode or FPS totals")
    selected = (
        list(range(total_episodes)) if episodes is None else [int(e) for e in episodes]
    )
    out_of_range = [e for e in selected if e < 0 or e >= total_episodes]
    if out_of_range:
        raise ValueError(f"Episode indices out of range: {sorted(out_of_range)}")

    resolved_repo_id = repo_id or f"local/{dataset_root.name}"
    runtime = load_embodiment(robot)
    weights = runtime.config.ik_weights
    coll = build_pruned_collision_model(
        urdf=runtime.load_urdf(load_meshes=True),
        robot=runtime.robot,
        home_q=runtime.config.home_q.astype(np.float32),
        arms=runtime.arms,
        gripper_joints={
            side: tuple(g.name for g in runtime.config.arms[side].gripper_joints)
            for side in runtime.arms
        },
        margin=cfg.self_collision_margin_m,
    )
    plane_z = weights.world_collision_plane_z
    plane = pk.collision.HalfSpace.from_point_and_normal(
        jnp.array([0.0, 0.0, 1.0]) * plane_z, jnp.array([0.0, 0.0, 1.0])
    )
    # vmap over configurations: pyroki's world distance does not accept a
    # batch axis directly, and calling it per frame paid JAX dispatch 300+
    # times per episode.
    world_fn = jax.jit(
        jax.vmap(
            lambda q: coll.compute_world_collision_distance(
                runtime.robot, q, plane
            ).reshape(-1)
        )
    )
    self_fn = jax.jit(
        lambda q: coll.compute_self_collision_distance(runtime.robot, q)
    )
    pedestal = _pedestal_columns(
        coll, runtime.robot, runtime.config.home_q.astype(np.float32), plane
    )
    meshes = collision_meshes(runtime)

    deployment_path = str(
        resolve_deployment_calibration(
            robot,
            explicit_path=None,
            profile=deployment_profile,
            rig_config=rig_config or DEFAULT_RIG_CONFIG,
        ).path
    )
    raw: list[dict[str, Any]] = []
    solves: dict[str, np.ndarray] = {}
    signature: str | None = None
    for episode in selected:
        args = _replay_args(
            root=dataset_root,
            repo_id=resolved_repo_id,
            revision=revision,
            robot=robot,
            episode=episode,
            deployment_profile=deployment_profile,
            rig_config=rig_config or DEFAULT_RIG_CONFIG,
        )
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                rollout = solve_episode(args)
        except SystemExit as exc:
            # The solver could not reach the demonstrated start pose at all.
            raw.append(
                {
                    "episode_index": episode,
                    "unreachable": str(exc),
                    "frame_count": 0,
                    "metrics": {},
                }
            )
            if progress:
                print(f"  ep {episode:3d}  UNREACHABLE START POSE")
            continue

        if signature is None:
            # Sign what the solve settled on, not what was asked for: screening
            # requests "auto" and lets the device come from metadata, while
            # conversion names both outright. Keying on the request would never
            # match an identical solve.
            resolved = copy.copy(args)
            for field in ("retarget_mode", "controller_device"):
                value = rollout.get(field)
                if value is not None and len(value):
                    setattr(resolved, field, str(value[0]))
            signature = solver_signature(resolved, deployment_path=deployment_path)
        for field in CACHED_ROLLOUT_FIELDS:
            value = rollout.get(field)
            if value is not None:
                solves[f"{episode}/{field}"] = np.asarray(value)
        qpos = np.asarray(rollout["qpos"], dtype=np.float32)
        pos = np.concatenate(
            [rollout["left_pos_error_m"], rollout["right_pos_error_m"]]
        )
        rot = np.concatenate(
            [rollout["left_rot_error_deg"], rollout["right_rot_error_deg"]]
        )
        per_frame = np.maximum(
            rollout["left_pos_error_m"], rollout["right_pos_error_m"]
        )
        metrics: dict[str, float | int] = {
            "position_error_mean_m": float(pos.mean()),
            "position_error_max_m": float(pos.max()),
            # Sustained miss, robust to the settling spike right after the
            # start solve: a single frame at 3 cm while the rate limit catches
            # up is not the same failure as an arm that cannot reach at all.
            "position_error_p99_m": float(np.percentile(per_frame, 99)),
            "position_error_frames_over_1cm": int((per_frame > 0.01).sum()),
            "rotation_error_mean_deg": float(rot.mean()),
            "rotation_error_max_deg": float(rot.max()),
            "initial_position_error_m": float(
                rollout["initial_max_position_error_m"][0]
            ),
            "initial_solve_iterations": int(rollout["initial_solve_iterations"][0]),
        }
        metrics.update(
            _collision_metrics(
                qpos,
                runtime=runtime,
                coll=coll,
                plane=plane,
                pedestal=pedestal,
                world_fn=world_fn,
                self_fn=self_fn,
                meshes=meshes,
            )
        )
        raw.append(
            {
                "episode_index": episode,
                "unreachable": None,
                "frame_count": int(len(qpos)),
                "metrics": metrics,
            }
        )
        if progress:
            print(
                f"  ep {episode:3d}  pos {metrics['position_error_mean_m'] * 100:5.2f}"
                f"/{metrics['position_error_p99_m'] * 100:5.2f}"
                f"/{metrics['position_error_max_m'] * 100:5.2f}cm  "
                f"rot {metrics['rotation_error_mean_deg']:5.1f}"
                f"/{metrics['rotation_error_max_deg']:5.1f}deg  "
                f"self {metrics['self_collision_min_clearance_m'] * 100:5.2f}cm"
                f"({metrics['self_collision_frames']})"
            )

    fences = _rotation_fences(raw, cfg)
    reports = [_grade(item, cfg=cfg, fps=fps, fences=fences) for item in raw]
    accepted = sum(report.accepted for report in reports)
    flagged = sorted(
        report.episode_index
        for report in reports
        if any(f.severity == "reject" for f in report.findings)
    )
    review = sorted(
        report.episode_index
        for report in reports
        if report.accepted and report.findings
    )
    return {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "kind": SCREENING_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": resolved_repo_id,
        "root": str(dataset_root),
        "robot": robot,
        "config": asdict(cfg),
        "rotation_fences_deg": fences,
        "payload_manifest": dataset_payload_manifest(dataset_root),
        "deployment_calibration_path": deployment_path,
        "solver_signature": signature,
        "solve_cache": str(solve_cache_path(dataset_root, robot))
        if solves
        else None,
        "robot_fingerprint": robot_fingerprint(
            robot, deployment_path=deployment_path
        ),
        "summary": {
            "total": len(reports),
            "accepted": accepted,
            "rejected": len(reports) - accepted,
            "reject_episode_indices": flagged,
            "review_episode_indices": review,
        },
        "episodes": [report.to_dict() for report in reports],
        "_solves": solves,
    }


def _rotation_fences(
    raw: list[dict[str, Any]], cfg: RetargetScreeningConfig
) -> dict[str, float]:
    means = [
        float(item["metrics"]["rotation_error_mean_deg"])
        for item in raw
        if item["unreachable"] is None
    ]
    if len(means) < 4:
        # Too few episodes for a meaningful distribution; absolute ceiling only.
        return {}
    values = np.asarray(means)
    median = float(np.median(values))
    q1, q3 = (float(v) for v in np.quantile(values, [0.25, 0.75]))
    return {
        "median_deg": median,
        "q1_deg": q1,
        "q3_deg": q3,
        "upper_fence_deg": max(
            cfg.rotation_median_multiplier * median,
            cfg.rotation_outlier_floor_deg,
        ),
    }


def _grade(
    item: dict[str, Any],
    *,
    cfg: RetargetScreeningConfig,
    fps: float,
    fences: dict[str, float],
) -> EpisodeQualityReport:
    findings: list[QualityFinding] = []
    metrics = dict(item["metrics"])

    if item["unreachable"] is not None:
        findings.append(
            QualityFinding(
                code="retarget_start_pose_unreachable",
                severity="reject",
                message=(
                    "The robot cannot reach the demonstrated start pose; the "
                    "episode cannot be retargeted at all."
                ),
                metrics={"solver_message": str(item["unreachable"])},
            )
        )
        return EpisodeQualityReport(
            episode_index=int(item["episode_index"]),
            frame_count=int(item["frame_count"]),
            duration_s=0.0,
            findings=tuple(findings),
            metrics=metrics,
        )

    if metrics.get("position_error_p99_m", 0.0) > cfg.max_position_error_m:
        findings.append(
            QualityFinding(
                code="retarget_position_error",
                severity="reject",
                message=(
                    "The robot does not reach the demonstrated trajectory: the "
                    "position error stays above the ceiling, not just spikes."
                ),
                metrics={
                    "position_error_p99_m": metrics["position_error_p99_m"],
                    "position_error_frames_over_1cm": metrics.get(
                        "position_error_frames_over_1cm", 0
                    ),
                    "limit_m": cfg.max_position_error_m,
                },
            )
        )
    elif metrics["position_error_max_m"] > cfg.max_position_error_m:
        findings.append(
            QualityFinding(
                code="retarget_position_spike",
                severity="warning",
                message=(
                    "A brief position excursion above the ceiling, typically the "
                    "solver settling after the start pose rather than a "
                    "reachability failure."
                ),
                metrics={
                    "position_error_max_m": metrics["position_error_max_m"],
                    "position_error_p99_m": metrics.get("position_error_p99_m", 0.0),
                    "limit_m": cfg.max_position_error_m,
                },
            )
        )
    if metrics["rotation_error_max_deg"] > cfg.max_rotation_error_deg:
        findings.append(
            QualityFinding(
                code="retarget_rotation_error",
                severity="warning",
                message="Retargeted TCP orientation error exceeds the ceiling.",
                metrics={
                    "rotation_error_max_deg": metrics["rotation_error_max_deg"],
                    "limit_deg": cfg.max_rotation_error_deg,
                },
            )
        )
    upper = fences.get("upper_fence_deg")
    if upper is not None and metrics["rotation_error_mean_deg"] > upper:
        findings.append(
            QualityFinding(
                code="retarget_rotation_outlier",
                severity="warning",
                message=(
                    "Orientation tracking is inconsistent with the rest of the "
                    "dataset, which usually means a different recording session "
                    "or tool grip."
                ),
                metrics={
                    "rotation_error_mean_deg": metrics["rotation_error_mean_deg"],
                    "upper_fence_deg": upper,
                },
            )
        )
    if metrics["self_collision_frames"] >= cfg.self_collision_min_frames:
        findings.append(
            QualityFinding(
                code="retarget_self_collision",
                severity="warning",
                message="The retargeted trajectory intersects the robot itself.",
                metrics={
                    "self_collision_frames": metrics["self_collision_frames"],
                    "self_collision_min_clearance_m": metrics[
                        "self_collision_min_clearance_m"
                    ],
                },
            )
        )
    return EpisodeQualityReport(
        episode_index=int(item["episode_index"]),
        frame_count=int(item["frame_count"]),
        duration_s=float(item["frame_count"]) / fps if fps else 0.0,
        findings=tuple(findings),
        metrics=metrics,
    )


@dataclass(frozen=True)
class ScreeningGate:
    """Whether a dataset is cleared for joint conversion on one embodiment."""

    report_path: Path
    status: str  # "clear" | "missing" | "stale" | "flagged"
    flagged: dict[int, list[dict[str, Any]]]
    detail: str

    @property
    def blocks(self) -> bool:
        return self.status != "clear"


def evaluate_screening_gate(
    root: str | Path,
    *,
    robot: str,
    episodes: list[int] | tuple[int, ...] | None = None,
) -> ScreeningGate:
    """Check the screening report before a dataset becomes joint targets.

    Blocks on any finding, not only rejections: a warning means a human has
    not yet decided, and silently training on it is the failure this gate
    exists to prevent.
    """
    dataset_root = Path(root)
    path = screening_report_path(dataset_root, robot)
    if not path.is_file():
        return ScreeningGate(
            report_path=path,
            status="missing",
            flagged={},
            detail=f"No retargeting screening report for {robot!r} at {path}.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=f"Cannot read screening report {path}: {exc}",
        )
    if payload.get("payload_manifest") != dataset_payload_manifest(dataset_root):
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(
                f"Screening report {path} is stale: the dataset payload changed "
                "since it was written."
            ),
        )
    recorded = payload.get("robot_fingerprint")
    current = robot_fingerprint(
        robot, deployment_path=payload.get("deployment_calibration_path")
    )
    if recorded != current:
        changed = (
            sorted(
                key
                for key in set(recorded) | set(current)
                if recorded.get(key) != current.get(key)
            )
            if isinstance(recorded, dict)
            else ["the whole fingerprint"]
        )
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(
                f"Screening report {path} is stale: {', '.join(changed)} changed "
                "since it was written, so every joint solution in it is invalid."
            ),
        )

    screened = {int(item["episode_index"]) for item in payload.get("episodes", ())}
    if episodes is None:
        # Coverage is judged against the dataset, not against the report: a
        # report written with --episodes must not look complete.
        info_path = dataset_root / "meta" / "info.json"
        try:
            total = int(
                json.loads(info_path.read_text(encoding="utf-8")).get(
                    "total_episodes", 0
                )
            )
        except (OSError, json.JSONDecodeError, ValueError):
            total = 0
        wanted = set(range(total)) if total > 0 else set(screened)
    else:
        wanted = {int(e) for e in episodes}
    missing = sorted(wanted - screened)
    if missing:
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(
                f"Screening report {path} does not cover episodes {missing}; "
                "re-run handumi dataset screen without --episodes."
            ),
        )

    flagged = {
        int(item["episode_index"]): list(item.get("findings", ()))
        for item in payload.get("episodes", ())
        if item.get("findings")
        and (wanted is None or int(item["episode_index"]) in wanted)
    }
    if flagged:
        return ScreeningGate(
            report_path=path,
            status="flagged",
            flagged=flagged,
            detail=(
                f"{len(flagged)} episode(s) still carry screening findings for "
                f"{robot!r}."
            ),
        )
    return ScreeningGate(
        report_path=path, status="clear", flagged={}, detail="Screening is clear."
    )


def format_gate_guidance(gate: ScreeningGate, *, root: Path, robot: str) -> str:
    """Explain how to satisfy the gate, with runnable commands."""
    lines = [gate.detail]
    if gate.status in {"missing", "stale"}:
        lines += [
            "",
            "Run the retargeting screening first:",
            f"  handumi dataset screen {root} --robot {robot}",
        ]
    else:
        rejects = sorted(
            index
            for index, findings in gate.flagged.items()
            if any(f.get("severity") == "reject" for f in findings)
        )
        warnings = sorted(set(gate.flagged) - set(rejects))
        for index in sorted(gate.flagged):
            codes = ", ".join(
                f"{f.get('severity')}:{f.get('code')}" for f in gate.flagged[index]
            )
            lines.append(f"  episode {index}: {codes}")
        if rejects:
            lines.append(f"Unusable on this robot: {rejects}")
        if warnings:
            lines.append(f"Needs your judgement: {warnings}")
        excluded = ",".join(str(i) for i in sorted(gate.flagged))
        lines += [
            "",
            "Review them, then remove the ones you reject:",
            f"  handumi dataset analyze {root} --quality-report {gate.report_path}",
            f"  handumi dataset curate {root} --output <new_root> "
            f"--exclude {excluded}",
        ]
    lines += [
        "",
        "To convert anyway, pass --allow-flagged-episodes.",
    ]
    return "\n".join(lines)


def write_screening_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the report, and the solved trajectories beside it.

    The trajectories live in a separate npz because they are arrays, not
    findings: the report stays a readable audit document, and conversion can
    reuse the solve instead of repeating it.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    solves = payload.pop("_solves", None)
    if solves:
        cache = payload.get("solve_cache")
        np.savez_compressed(
            Path(cache) if cache else output.with_name(output.stem + "_solves.npz"),
            **solves,
        )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_cached_solve(
    root: str | Path,
    *,
    robot: str,
    episode: int,
    signature: str,
) -> dict[str, np.ndarray] | None:
    """Return a previously solved rollout, or None when it cannot be trusted.

    Freshness of the dataset and of the robot geometry is already enforced by
    the screening gate, so this only has to confirm the solver settings match
    and that this episode is present.
    """
    report_path = screening_report_path(root, robot)
    cache_path = solve_cache_path(root, robot)
    if not report_path.is_file() or not cache_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("solver_signature") != signature:
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            prefix = f"{episode}/"
            keys = [name for name in archive.files if name.startswith(prefix)]
            if not any(name == f"{prefix}qpos" for name in keys):
                return None
            return {name[len(prefix) :]: archive[name] for name in keys}
    except (OSError, ValueError):
        return None


def render_screening_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# Retargeting screening: {payload['robot']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Episodes: {summary['total']} "
        f"({summary['accepted']} accepted, {summary['rejected']} rejected)",
    ]
    fences = payload.get("rotation_fences_deg") or {}
    if fences:
        lines.append(
            f"- Rotation outlier threshold: {fences['upper_fence_deg']:.1f} deg "
            f"(median {fences['median_deg']:.1f}, "
            f"Q1 {fences['q1_deg']:.1f}, Q3 {fences['q3_deg']:.1f})"
        )
    lines += [
        "",
        "| ep | frames | pos mean/max (cm) | rot mean/max (deg) | self (cm/frames) "
        "| table (cm/frames) | status |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for episode in payload["episodes"]:
        m = episode.get("metrics") or {}
        if not m:
            lines.append(
                f"| {episode['episode_index']} | - | - | - | - | - | "
                f"**{episode['status']}** |"
            )
            continue
        lines.append(
            f"| {episode['episode_index']} | {episode['frame_count']} "
            f"| {m['position_error_mean_m'] * 100:.2f} / "
            f"{m['position_error_max_m'] * 100:.2f} "
            f"| {m['rotation_error_mean_deg']:.1f} / "
            f"{m['rotation_error_max_deg']:.1f} "
            f"| {m['self_collision_min_clearance_m'] * 100:.2f} / "
            f"{m['self_collision_frames']} "
            f"| {m['table_min_clearance_m'] * 100:.2f} / "
            f"{m['table_penetration_frames']} "
            f"| {episode['status']} |"
        )
    findings = [
        (episode["episode_index"], finding)
        for episode in payload["episodes"]
        for finding in episode.get("findings", ())
    ]
    if findings:
        lines += ["", "## Findings", ""]
        for index, finding in findings:
            lines.append(
                f"- episode {index}: **{finding['severity']}** "
                f"`{finding['code']}` -- {finding['message']}"
            )
    return "\n".join(lines) + "\n"
