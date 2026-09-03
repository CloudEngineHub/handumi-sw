"""``handumi replay-real`` streams stored joints to a hardware backend.

No hardware is involved here: the plan is checked against the robot limits on
synthetic datasets (canonical and exported layouts), and the hardware run is
exercised through a fake backend that records every call and echoes the
commanded joints back as feedback.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from handumi.dataset.canonical import CANONICAL_STATE_LAYOUT
from handumi.dataset.external_layouts import BI_PIPER_FOLLOWER
from handumi.dataset.writer import EpisodeResult, write_dataset
from handumi.robots.registry import load_embodiment
from handumi.scripts.export_dataset import export_dataset
from handumi.scripts.replay import replay_in_real as rr

FPS = 30
JOINT_NAMES = [
    *(f"left_joint{i}.pos" for i in range(1, 7)),
    "left_gripper.width_m",
    *(f"right_joint{i}.pos" for i in range(1, 7)),
    "right_gripper.width_m",
]


@pytest.fixture(scope="module")
def runtime():
    return load_embodiment("piper")


def _smooth_episode(runtime, index: int, frames: int, *, spike_frame: int | None = None) -> EpisodeResult:
    """A slow sinusoid inside the Piper limits, optionally with one fast frame."""
    lower = np.array([e.lower_rad for e in BI_PIPER_FOLLOWER.arm_encodings])
    upper = np.array([e.upper_rad for e in BI_PIPER_FOLLOWER.arm_encodings])
    center = (lower + upper) / 2.0
    amplitude = 0.15 * (upper - lower) / 2.0
    t = np.arange(frames + 1) / FPS
    joints = np.zeros((frames + 1, 14), dtype=np.float32)
    for base in (0, 7):
        phase = np.sin(2.0 * np.pi * t / 2.0)[:, None]
        joints[:, base : base + 6] = center + amplitude * phase
        joints[:, base + 6] = 0.5 * runtime.config.gripper_max_width_m * (0.5 + 0.5 * phase[:, 0])
    if spike_frame is not None:
        joints[spike_frame, 2] += 0.2
    return EpisodeResult(
        episode_index=index,
        states=joints[:-1],
        actions=joints[1:],
        task="wave",
        calibration_id=0,
        source_kind=0,
    )


def _write(root: Path, runtime, episodes: list[EpisodeResult]) -> Path:
    source = root.parent / f"{root.name}-raw"
    (source / "meta").mkdir(parents=True, exist_ok=True)
    write_dataset(
        output_root=root,
        source_root=source,
        source_info={"features": {}, "fps": FPS},
        episodes=episodes,
        robot_type="piper",
        joint_names=JOINT_NAMES,
        fps=FPS,
        handumi_metadata={
            "state_layout": CANONICAL_STATE_LAYOUT,
            "target_robot": {"name": "piper"},
        },
    )
    return root


@pytest.fixture
def canonical_dataset(tmp_path: Path, runtime) -> Path:
    return _write(
        tmp_path / "wave-piper-joints",
        runtime,
        [_smooth_episode(runtime, 0, 12), _smooth_episode(runtime, 1, 9)],
    )


@pytest.fixture
def exported_dataset(canonical_dataset: Path) -> Path:
    output = canonical_dataset.parent / "wave-bi_piper_follower"
    export_dataset(
        canonical_dataset, output, layout=BI_PIPER_FOLLOWER, camera_map={},
        source_repo_id="local/wave", strict=True,
    )
    return output


def _args(root: Path, *extra: str):
    rig = root.parent / "missing-rig.yaml"
    return rr.parse_args([str(root), "--rig-config", str(rig), *extra])


def test_plan_decodes_exported_and_canonical_identically(canonical_dataset, exported_dataset) -> None:
    canonical = rr.build_plan(_args(canonical_dataset, "--episode", "0", "1"))
    exported = rr.build_plan(_args(exported_dataset, "--episode", "0", "1"))
    assert canonical.robot == exported.robot == "piper"
    assert exported.external is not None and canonical.external is None
    assert [e.episode for e in exported.episodes] == [0, 1]
    for a, b in zip(canonical.episodes, exported.episodes, strict=True):
        np.testing.assert_allclose(a.qpos, b.qpos, atol=1e-5)
        np.testing.assert_allclose(a.openings, b.openings, atol=1e-5)
    assert exported.required_speed_deg_s < exported.joint_speed_limit_deg_s
    assert exported.notes == ()
    text = rr.describe_plan(exported, _args(exported_dataset, "--episode", "0", "1"))
    assert "lerobot_bi_piper_follower" in text and "Episodes: 0, 1" in text
    assert "Table placement: not recorded" in text
    sim_info = {"handumi": {"deployment_calibration": {
        "profile": "sim", "scope": "simulation", "verified": False,
        "path": "configs/calibration/table/sim/piper.yaml"}}}
    placement = "\n".join(rr.describe_table_placement(sim_info))
    assert "profile sim (unverified" in placement and "x=0.45 m forward" in placement
    assert "not a measured installation" in placement


def test_plan_refuses_a_robot_the_dataset_was_not_converted_for(exported_dataset) -> None:
    with pytest.raises(SystemExit, match="cannot drive 'openarmv1'"):
        rr.build_plan(_args(exported_dataset, "--robot", "openarmv1"))


def test_inactive_side_is_parked_at_home(canonical_dataset, runtime) -> None:
    plan = rr.build_plan(_args(canonical_dataset, "--side", "left"))
    episode = plan.episodes[0]
    right = runtime.arm_joint_indices("right")
    np.testing.assert_array_equal(episode.qpos[:, right], np.tile(plan.home_q[right], (episode.frames, 1)))
    assert np.all(episode.openings[:, 1] == 0.0)
    assert np.any(episode.openings[:, 0] > 0.0)


def test_speed_spike_is_a_note_and_sustained_overrun_aborts(tmp_path: Path, runtime) -> None:
    spiky = _write(tmp_path / "spiky-piper-joints", runtime, [_smooth_episode(runtime, 0, 90, spike_frame=10)])
    plan = rr.build_plan(_args(spiky))
    assert "left_joint3 at frame 10" in plan.notes[0]
    assert plan.required_speed_deg_s > plan.joint_speed_limit_deg_s

    fast = _smooth_episode(runtime, 0, 30)
    fast.states[::2, 2] += 0.15  # every other frame jumps 0.15 rad on one joint
    fast_root = _write(tmp_path / "fast-piper-joints", runtime, [fast])
    with pytest.raises(SystemExit, match="Rerun with --speed"):
        rr.build_plan(_args(fast_root))
    slowed = rr.build_plan(_args(fast_root, "--speed", "0.3"))
    assert slowed.required_speed_deg_s < slowed.joint_speed_limit_deg_s


def test_stream_prediction_reflects_the_backend_limits(canonical_dataset, runtime) -> None:
    plan = rr.build_plan(_args(canonical_dataset))
    assert plan.joint_acceleration_limit_deg_s2 == runtime.config.real.max_joint_acceleration_deg_s2
    # A 40 deg/s sinusoid lags the 720 deg/s^2 envelope by about v^2/2a = 1.1 deg.
    assert all(p.max_deg < 2.0 for p in plan.predictions), plan.predictions

    # A fast sinusoid: the braking envelope lags it by about v^2 / 2a, so a
    # higher acceleration limit must predict a smaller error.
    t = np.arange(0.0, 3.0, 1.0 / FPS)
    q = np.tile(np.asarray(runtime.config.home_q, dtype=np.float32), (len(t), 1))
    q[:, 2] += np.deg2rad(20.0) * np.sin(2.0 * np.pi * t)  # peak 126 deg/s
    indices = rr.motion_joint_indices(runtime, ("left", "right"))
    slow = rr.predict_tracking(q, FPS, indices, max_velocity_deg_s=180.0, max_acceleration_deg_s2=720.0)
    fast = rr.predict_tracking(q, FPS, indices, max_velocity_deg_s=180.0, max_acceleration_deg_s2=5760.0)
    rate_limited = rr.predict_tracking(q, FPS, indices, max_velocity_deg_s=180.0, max_acceleration_deg_s2=None)
    assert slow.max_deg > 3.0 > fast.max_deg
    assert rate_limited.max_deg < 1.0  # a pure rate limiter follows anything below its rate

    override = rr.build_plan(_args(canonical_dataset, "--accel", "5760"))
    assert override.joint_acceleration_limit_deg_s2 == 5760.0
    assert rr.runtime_with_acceleration(runtime, 5760.0).config.real.max_joint_acceleration_deg_s2 == 5760.0
    assert runtime.config.real.max_joint_acceleration_deg_s2 != 5760.0


def test_clip_to_joint_limits_only_absorbs_export_overshoot(runtime) -> None:
    lower = np.asarray(runtime.robot.joints.lower_limits, dtype=np.float32)
    q = np.tile(np.asarray(runtime.config.home_q, dtype=np.float32), (3, 1))
    q[1, 0] = lower[0] - 1e-3
    clipped, count, worst = rr.clip_to_joint_limits(q, runtime)
    assert count == 1 and worst == pytest.approx(1e-3, abs=1e-6)
    assert clipped[1, 0] == lower[0]
    q[2, 0] = lower[0] - 0.05
    with pytest.raises(SystemExit, match="outside its URDF limit"):
        rr.clip_to_joint_limits(q, runtime)


def test_estimate_tracking_lag_recovers_a_pure_delay() -> None:
    command_time = np.arange(0.0, 4.0, 1.0 / FPS)
    commands = np.stack([np.sin(command_time), np.cos(0.7 * command_time)], axis=1)
    sample_time = np.arange(0.5, 4.5, 0.011)
    delayed = np.stack([np.sin(sample_time - 0.06), np.cos(0.7 * (sample_time - 0.06))], axis=1)
    lag = rr.estimate_tracking_lag(command_time, commands, sample_time, delayed)
    assert lag == pytest.approx(0.06, abs=rr.TRACKING_LAG_STEP_S)


class FakeBackend:
    """Records the teleop-backend protocol calls and echoes commands back."""

    name = "fake"

    def __init__(self, *, fail_after_health_checks: int | None = None) -> None:
        self.calls: list[str] = []
        self.written: list[tuple[np.ndarray, dict[str, float]]] = []
        self.active_sides = ("left", "right")
        self._lock = threading.Lock()
        self._fail_after = fail_after_health_checks
        self._health_checks = 0

    def setup(self, *, repair: bool = True) -> None:
        self.calls.append("setup")

    def connect(self) -> None:
        self.calls.append("connect")

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def read(self, base_q=None):
        with self._lock:
            if self.written:
                return self.written[-1][0].copy()
        return np.asarray(base_q, dtype=np.float32).copy()

    def read_gripper_openings(self) -> dict[str, float]:
        with self._lock:
            return dict(self.written[-1][1]) if self.written else {}

    def home(self, q) -> None:
        self.calls.append("home")

    def move_home(self, q) -> None:
        self.calls.append("move_home")

    def write(self, q, gripper_openings) -> None:
        with self._lock:
            self.written.append((np.asarray(q, dtype=np.float32).copy(), dict(gripper_openings)))

    def hold(self, base_q):
        self.calls.append("hold")
        return np.asarray(base_q, dtype=np.float32).copy()

    def check_health(self) -> None:
        self._health_checks += 1
        if self._fail_after is not None and self._health_checks > self._fail_after:
            raise RuntimeError("CAN bus went away")


def test_run_plan_streams_every_frame_and_reports_tracking(monkeypatch, tmp_path, exported_dataset) -> None:
    backend = FakeBackend()
    seen: dict[str, object] = {}

    def fake_backend(robot, *, runtime, rig_config, active_sides):
        seen["accel"] = runtime.config.real.max_joint_acceleration_deg_s2
        return backend

    monkeypatch.setattr(rr, "make_real_backend", fake_backend)
    args = _args(
        exported_dataset, "--yes", "--approach-seconds", "1.0",
        "--output-dir", str(tmp_path / "logs"), "--accel", "2880",
    )
    plan = rr.build_plan(args)
    reports = rr.run_plan(plan, args)
    assert seen["accel"] == 2880.0

    assert backend.calls == ["setup", "connect", "home", "move_home", "disconnect"]
    assert len(reports) == 1 and reports[0].passed, reports[0].reasons
    assert reports[0].lag_s <= 0.1
    assert reports[0].gripper_max_pct is not None and reports[0].gripper_max_pct < 5.0
    # The last command the stream published is the last frame of the episode.
    np.testing.assert_allclose(backend.written[-1][0], plan.episodes[0].qpos[-1], atol=1e-5)
    npz = tmp_path / "logs" / "episode_000000_piper.npz"
    payload = np.load(npz)
    assert payload["commanded_qpos"].shape == plan.episodes[0].qpos.shape
    assert len(payload["measured_time_s"]) == reports[0].samples or len(payload["measured_time_s"]) > 0
    summary = json.loads(npz.with_suffix(".json").read_text())
    assert summary["passed"] is True and summary["episode"] == 0


def test_run_plan_holds_and_disables_on_a_backend_fault(monkeypatch, tmp_path, canonical_dataset) -> None:
    backend = FakeBackend(fail_after_health_checks=3)
    monkeypatch.setattr(rr, "make_real_backend", lambda *a, **k: backend)
    args = _args(canonical_dataset, "--yes", "--approach-seconds", "1.0", "--output-dir", str(tmp_path / "logs"))
    plan = rr.build_plan(args)
    with pytest.raises(SystemExit, match="did not complete"):
        rr.run_plan(plan, args)
    assert backend.calls[:3] == ["setup", "connect", "home"]
    assert "hold" in backend.calls and "move_home" not in backend.calls
    assert backend.calls[-1] == "disconnect"


def test_parse_args_rejects_unsafe_values(canonical_dataset) -> None:
    for extra, message in (
        (("--speed", "1.5"), "--speed"),
        (("--episode", "0", "0"), "twice"),
        (("--approach-seconds", "-1"), "--approach-seconds"),
    ):
        with pytest.raises(SystemExit):
            _args(canonical_dataset, *extra)
    with pytest.raises(SystemExit, match="cannot drive"):
        rr.build_plan(_args(canonical_dataset, "--robot", "openarmv1"))


def test_local_dataset_does_not_fetch_from_the_hub(monkeypatch, canonical_dataset) -> None:
    monkeypatch.setattr(
        rr,
        "ensure_metadata",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("local roots must not fetch")),
    )
    args = _args(canonical_dataset)
    assert args.dataset_root == canonical_dataset
    assert args.repo_id == f"local/{canonical_dataset.name}"


def test_hub_id_downloads_into_the_local_cache(monkeypatch, tmp_path, canonical_dataset) -> None:
    cache = tmp_path / "outputs" / "datasets"
    fetched: list[str] = []

    def fake_root(repo_id: str) -> Path:
        return cache / repo_id.rstrip("/").split("/")[-1]

    def fake_ensure(*, repo_id, root, revision):
        fetched.append(repo_id)
        dest = Path(root)
        shutil.copytree(canonical_dataset, dest, dirs_exist_ok=True)
        return json.loads((dest / "meta" / "info.json").read_text())

    monkeypatch.setattr("handumi.dataset.selection.dataset_root_from_repo_id", fake_root)
    monkeypatch.setattr(rr, "ensure_metadata", fake_ensure)
    args = rr.parse_args(
        [
            "murobotics/tblock-all-piper-clean-bi_piper_follower",
            "--rig-config",
            str(tmp_path / "missing-rig.yaml"),
            "--episode",
            "0",
        ]
    )
    assert fetched == ["murobotics/tblock-all-piper-clean-bi_piper_follower"]
    assert args.repo_id == "murobotics/tblock-all-piper-clean-bi_piper_follower"
    assert args.dataset_root == cache / "tblock-all-piper-clean-bi_piper_follower"
    assert (args.dataset_root / "meta" / "info.json").is_file()
    assert args.robot == "piper"
