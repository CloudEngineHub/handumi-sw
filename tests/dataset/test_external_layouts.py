"""External joint layouts must round-trip and must describe the URDF they claim.

Nothing in an external dataset says which limits and signs produced its
numbers, so the constants here are the only thing standing between a correct
export and a mirrored, rescaled robot. These tests pin them against the
embodiment's own model and, when the reference capture is on disk, against
real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from handumi.dataset.canonical import (
    CANONICAL_STATE_LAYOUT,
    expand_canonical_trajectory,
)
from handumi.dataset.external_layouts import (
    BI_OPENARM_FOLLOWER,
    BI_PIPER_FOLLOWER,
    LIMIT_TOLERANCE_RAD,
    check_layout_limits,
    clip_to_driver_range,
    detect_external_layout,
    external_layout_for_name,
    from_canonical,
    out_of_range_counts,
    to_canonical,
)
from handumi.robots.registry import load_embodiment

REFERENCE_ROOT = Path("outputs/datasets/bi_piper_pick_and_place_fruits_mantra")


@pytest.fixture(scope="module")
def runtime():
    return load_embodiment(BI_PIPER_FOLLOWER.robot)


def _piper_limits() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([e.lower_rad for e in BI_PIPER_FOLLOWER.arm_encodings])
    upper = np.array([e.upper_rad for e in BI_PIPER_FOLLOWER.arm_encodings])
    return lower, upper


def _random_canonical(runtime, frames: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower, upper = _piper_limits()
    canonical = np.zeros((frames, BI_PIPER_FOLLOWER.size), dtype=np.float32)
    for base in (0, 7):
        canonical[:, base : base + 6] = rng.uniform(lower, upper, size=(frames, 6))
        canonical[:, base + 6] = rng.uniform(
            0.0, runtime.config.gripper_max_width_m, size=frames
        )
    return canonical


def test_layout_names_follow_the_external_stack() -> None:
    names = BI_PIPER_FOLLOWER.names
    assert len(names) == BI_PIPER_FOLLOWER.size == 14
    assert names[0] == "left_shoulder_pan.pos"
    assert names[6] == "left_gripper.pos"
    assert names[7] == "right_shoulder_pan.pos"
    assert names[13] == "right_gripper.pos"
    assert external_layout_for_name("bi_piper_follower") is BI_PIPER_FOLLOWER
    assert external_layout_for_name("bi_openarm_follower") is BI_OPENARM_FOLLOWER
    with pytest.raises(ValueError, match="Unknown external layout"):
        external_layout_for_name("nope")


def test_layout_limits_are_the_urdf_limits(runtime) -> None:
    check_layout_limits(BI_PIPER_FOLLOWER, runtime)
    lower, upper = _piper_limits()
    urdf_lower = np.asarray(runtime.robot.joints.lower_limits)[:6]
    urdf_upper = np.asarray(runtime.robot.joints.upper_limits)[:6]
    assert np.abs(lower - urdf_lower).max() < LIMIT_TOLERANCE_RAD
    assert np.abs(upper - urdf_upper).max() < LIMIT_TOLERANCE_RAD


def test_canonical_round_trip(runtime) -> None:
    canonical = _random_canonical(runtime)
    external = from_canonical(canonical, layout=BI_PIPER_FOLLOWER, runtime=runtime)
    assert external.dtype == np.float32 and external.shape == canonical.shape
    # Every arm column lands in the driver's accepted range, gripper in 0..100.
    assert external[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].min() >= -100.0 - 1e-3
    assert external[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].max() <= 100.0 + 1e-3
    assert external[:, [6, 13]].min() >= 0.0 and external[:, [6, 13]].max() <= 100.0 + 1e-3
    restored = to_canonical(external, layout=BI_PIPER_FOLLOWER, runtime=runtime)
    np.testing.assert_allclose(restored, canonical, atol=1e-5)


def test_external_round_trip(runtime) -> None:
    rng = np.random.default_rng(1)
    external = rng.uniform(-100.0, 100.0, size=(32, 14)).astype(np.float32)
    external[:, [6, 13]] = rng.uniform(0.0, 100.0, size=(32, 2))
    canonical = to_canonical(external, layout=BI_PIPER_FOLLOWER, runtime=runtime)
    back = from_canonical(canonical, layout=BI_PIPER_FOLLOWER, runtime=runtime)
    np.testing.assert_allclose(back, external, atol=1e-3)


def test_signs_flip_only_the_declared_joints(runtime) -> None:
    # A positive URDF angle on a flipped joint must come out below the
    # normalized midpoint, and above it on an unflipped one.
    canonical = np.zeros((1, 14), dtype=np.float32)
    canonical[0, 0] = 0.5  # left joint1: sign -1
    canonical[0, 4] = 0.5  # left joint5: sign +1
    external = from_canonical(canonical, layout=BI_PIPER_FOLLOWER, runtime=runtime)
    assert external[0, 0] < 0.0
    assert external[0, 4] > 0.0


def test_out_of_range_counts_by_column() -> None:
    external = np.zeros((4, 14), dtype=np.float32)
    external[0, 1] = 101.0
    external[1, 6] = -0.5
    external[2, 13] = 100.5
    counts = out_of_range_counts(external, layout=BI_PIPER_FOLLOWER)
    assert counts == {
        "left_shoulder_lift.pos": 1,
        "left_gripper.pos": 1,
        "right_gripper.pos": 1,
    }


def test_clip_folds_back_solver_noise_but_not_real_violations() -> None:
    external = np.zeros((3, 14), dtype=np.float32)
    external[0, 2] = 100.05   # elbow: 0.05 units = 7.4e-4 rad past the limit -> noise
    external[1, 2] = 103.0    # 4.5e-2 rad past the limit -> a real violation
    external[2, 6] = -1e-5    # gripper float round-off at closed
    clipped, counts, worst = clip_to_driver_range(
        external, layout=BI_PIPER_FOLLOWER, tolerance_rad=2e-3
    )
    assert clipped[0, 2] == 100.0 and clipped[2, 6] == 0.0
    assert clipped[1, 2] == pytest.approx(103.0)
    assert counts == {"left_elbow_flex.pos": 1, "left_gripper.pos": 1}
    assert worst == pytest.approx(3.0 * 2.967 / 200.0, rel=1e-3)
    # Untouched input, and nothing changes when nothing overshoots.
    assert external[0, 2] == pytest.approx(100.05)
    same, none, zero = clip_to_driver_range(
        np.zeros((2, 14), np.float32), layout=BI_PIPER_FOLLOWER, tolerance_rad=2e-3
    )
    assert none == {} and zero == 0.0 and not same.any()


def test_detect_external_layout_from_metadata() -> None:
    assert detect_external_layout({"robot_type": "bi_piper_follower"}) is BI_PIPER_FOLLOWER
    assert (
        detect_external_layout({"robot_type": "piper", "handumi": {"state_layout": "lerobot_bi_piper_follower"}})
        is BI_PIPER_FOLLOWER
    )
    canonical = {"robot_type": "piper", "handumi": {"state_layout": CANONICAL_STATE_LAYOUT}}
    assert detect_external_layout(canonical) is None
    assert detect_external_layout({"robot_type": "handumi_raw"}) is None


@pytest.mark.skipif(
    not (REFERENCE_ROOT / "meta" / "info.json").is_file(),
    reason="reference XHUMAN capture is not downloaded",
)
def test_reference_capture_denormalizes_into_the_urdf(runtime) -> None:
    import glob

    import pyarrow.parquet as pq

    path = sorted(glob.glob(str(REFERENCE_ROOT / "data" / "**" / "*.parquet"), recursive=True))[0]
    table = pq.read_table(path)
    mask = table.column("episode_index").to_numpy() == 0
    states = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))[mask]
    canonical = to_canonical(states.astype(np.float32), layout=BI_PIPER_FOLLOWER, runtime=runtime)
    qpos, _ = expand_canonical_trajectory(canonical, runtime=runtime)
    lower = np.asarray(runtime.robot.joints.lower_limits)
    upper = np.asarray(runtime.robot.joints.upper_limits)
    assert int(((qpos < lower - 1e-6) | (qpos > upper + 1e-6)).sum()) == 0

    # Wrong signs on joint 1 would mirror each arm across the robot's midline;
    # a real rest pose keeps left at +y and right at -y, both in front.
    solver = runtime.solver_cls()
    left, right = solver.fk_pose7(qpos[0])
    assert left[0] > 0.0 and right[0] > 0.0
    assert left[1] > 0.1 and right[1] < -0.1


@pytest.fixture(scope="module")
def openarm_runtime():
    return load_embodiment(BI_OPENARM_FOLLOWER.robot)


def test_openarm_layout_is_degrees_like_lerobot(openarm_runtime) -> None:
    layout = BI_OPENARM_FOLLOWER
    assert layout.size == 16 and layout.names[0] == "left_joint_1.pos"
    assert layout.names[7] == "left_gripper.pos" and layout.names[15] == "right_gripper.pos"
    check_layout_limits(layout, openarm_runtime)  # nothing ranged to check, must not raise

    canonical = np.zeros((1, 16), dtype=np.float32)
    canonical[0, 0] = np.pi / 2          # left joint_1 = 90 deg
    canonical[0, 7] = 0.088              # left gripper fully open
    canonical[0, 15] = 0.0               # right gripper closed
    external = from_canonical(canonical, layout=layout, runtime=openarm_runtime)
    assert external[0, 0] == pytest.approx(90.0, abs=1e-4)
    assert external[0, 7] == pytest.approx(-60.0, abs=1e-3)   # open = -60 deg on the motor
    assert external[0, 15] == pytest.approx(0.0, abs=1e-6)
    # Degrees carry no driver range: nothing is ever reported or clipped.
    assert out_of_range_counts(external * 10.0, layout=layout) == {}
    same, counts, worst = clip_to_driver_range(external * 10.0, layout=layout, tolerance_rad=1e-3)
    assert counts == {} and worst == 0.0 and np.array_equal(same, (external * 10.0).astype(np.float32))


def test_openarm_round_trip(openarm_runtime) -> None:
    rng = np.random.default_rng(3)
    lower = np.asarray(openarm_runtime.robot.joints.lower_limits)
    upper = np.asarray(openarm_runtime.robot.joints.upper_limits)
    canonical = np.zeros((40, 16), dtype=np.float32)
    for base in (0, 8):
        canonical[:, base : base + 7] = rng.uniform(lower[:7], upper[:7], size=(40, 7))
        canonical[:, base + 7] = rng.uniform(0.0, 0.088, size=40)
    external = from_canonical(canonical, layout=BI_OPENARM_FOLLOWER, runtime=openarm_runtime)
    restored = to_canonical(external, layout=BI_OPENARM_FOLLOWER, runtime=openarm_runtime)
    np.testing.assert_allclose(restored, canonical, atol=1e-5)


def test_use_degrees_mirrors_the_plugin_option(runtime) -> None:
    from dataclasses import replace

    from handumi.dataset.external_layouts import DEGREES, RANGE_M100_100

    # XHUMAN's Piper plugin has no use_degrees: the request is refused, not fudged.
    with pytest.raises(ValueError, match="no use_degrees option"):
        BI_PIPER_FOLLOWER.with_use_degrees(True)
    # A Damiao follower only speaks degrees: asking for them changes nothing.
    assert BI_OPENARM_FOLLOWER.with_use_degrees(True) is BI_OPENARM_FOLLOWER
    with pytest.raises(ValueError, match="only records degrees"):
        BI_OPENARM_FOLLOWER.with_use_degrees(False)

    # A Feetech-style plugin flips arm joints only; the gripper keeps its mode.
    feetech_like = replace(BI_PIPER_FOLLOWER, degrees_option="optional")
    degrees = feetech_like.with_use_degrees(True)
    assert degrees.name == "lerobot_bi_piper_follower_degrees" and degrees.use_degrees
    assert all(enc.mode == DEGREES for enc in degrees.arm_encodings)
    assert degrees.gripper_encoding.mode == "range_0_100"
    assert degrees.with_use_degrees(False).name == feetech_like.name
    canonical = np.zeros((1, 14), dtype=np.float32)
    canonical[0, 0] = np.pi / 2
    external = from_canonical(canonical, layout=degrees, runtime=runtime)
    assert external[0, 0] == pytest.approx(-90.0, abs=1e-4)  # sign -1 kept
    assert all(enc.mode == RANGE_M100_100 for enc in feetech_like.arm_encodings)
    assert detect_external_layout(
        {"robot_type": "piper", "handumi": {"state_layout": "lerobot_bi_piper_follower"}}
    ) is BI_PIPER_FOLLOWER
