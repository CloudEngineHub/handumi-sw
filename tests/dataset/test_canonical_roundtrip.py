"""The canonical joint vector must survive a round trip back to URDF qpos.

``handumi replay-joints`` renders a converted dataset by inverting the
projection ``handumi convert`` applied, so anything the inverse loses shows up
as a robot that does not stand where the dataset says it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from handumi.dataset.canonical import (
    canonical_joint_layout,
    canonicalize_joint_trajectory,
    expand_canonical_trajectory,
)
from handumi.robots.registry import load_embodiment

EMBODIMENTS = ("piper", "openarmv1")


@pytest.fixture(scope="module", params=EMBODIMENTS)
def runtime(request):
    return load_embodiment(request.param)


def _arm_and_finger_indices(runtime) -> tuple[list[int], list[int]]:
    layout = canonical_joint_layout(runtime)
    arm = [
        index
        for index, side in zip(layout.indices, layout.gripper_sides, strict=True)
        if side is None and index is not None
    ]
    fingers = sorted(
        finger.index
        for side_fingers in (runtime.finger_joints or {}).values()
        for finger in side_fingers
    )
    return arm, fingers


def _sample(runtime, frames: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    nq = len(runtime.joint_names)
    qpos = np.tile(np.asarray(runtime.config.home_q, dtype=np.float32), (frames, 1))
    arm, _ = _arm_and_finger_indices(runtime)
    qpos[:, arm] += rng.uniform(-0.4, 0.4, size=(frames, len(arm))).astype(np.float32)
    assert qpos.shape[1] == nq
    normalized = rng.uniform(0.0, 1.0, size=(frames, 2)).astype(np.float32)
    # set_finger_positions is what replay writes, so start from its output.
    for frame in range(frames):
        runtime.set_finger_positions(
            qpos[frame], {"left": float(normalized[frame, 0]), "right": float(normalized[frame, 1])}
        )
    return qpos, normalized


def test_arm_joints_round_trip_exactly(runtime) -> None:
    qpos, normalized = _sample(runtime)
    canonical = canonicalize_joint_trajectory(
        qpos, runtime=runtime, gripper_normalized=normalized
    )
    restored, _ = expand_canonical_trajectory(canonical, runtime=runtime)
    arm, _ = _arm_and_finger_indices(runtime)
    # Arm columns are copied, never rescaled: any drift here is a layout bug.
    np.testing.assert_array_equal(restored[:, arm], qpos[:, arm])


def test_gripper_survives_the_width_round_trip(runtime) -> None:
    qpos, normalized = _sample(runtime)
    canonical = canonicalize_joint_trajectory(
        qpos, runtime=runtime, gripper_normalized=normalized
    )
    restored, restored_normalized = expand_canonical_trajectory(
        canonical, runtime=runtime
    )
    _, fingers = _arm_and_finger_indices(runtime)
    if not fingers:
        pytest.skip(f"{runtime.name} declares no gripper joints")
    # Conversion stores a physical width, so the opening passes through
    # normalized * max_width and back in float32 rather than being copied.
    np.testing.assert_allclose(restored_normalized, normalized, atol=1e-6)
    np.testing.assert_allclose(restored[:, fingers], qpos[:, fingers], atol=1e-7)


def test_expansion_rejects_a_foreign_layout(runtime) -> None:
    layout = canonical_joint_layout(runtime)
    wrong = np.zeros((4, layout.size + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="canonical columns"):
        expand_canonical_trajectory(wrong, runtime=runtime)
