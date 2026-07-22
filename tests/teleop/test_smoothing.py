import numpy as np
import pytest

from handumi.teleop.smoothing import JointCommandSmoother, JointSmoothingConfig


def test_smoother_initializes_at_first_target_without_jump():
    smoother = JointCommandSmoother()
    q = np.array([1.0, -2.0], dtype=np.float32)

    np.testing.assert_array_equal(smoother.smooth(q, dt=1.0 / 30.0), q)


def test_smoother_limits_per_joint_velocity():
    smoother = JointCommandSmoother(
        JointSmoothingConfig(cutoff_hz=0.0, max_velocity_rad_s=0.6)
    )
    smoother.reset(np.array([0.0, 0.0], dtype=np.float32))

    q = smoother.smooth(np.array([10.0, -10.0], dtype=np.float32), dt=0.1)

    np.testing.assert_allclose(q, np.array([0.06, -0.06], dtype=np.float32))


def test_smoother_low_passes_jitter_when_velocity_limit_is_disabled():
    smoother = JointCommandSmoother(
        JointSmoothingConfig(cutoff_hz=1.0, max_velocity_rad_s=0.0)
    )
    smoother.reset(np.array([0.0], dtype=np.float32))

    q = smoother.smooth(np.array([1.0], dtype=np.float32), dt=0.1)

    assert 0.0 < q.item() < 1.0


def test_smoother_rejects_negative_tuning():
    with pytest.raises(ValueError):
        JointCommandSmoother(JointSmoothingConfig(cutoff_hz=-1.0))
    with pytest.raises(ValueError):
        JointCommandSmoother(JointSmoothingConfig(max_velocity_rad_s=-1.0))
