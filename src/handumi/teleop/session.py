"""The shared per-frame teleoperation pipeline.

Simulation, physical teleoperation, and recording deliberately share this
module.  Mode-specific code decides *when* an arm may start and what to do
with the resulting command; this module owns the motion mapping itself:
tracking sample -> calibrated hand poses -> anchors -> IK -> smoothed command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from handumi.retargeting.handumi_to_robot import raw_state_pose7_pair
from handumi.teleop.common import (
    TeleopMotionSmoother,
    sample_state,
    tracking_sample_time_ns,
)
from handumi.teleop.core import TeleopController, TeleopStep


@dataclass(frozen=True)
class TeleopInputs:
    """Normalized live inputs, available before an operator-start decision."""

    raw_source_poses: dict[str, np.ndarray]
    side_tracked: dict[str, bool]
    openings: dict[str, float]
    sample_time_ns: int


@dataclass(frozen=True)
class TeleopFrame:
    """One mode-independent result of processing live teleoperation inputs."""

    inputs: TeleopInputs
    source_poses: dict[str, np.ndarray]
    anchored_sides: tuple[str, ...]
    step: TeleopStep
    q: np.ndarray


class TeleopSession:
    """Run the common live-teleoperation motion pipeline once per frame.

    Callers retain ownership of input policy (claps, auto-start, recovery) and
    output policy (rendering, hardware writes, or dataset storage).  Keeping
    those effects outside lets every mode share the safety-critical retargeting
    and smoothing path without making recording depend on a simulator.
    """

    def __init__(
        self,
        controller: TeleopController,
        motion_smoother: TeleopMotionSmoother,
    ) -> None:
        self.controller = controller
        self.motion_smoother = motion_smoother

    @staticmethod
    def inputs(sample: Any, widths: Any) -> TeleopInputs:
        """Normalize a sample once, including data needed by start policies."""
        side_tracked = {
            "left": bool(sample.left_tracked),
            "right": bool(sample.right_tracked),
        }
        state = sample_state(sample, widths)
        raw_source_poses: dict[str, np.ndarray] = dict(
            zip(("left", "right"), raw_state_pose7_pair(state), strict=True)
        )
        return TeleopInputs(
            raw_source_poses=raw_source_poses,
            side_tracked=side_tracked,
            openings={
                "left": float(widths.left_normalized),
                "right": float(widths.right_normalized),
            },
            sample_time_ns=tracking_sample_time_ns(sample),
        )

    def advance(
        self,
        inputs: TeleopInputs,
        *,
        now_s: float,
        start_sides: tuple[str, ...] = (),
    ) -> TeleopFrame:
        """Transform normalized inputs into the next robot command."""
        self.motion_smoother.anchor_sources(inputs.raw_source_poses, start_sides)
        source_poses = self.motion_smoother.smooth_source_poses(
            inputs.raw_source_poses,
            inputs.side_tracked,
            inputs.sample_time_ns,
        )
        anchored_sides = self.controller.anchor(
            source_poses, inputs.side_tracked, start_sides
        )
        step = self.controller.step(source_poses, inputs.side_tracked, inputs.openings)
        q = self.motion_smoother.smooth_joint_command(step.q, now_s)
        return TeleopFrame(
            inputs=inputs,
            source_poses=source_poses,
            anchored_sides=anchored_sides,
            step=step,
            q=q,
        )
