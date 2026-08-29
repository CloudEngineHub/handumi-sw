"""Verification gates for one generated clip.

Gates decide correctness; a human still decides whether it looks like a robot.
They are cheap and local, so they run after every generation and after every
free local fix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

Severity = Literal["reject", "warning"]


@dataclass(frozen=True)
class GateThresholds:
    """Where a measurement stops being variation and becomes a finding.

    Calibrated from the runs on `tblock` at 360p, whose spread was: 12.9-22.7%
    of pixels changed, 2.5-5.5 difference outside the edit, 67.7-85.7% of skin
    removed and 0.029-0.088 flicker. The warning limits sit outside that envelope
    so they flag an outlier rather than normal variation; with a sample this
    small a tighter limit would only cry wolf.
    """

    min_skin_reduction_pct: float = 60.0
    max_diff_outside_edit: float = 8.0
    max_static_diff: float = 0.15
    max_model_changed_pct: float = 35.0


@dataclass(frozen=True)
class GateFinding:
    """One reason a generated episode needs a decision."""

    code: str
    severity: Severity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Standard YCrCb skin gate. It also fires on warm, cream-coloured cloth, so read
# the reduction between input and output rather than the absolute fraction.
SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return np.array(frames)


def skin_fraction(frames: np.ndarray) -> np.ndarray:
    return np.array([
        float(cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb), SKIN_LOWER, SKIN_UPPER).mean()) / 255.0
        for f in frames
    ])


@dataclass
class GateReport:
    episode: int
    frames_expected: int
    frames_got: int
    model_changed_pct: float
    diff_outside_edit_before: float
    diff_outside_edit_after: float
    skin_fraction_input: float
    skin_fraction_output: float
    skin_reduction_pct: float
    scene_cuts: int
    static_diff_input: float
    static_diff_output: float
    findings: list[GateFinding] = field(default_factory=list)

    @property
    def frames_pass(self) -> bool:
        return self.frames_got == self.frames_expected

    @property
    def accepted(self) -> bool:
        """Rejections are mechanical; warnings are for a person to weigh."""
        return not any(finding.severity == "reject" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["findings"] = [f.to_dict() for f in self.findings]
        data["G1_frames_verdict"] = "PASS" if self.frames_pass else "FAIL"
        data["status"] = "accepted" if self.accepted else "rejected"
        return data

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def evaluate(
    source: np.ndarray,
    generated: np.ndarray,
    composited: np.ndarray,
    *,
    episode: int,
    frames_expected: int,
    difference_threshold: int = 28,
    thresholds: GateThresholds | None = None,
) -> GateReport:
    count = min(len(source), len(generated), len(composited))
    source, generated, composited = source[:count], generated[:count], composited[:count]

    model_diff = np.abs(source.astype(np.int16) - generated.astype(np.int16)).mean(axis=3)
    edited = model_diff > difference_threshold
    composited_diff = np.abs(source.astype(np.int16) - composited.astype(np.int16)).mean(axis=3)

    skin_in, skin_out = skin_fraction(source), skin_fraction(composited)
    consecutive = np.abs(composited[1:].astype(np.int16) - composited[:-1].astype(np.int16)).mean(axis=(1, 2, 3))

    # "Static" means static in the recording, so drift there is the model's.
    motion = np.abs(source[1:].astype(np.int16) - source[:-1].astype(np.int16)).mean(axis=(0, 3))
    static = motion < np.percentile(motion, 40)

    def _mean(values: np.ndarray) -> float:
        """Zero, not NaN, when a selection is empty.

        A NaN here would pass or fail a threshold unpredictably, which is worse
        than a number that plainly says "nothing to measure".
        """
        return float(values.mean()) if values.size else 0.0

    def temporal(frames: np.ndarray) -> float:
        deltas = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).mean(axis=3)
        return _mean(deltas[:, static])

    thresholds = thresholds or GateThresholds()
    report = GateReport(
        episode=episode,
        frames_expected=frames_expected,
        frames_got=len(composited),
        model_changed_pct=round(100 * float(edited.mean()), 1),
        diff_outside_edit_before=round(_mean(model_diff[~edited]), 2),
        diff_outside_edit_after=round(_mean(composited_diff[~edited]), 2),
        skin_fraction_input=round(_mean(skin_in), 5),
        skin_fraction_output=round(_mean(skin_out), 5),
        skin_reduction_pct=round(100 * (1 - skin_out.mean() / max(skin_in.mean(), 1e-9)), 1),
        scene_cuts=int((consecutive > _mean(consecutive) + 6 * consecutive.std()).sum())
        if consecutive.size
        else 0,
        static_diff_input=round(temporal(source), 3),
        static_diff_output=round(temporal(composited), 3),
    )
    report.findings.extend(_findings(report, thresholds))
    return report


def _findings(report: GateReport, limits: GateThresholds) -> list[GateFinding]:
    """Grade one report: mechanical failures reject, judgement calls warn."""
    findings: list[GateFinding] = []
    if not report.frames_pass:
        findings.append(GateFinding(
            "frame_count_mismatch", "reject",
            f"{report.frames_got} frames for {report.frames_expected} action rows.",
        ))
    if report.scene_cuts:
        findings.append(GateFinding(
            "scene_cut", "reject",
            f"{report.scene_cuts} scene cut(s): the model restarted the shot mid-episode.",
        ))
    if report.skin_reduction_pct < limits.min_skin_reduction_pct:
        findings.append(GateFinding(
            "operator_still_visible", "warning",
            f"Skin fell only {report.skin_reduction_pct:.1f}% "
            f"(limit {limits.min_skin_reduction_pct:.0f}%); the operator may still be in frame.",
        ))
    if report.diff_outside_edit_after > limits.max_diff_outside_edit:
        findings.append(GateFinding(
            "scene_drift", "warning",
            f"Difference outside the edit is {report.diff_outside_edit_after:.2f} "
            f"(limit {limits.max_diff_outside_edit:.1f}); the untouched scene moved.",
        ))
    if report.static_diff_output > limits.max_static_diff:
        findings.append(GateFinding(
            "flicker", "warning",
            f"Static regions vary by {report.static_diff_output:.3f} "
            f"(limit {limits.max_static_diff:.2f}).",
        ))
    if report.model_changed_pct > limits.max_model_changed_pct:
        findings.append(GateFinding(
            "over_painting", "warning",
            f"The model changed {report.model_changed_pct:.1f}% of pixels "
            f"(limit {limits.max_model_changed_pct:.0f}%).",
        ))
    return findings
