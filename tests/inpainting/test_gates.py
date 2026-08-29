from __future__ import annotations

import numpy as np

from handumi.inpainting import ClipSpec, GateThresholds, evaluate


def _clip(frames: int = 8, height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(90, 150, size=(frames, height, width, 3), dtype=np.uint8)


def test_an_unedited_clip_scores_a_perfect_pass():
    """A suite that cannot recognise an identical video is not a suite."""
    source = _clip()
    report = evaluate(source, source.copy(), source.copy(), episode=0, frames_expected=len(source))

    assert report.frames_pass
    assert report.model_changed_pct == 0.0
    assert report.diff_outside_edit_after == 0.0
    assert report.skin_reduction_pct == 0.0
    assert report.scene_cuts == 0


def test_dropped_frames_fail_the_frame_gate():
    source = _clip()
    short = source[:-2]
    report = evaluate(source, short, short, episode=0, frames_expected=len(source))
    assert not report.frames_pass
    assert report.to_dict()["G1_frames_verdict"] == "FAIL"


def test_edited_area_is_reported():
    source = _clip()
    generated = source.copy()
    generated[:, 10:30, 10:40] = 255
    report = evaluate(source, generated, generated, episode=0, frames_expected=len(source))
    assert report.model_changed_pct > 15


def test_a_clip_over_the_upload_limit_is_compressed_not_refused():
    """The cap is on duration, so a long episode is sped up for the call."""
    spec = ClipSpec(episode=0, first_frame=0, frames=320, fps=30, width=672, height=376)
    assert spec.seconds > 10
    spec.validate_uploadable()
    assert spec.time_scale > 1.0


def test_clip_name_and_timing():
    spec = ClipSpec(episode=0, first_frame=0, frames=300, fps=30, width=672, height=376)
    assert spec.name == "ep000_f000-299"
    assert spec.seconds == 10.0
    spec.validate_uploadable()


def test_gate_report_creates_its_directory(tmp_path):
    """Artifacts must not fail because a run directory does not exist yet."""
    source = _clip()
    report = evaluate(source, source.copy(), source.copy(), episode=0, frames_expected=len(source))
    written = report.write(tmp_path / "metrics" / "ep000.json")
    assert written.exists()


def test_a_clip_shorter_than_its_episode_is_flagged():
    """Frames with an action row and no picture would mislabel the dataset."""
    spec = ClipSpec(episode=0, first_frame=0, frames=300, fps=30,
                    width=672, height=376, episode_frames=467)
    assert not spec.covers_episode
    assert spec.missing_frames == 167


def test_a_clip_that_spans_its_episode_passes():
    spec = ClipSpec(episode=5, first_frame=0, frames=146, fps=30,
                    width=672, height=376, episode_frames=146)
    assert spec.covers_episode
    assert spec.missing_frames == 0


def test_coverage_accounts_for_the_starting_offset():
    spec = ClipSpec(episode=0, first_frame=100, frames=200, fps=30,
                    width=672, height=376, episode_frames=300)
    assert spec.covers_episode


def _still_clip(frames: int = 8, height: int = 48, width: int = 64) -> np.ndarray:
    """A fixed camera on a still, skinless scene: nothing here should be flagged."""
    return np.full((frames, height, width, 3), 120, dtype=np.uint8)


def test_a_clean_episode_is_accepted_with_no_findings():
    source = _still_clip()
    report = evaluate(source, source.copy(), source.copy(), episode=0, frames_expected=len(source))
    assert report.accepted
    assert report.findings == []
    assert report.to_dict()["status"] == "accepted"


def test_a_frame_mismatch_rejects_the_episode():
    """Mechanical: pictures and labels disagree, and no judgement can fix that."""
    source = _still_clip()
    short = source[:-2]
    report = evaluate(source, short, short, episode=0, frames_expected=len(source))
    assert not report.accepted
    assert [f.code for f in report.findings] == ["frame_count_mismatch"]
    assert report.findings[0].severity == "reject"


def test_an_operator_left_in_frame_warns_rather_than_rejects():
    """Context decides, so a leftover operator is a warning, not a rejection."""
    source = _still_clip()
    source[:, 10:30, 10:40] = (120, 150, 200)   # skin tones the edit should remove
    unchanged = source.copy()                    # ... but did not

    report = evaluate(source, unchanged, unchanged, episode=0, frames_expected=len(source))

    assert report.skin_reduction_pct == 0.0
    assert report.accepted, "a warning must not reject the episode"
    assert [f.code for f in report.findings] == ["operator_still_visible"]
    assert report.findings[0].severity == "warning"


def test_thresholds_are_reported_as_findings_not_hidden():
    source = _still_clip()
    report = evaluate(source, source.copy(), source.copy(), episode=0,
                      frames_expected=len(source),
                      thresholds=GateThresholds(max_model_changed_pct=-1.0))
    codes = {f.code for f in report.findings}
    assert "over_painting" in codes
    assert all(f.message for f in report.findings), "a finding must say what it saw"


def test_no_metric_is_nan_when_nothing_was_edited():
    """A NaN would pass or fail a threshold unpredictably."""
    source = _still_clip()
    report = evaluate(source, source.copy(), source.copy(), episode=0, frames_expected=len(source))
    for name, value in report.to_dict().items():
        if isinstance(value, float):
            assert value == value, f"{name} is NaN"
