from __future__ import annotations

import numpy as np

from handumi.inpainting import MaskConfig, composite, edit_mask, operator_footprint


def _clip(frames: int = 6, height: int = 64, width: int = 96) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 60, size=(frames, height, width, 3), dtype=np.uint8) + 90


def test_untouched_pixels_survive_compositing():
    """Outside the edit, the composite must be the recorded frames."""
    source = _clip()
    generated = source.copy()
    generated[:, 20:40, 30:60] = 255  # the model changed one patch

    masks = edit_mask(source, generated)
    result = composite(source, generated, masks)

    far = (slice(None), slice(0, 8), slice(0, 8))
    assert np.array_equal(result[far], source[far])


def test_edited_region_takes_generated_pixels():
    source = _clip()
    generated = source.copy()
    generated[:, 20:40, 30:60] = 255

    result = composite(source, generated, edit_mask(source, generated))
    centre = result[:, 28:32, 42:48].astype(int)
    assert centre.mean() > source[:, 28:32, 42:48].astype(int).mean() + 50


def test_identical_clips_produce_an_empty_mask():
    source = _clip()
    masks = edit_mask(source, source.copy())
    assert masks.max() == 0
    assert np.array_equal(composite(source, source.copy(), masks), source)


def test_temporal_radius_widens_a_single_frame_edit():
    source = _clip(frames=7)
    generated = source.copy()
    generated[3, 20:40, 30:60] = 255

    narrow = edit_mask(source, generated, MaskConfig(temporal_radius=0))
    wide = edit_mask(source, generated, MaskConfig(temporal_radius=2))

    assert narrow[1].max() == 0
    assert wide[1].max() > 0, "neighbouring frames should share the mask so the seam holds still"


def _static_clip(frames: int = 8, height: int = 128, width: int = 192) -> np.ndarray:
    """A fixed camera on a still scene: only what is injected below moves."""
    return np.full((frames, height, width, 3), 120, dtype=np.uint8)


def test_anchoring_drops_drift_away_from_the_operator():
    """The model re-renders every frame; only the operator's region may change."""
    source = _static_clip()
    source[:, 20:60, 20:60] = 30          # the operator, moving
    source[4:, 20:60, 60:100] = 30

    generated = source.copy()
    generated[:, 20:60, 20:100] = 200     # the intended edit, where the operator was
    generated[:, 100:120, 150:185] = 200  # drift in a corner the operator never reached

    loose = edit_mask(source, generated, anchored=False)
    tight = edit_mask(source, generated, anchored=True)

    assert tight.mean() < loose.mean(), "anchoring must shrink the mask, not grow it"
    assert loose[:, 110, 170].max() > 0, "the loose mask keeps the corner drift"
    assert tight[:, 110, 170].max() == 0, "the anchored mask drops it"


def test_operator_footprint_covers_what_moves():
    source = _static_clip()
    source[3:, 20:60, 30:70] = 250        # something appears and stays
    footprint = operator_footprint(source)
    assert footprint[4, 40, 50], "a moving region belongs to the operator footprint"
    assert not footprint[0, 120, 180], "a corner that never moves does not"
