"""Hand-pose / tactile overlay drawn on the left panel of the review video."""

import numpy as np

from stages.qa.hand_overlay import (
    ACTIVE_COLOR,
    LOW_SCORE_COLOR,
    PASSIVE_COLOR,
    draw_frame_overlay,
    force_color,
    landmark_region_ids,
    region_force_series,
)
from stages.reconstruction.hands.tactile import N_REGIONS, REGION_NAMES

FRAME_HW = (600, 800)  # large enough that the legend never overlaps the drawn hands


def _hand(x: float = 200.0) -> np.ndarray:
    """A flat synthetic hand well below the legend."""
    lm = np.zeros((21, 2))
    for finger in range(5):
        for k in range(4):
            lm[1 + 4 * finger + k] = (x + 30 * k, 300.0 + 40 * finger)
    lm[0] = (x - 60, 380.0)
    return lm


def _at(img: np.ndarray, p: np.ndarray) -> tuple[int, ...]:
    return tuple(int(c) for c in img[int(round(p[1])), int(round(p[0]))])


def test_landmark_regions_cover_tips_pads_and_palm():
    ids = landmark_region_ids()
    assert ids.shape == (21,)
    assert ids[0] == REGION_NAMES.index("palm")
    for finger, tip in enumerate((4, 8, 12, 16, 20)):
        assert REGION_NAMES[ids[tip]].endswith("_tip")
        assert ids[tip] == finger
        assert REGION_NAMES[ids[tip - 1]].endswith("_pad")
    assert ids.max() < N_REGIONS


def test_force_color_runs_from_cold_to_hot():
    cold, hot = force_color(0.0), force_color(1.0)
    assert cold[0] > cold[2]  # BGR: blue end
    assert hot[2] > hot[0]  # red end
    assert force_color(5.0) == hot  # clipped, not extrapolated


def test_force_color_matches_the_opencv_jet_map():
    import cv2

    for v in np.linspace(0.0, 1.0, 1001):
        ref = cv2.applyColorMap(np.uint8([[v * 255]]), cv2.COLORMAP_JET)[0, 0]
        assert force_color(v) == tuple(int(c) for c in ref)


def test_overlay_draws_both_hands_and_marks_low_score():
    frame = np.zeros((*FRAME_HW, 3), np.uint8)
    left_lm, right_lm = _hand(100.0), _hand(450.0)
    j2d = np.stack([left_lm, right_lm])
    good = np.array([0.9, 0.9])

    both = draw_frame_overlay(frame, j2d, np.ones(2, bool), good, "left")
    assert frame.max() == 0  # input frame is not modified
    assert _at(both, left_lm[6]) == ACTIVE_COLOR  # the retargeted hand
    assert _at(both, right_lm[6]) == PASSIVE_COLOR  # the idle hand

    # A hand below the score gate is still drawn, but greyed out rather than in its side colour.
    dropped = draw_frame_overlay(frame, j2d, np.ones(2, bool), np.array([0.9, 0.2]), "left")
    assert _at(dropped, left_lm[6]) == ACTIVE_COLOR
    assert _at(dropped, right_lm[6]) == LOW_SCORE_COLOR

    # An invalid hand is not drawn at all.
    one = draw_frame_overlay(frame, j2d, np.array([True, False]), good, "left")
    assert one[250:, 400:].max() == 0


def test_overlay_paints_tactile_force_on_the_landmarks():
    frame = np.zeros((*FRAME_HW, 3), np.uint8)
    left_lm = _hand(100.0)
    j2d = np.stack([left_lm, _hand(450.0)])
    force = np.zeros((2, N_REGIONS))
    force[0, REGION_NAMES.index("index_tip")] = 2.0

    out = draw_frame_overlay(frame, j2d, np.ones(2, bool), np.array([0.9, 0.9]), "left", region_force=force)
    assert _at(out, left_lm[8]) == force_color(1.0)  # the loaded region tops the colour scale
    assert _at(out, left_lm[12]) == force_color(0.0)  # an unloaded region stays cold


def test_region_force_series_resamples_to_the_frame_count(tmp_path):
    assert region_force_series(tmp_path / "missing.npz", 5) is None
    f = np.arange(3 * 2 * N_REGIONS, dtype=float).reshape(3, 2, N_REGIONS)
    np.savez(tmp_path / "t.npz", force_region=f)
    out = region_force_series(tmp_path / "t.npz", 7)
    assert out.shape == (7, 2, N_REGIONS)
    assert np.allclose(out[0], f[0]) and np.allclose(out[-1], f[-1])


def test_overlay_scales_each_hand_independently():
    """A hand pressing hard (passive, on the table) must not wash out the other hand's grasp forces."""
    frame = np.zeros((*FRAME_HW, 3), np.uint8)
    left_lm, right_lm = _hand(100.0), _hand(450.0)
    j2d = np.stack([left_lm, right_lm])
    force = np.zeros((2, N_REGIONS))
    force[0, REGION_NAMES.index("index_tip")] = 2.0
    force[1, REGION_NAMES.index("index_tip")] = 40.0

    out = draw_frame_overlay(frame, j2d, np.ones(2, bool), np.array([0.9, 0.9]), "left", region_force=force)
    assert _at(out, left_lm[8]) == force_color(1.0)  # per-hand maximum, not the global one
    assert _at(out, right_lm[8]) == force_color(1.0)

    half = draw_frame_overlay(
        frame,
        j2d,
        np.ones(2, bool),
        np.array([0.9, 0.9]),
        "left",
        region_force=force,
        force_scale_n=np.array([4.0, 40.0]),
    )
    assert _at(half, left_lm[8]) == force_color(0.5)


def test_region_force_series_reads_the_simulated_bundle(tmp_path):
    f = np.arange(4 * 2 * N_REGIONS, dtype=float).reshape(4, 2, N_REGIONS)
    np.savez(tmp_path / "sim_tactile.npz", normal_force_n=f, contact=f > 0)
    out = region_force_series(tmp_path / "sim_tactile.npz", 4)
    assert np.allclose(out, f)
