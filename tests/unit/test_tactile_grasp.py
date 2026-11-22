import numpy as np

from stages.retargeting.tactile_grasp import (
    PAD_VERTICES,
    contact_span,
    contact_spans,
    grasp_onset,
    pick_span,
    pinch_contact,
    tactile_release_index,
)


def test_pinch_contact_needs_thumb_and_a_finger():
    p = np.zeros((30, 2, 778))
    p[5:20, 1, list(PAD_VERTICES["thumb"])] = 0.9
    p[10:20, 1, list(PAD_VERTICES["index"])] = 0.9
    on = pinch_contact(p, 1)
    assert not on[:9].any() and on[11:19].all() and not on[21:].any()
    assert not pinch_contact(p, 0).any()


def test_contact_span_picks_longest_run_and_drops_outside_samples():
    fi = np.arange(-5, 25)
    on = np.zeros(30, bool)
    on[0:8] = True  # mostly before the clip
    on[12:20] = True  # frames 7..14
    on[22:30] = True  # frames 17..24, only 17..19 inside a 20-frame clip
    assert contact_span(fi, on, 20) == (7, 14)
    assert contact_span(fi, np.zeros(30, bool), 20) is None


def test_grasp_onset_finds_start_of_main_rise_after_a_slide():
    # The pinch slides down onto the table, rests, then rises 10 cm (the pick).
    z = np.concatenate([np.linspace(0.95, 0.85, 20), np.full(10, 0.85), np.linspace(0.85, 0.95, 10)])
    onset, rise = grasp_onset(z, 0, len(z) - 1)
    assert 19 <= onset <= 30 and abs(rise - 0.10) < 1e-6
    onset, _ = grasp_onset(z, 25, len(z) - 1)
    assert 25 <= onset <= 30


def test_pick_span_prefers_the_contact_with_the_lift_over_the_longest_one():
    # Short contact while the pinch rises 9 cm (the pick), then a long contact holding it still (place).
    rise, fall = np.linspace(0.88, 0.97, 8), np.linspace(0.97, 0.87, 12)
    z = np.concatenate([np.full(10, 0.88), rise, fall, np.full(30, 0.87)])
    on = np.zeros(len(z), bool)
    on[8:18] = True
    on[28:58] = True
    spans = contact_spans(np.arange(len(z)), on, len(z))
    assert spans == [(8, 17), (28, 57)] and contact_span(np.arange(len(z)), on, len(z)) == (28, 57)
    (a, b), onset, rise = pick_span(spans, z, 0, len(z) - 1)
    assert (a, b) == (8, 17) and 8 <= onset <= 10 and rise > 0.08
    assert pick_span(spans, z, 20, 25) is None


def test_tactile_release_index_only_moves_for_a_clearly_early_end():
    t = np.arange(300) / 30.0
    hold = np.arange(30, 250)
    assert tactile_release_index((30, 100), hold, t) == 100  # tactile ends 5 s early
    assert tactile_release_index((30, 240), hold, t) == 249  # 0.3 s early: keep the geometry
    assert tactile_release_index(None, hold, t) == 249
    assert tactile_release_index((10, 20), hold, t) == 249  # span before the hold: ignore
