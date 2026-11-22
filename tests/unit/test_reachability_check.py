import numpy as np

from skills.reachability_check import reach_events, window_unreachable_fraction

T = np.linspace(0.0, 1.0, 11)


def test_window_fraction_counts_only_frames_inside() -> None:
    err = np.where(T >= 0.5, 0.05, 0.0)
    assert window_unreachable_fraction(err, T, (0.5, 1.0)) == 1.0
    assert window_unreachable_fraction(err, T, (0.0, 0.4)) == 0.0
    assert window_unreachable_fraction(err, T, (2.0, 3.0)) is None


def test_unreachable_place_after_a_grasp_only_stance_fires_both_events() -> None:
    reach = {"place_unreach_frac": 0.5, "grasp_unreach_frac": 0.0, "stance_scored_on_place": False}
    kinds = [e.kind for e in reach_events(reach, 2.0, "place", "perceived")]
    assert kinds == ["PLACE_UNREACHABLE", "STANCE_GRASP_ONLY"]


def test_place_scored_stance_only_reports_the_unreachable_place() -> None:
    reach = {"place_unreach_frac": 0.5, "grasp_unreach_frac": 0.0, "stance_scored_on_place": True}
    assert [e.kind for e in reach_events(reach, 2.0, "place", None)] == ["PLACE_UNREACHABLE"]


def test_reachable_or_unknown_place_is_silent() -> None:
    assert (
        reach_events({"place_unreach_frac": 0.05, "stance_scored_on_place": False}, 2.0, "place", None) == []
    )
    assert reach_events({"place_unreach_frac": None}, 2.0, "place", None) == []
    assert reach_events(None, 2.0, "place", None) == []
