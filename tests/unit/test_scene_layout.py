"""scene_layout skill: inventory report -> role-swap / rebased / grip-gap events and `scene.*` hints."""

import pytest

from contracts.real2sim import QaSummary
from contracts.sim_params import SceneParams, SimParams
from skills.attribution import build_hints
from skills.scene_layout import layout_events

CLIP_108 = {
    "source": "square table",
    "source_by_name": "round table",
    "source_from": "grip",
    "receptacle": "round table",
    "receptacle_rebased": True,
    "receptacle_shift_m": [-0.34, -0.02, 0.0],
    "grip_gap_m": 0.0,
    "held_match": None,
}


def _kinds(report):
    return [e.kind for e in layout_events(report)]


def test_clip_108_report_fires_swap_and_rebase_but_not_gap():
    assert _kinds(CLIP_108) == ["SCENE_ROLE_SWAP", "SCENE_RECEPTACLE_REBASED"]


def test_clean_or_missing_report_fires_nothing():
    assert layout_events(None) == []
    assert _kinds({"source": "box", "grip_gap_m": 0.02, "receptacle_rebased": False}) == []
    # a small rebase is the normal re-anchoring, not a layout defect
    assert _kinds({"receptacle_rebased": True, "receptacle_shift_m": [0.03, 0.0, 0.0]}) == []


def test_unregistered_or_large_gap_fires_grip_gap():
    assert _kinds({"unregistered": True, "grip_gap_m": 0.21}) == ["SCENE_GRIP_GAP"]
    assert _kinds({"grip_gap_m": 0.1, "source": "stand"}) == ["SCENE_GRIP_GAP"]


def test_hints_name_the_scene_knobs():
    events = layout_events(CLIP_108)
    hints = build_hints(SimParams(), events, QaSummary(success=True, failure_reasons=()), {}, {})
    by = {(h.pattern, h.param): h for h in hints}
    swap = by[("SCENE_ROLE_SWAP", "scene.source_from")]
    assert swap.current == "grip" and swap.direction == "set" and "plan" in swap.affects
    assert by[("SCENE_ROLE_SWAP", "scene.max_registration_m")].range == (0.02, 0.5)
    assert ("SCENE_RECEPTACLE_REBASED", "objects.receptacle.pos_offset_m") in by


def test_scene_params_validate():
    assert SceneParams().source_from == "grip" and SceneParams().max_registration_m == 0.15
    with pytest.raises(ValueError):
        SceneParams(source_from="geometry")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SceneParams(max_registration_m=0.8)
