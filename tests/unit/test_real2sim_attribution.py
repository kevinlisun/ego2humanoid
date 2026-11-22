import numpy as np
import pytest

from contracts.real2sim import FeedbackEvent, QaSummary
from contracts.sim_params import ObjectPhysics, SimParams
from skills import residuals as res
from skills.attribution import (
    RULES,
    build_hints,
    fired_patterns,
    get_param,
    param_affects,
    param_exists,
    param_range,
)

PHASES = {"close": (1.0, 1.5), "carry": (2.0, 3.0)}


def _ev(kind: str, t: float = 2.2, obj: str | None = "cup", detail: str = "") -> FeedbackEvent:
    return FeedbackEvent(t_s=t, phase="carry", kind=kind, object=obj, detail=detail)


def _qa(*reasons: str, success: bool = False) -> QaSummary:
    return QaSummary(success=success, failure_reasons=reasons)


def test_every_rule_param_exists_in_sim_params() -> None:
    for pattern, rules in RULES.items():
        for rule in rules:
            path = rule.param.format(name="obj")
            assert param_exists(path), f"{pattern}: {path}"
            get_param(SimParams(), path)


def test_unknown_paths_do_not_exist() -> None:
    assert not param_exists("trajectory.nope")
    assert not param_exists("objects.held")
    assert not param_exists("objects.held.nope")


def test_get_param_object_selector_falls_back_to_default() -> None:
    p = SimParams(objects={"held": ObjectPhysics(friction=0.6)})
    assert get_param(p, "objects.held.friction") == 0.6
    assert get_param(p, "objects.held.mass_kg") is None
    assert get_param(p, "objects.receptacle.pos_offset_m") == (0.0, 0.0, 0.0)
    assert get_param(p, "pd.groups.hand.kp") == 8.0


def test_affects_and_range() -> None:
    assert param_affects("objects.held.friction") == ("sim", "plan")
    assert param_affects("contact.noslip_iterations") == ("sim",)
    assert param_range("objects.held.friction") == (0.05, 3.0)
    assert param_range("pd.groups.hand.kp") == (1, 50)
    assert param_range("trajectory.squeeze_m") == (0.0, 0.03)


@pytest.mark.parametrize(
    ("events", "qa", "pattern"),
    [
        ([_ev("SIM_UNSTABLE", obj=None)], _qa(), "SIM_UNSTABLE"),
        ([], _qa("NO_CONTACT"), "NO_CONTACT"),
        ([_ev("EJECT", t=1.3)], _qa("EJECTED"), "EJECT_AT_CLOSE"),
        ([_ev("SLIP")], _qa(), "SLIP"),
        ([], _qa("DROPPED_IN_CARRY"), "SLIP"),
        ([_ev("ROTATE_IN_GRASP")], _qa(), "ROTATE_IN_GRASP"),
        ([], _qa("STUCK_AFTER_RELEASE"), "STUCK_AFTER_RELEASE"),
        ([], _qa("AIRBORNE_RELEASE"), "AIRBORNE_RELEASE"),
        ([_ev("DISTURBED", obj="bowl")], _qa(), "DISTURBED"),
        ([_ev("PENETRATION", obj=None, detail="robot-table peak 8 mm")], _qa(), "PENETRATION_ROBOT_TABLE"),
        ([_ev("TRACKING_ERROR", obj=None, detail="hand joints peak 0.2 rad")], _qa(), "TRACKING_ERROR_HAND"),
        ([_ev("GRASP_PENETRATION", t=1.0)], _qa(), "GRASP_PENETRATION"),
        ([_ev("START_EMBEDDED", t=0.0, obj="scene_bowl")], _qa(), "START_EMBEDDED"),
        ([_ev("PLACE_BURIED", obj="perceived")], _qa(), "PLACE_BURIED"),
        ([_ev("HELD_MESH_FALLBACK_BOX", t=0.0, obj="perceived")], _qa(), "HELD_MESH_FALLBACK_BOX"),
        ([_ev("PLACE_UNREACHABLE", obj="perceived")], _qa(), "PLACE_UNREACHABLE"),
        ([_ev("STANCE_GRASP_ONLY", obj="perceived")], _qa(), "STANCE_GRASP_ONLY"),
    ],
)
def test_pattern_fires_and_hints_use_real_params(events, qa, pattern) -> None:
    fired = [p for p, _ in fired_patterns(events, qa, {}, PHASES)]
    assert pattern in fired
    hints = build_hints(SimParams(), events, qa, {}, PHASES)
    mine = [h for h in hints if h.pattern == pattern]
    assert mine and all(param_exists(h.param) for h in mine)


def test_context_dependent_patterns() -> None:
    kw = {"has_receptacle": True}
    assert ("NOT_PLACED", None) in fired_patterns([], _qa("NOT_PLACED"), {}, PHASES, **kw)
    assert not fired_patterns([], _qa("NOT_PLACED"), {}, PHASES)
    assert not fired_patterns([_ev("EJECT", t=2.5)], _qa(), {}, PHASES)  # not near the close
    assert ("GRIP_FORCE_LOW", None) in fired_patterns([], _qa(), {}, PHASES, grip_ratio=0.3)
    assert not fired_patterns([], _qa(), {}, PHASES, grip_ratio=0.9)
    late = fired_patterns([], _qa(), {"grasp_dt_s": 0.4}, PHASES, grasp_sign_s=0.4)
    early = fired_patterns([], _qa(), {"grasp_dt_s": 0.4}, PHASES, grasp_sign_s=-0.4)
    assert ("GRASP_LATE", None) in late and ("GRASP_EARLY", None) in early
    assert ("LIFT_RATIO", None) in fired_patterns([], _qa(success=True), {"lift_ratio": 0.4}, PHASES)
    assert not fired_patterns([], _qa(), {"lift_ratio": 0.4}, PHASES)
    assert not fired_patterns([_ev("LIFT_OFF"), _ev("FIRST_TOUCH")], _qa(success=True), {}, PHASES)


def test_disturbed_hint_targets_the_named_object() -> None:
    hints = build_hints(SimParams(), [_ev("DISTURBED", obj="bowl")], _qa(), {}, PHASES)
    assert [h.param for h in hints] == ["objects.bowl.static", "objects.bowl.mass_kg"]
    assert hints[0].direction == "set" and hints[0].current is None


def test_hint_carries_current_value() -> None:
    p = SimParams(objects={"held": ObjectPhysics(friction=0.6)})
    hint = next(
        h for h in build_hints(p, [_ev("SLIP")], _qa(), {}, PHASES) if h.param == "objects.held.friction"
    )
    assert hint.current == 0.6 and hint.range == (0.05, 3.0) and hint.direction == "up"


def test_carry_path_zero_when_sim_follows_human_displacement() -> None:
    t = np.arange(0, 3, 0.02)
    ref_t = np.arange(0, 3, 0.1)
    human = np.stack([ref_t * 0.1, 0 * ref_t, 0 * ref_t], 1)
    sim = np.stack(
        [0.7 + 0.1 * (t - 0.4), 0.2 + 0 * t, 0 * t], 1
    )  # offset start, grasp at 0.4 s, human at 1.0 s
    r = res.carry_path_m(t, sim, (1.0, 2.0), 0.4, 1.0, ref_t, human)
    assert r == pytest.approx(0.0, abs=1e-9)
    sim[:, 1] += np.where(t > 1.0, 0.05, 0.0)
    assert res.carry_path_m(t, sim, (1.0, 2.0), 0.4, 1.0, ref_t, human) == pytest.approx(0.05, abs=1e-3)


def test_place_err_variants() -> None:
    pos = np.array([[0, 0, 0.0], [0.3, 0.4, 0.1]])
    assert res.place_err_m(pos, np.array([0.3, 0.3, 0.0]), None) == pytest.approx(0.1)
    off = (np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.4, 0.0]))
    assert res.place_err_m(pos, None, off) == pytest.approx(0.0)
    assert res.place_err_m(pos, None, None) is None


def test_held_track_needs_five_samples_and_aligns_first() -> None:
    t = np.linspace(0, 1, 51)
    sim = np.stack([t, 0 * t, 0 * t], 1) + 5.0
    ref_t = np.linspace(0, 1, 6)
    ref = np.stack([ref_t, 0 * ref_t, 0 * ref_t], 1)
    assert res.held_track_m(t, sim, ref_t, ref) == pytest.approx(0.0, abs=1e-9)
    assert res.held_track_m(t, sim, ref_t[:4], ref[:4]) is None


def test_tactile_and_ratios() -> None:
    t = np.arange(10) * 0.1
    sim = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], bool)
    ref = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], bool)
    assert res.tactile_residual(t, sim, t, ref) == pytest.approx(1 - 3 / 5)
    assert res.lift_ratio(0.06, 0.08) == pytest.approx(0.25)
    assert res.lift_ratio(0.06, None) is None
    assert res.timing_residual(1.0, 1.25) == pytest.approx(0.25)


def test_match_score_failure_and_success() -> None:
    assert res.match_score({}, False, ("DROPPED_IN_CARRY", "STUCK_AFTER_RELEASE")) == pytest.approx(1.5)
    assert res.match_score({}, False, ("NO_CONTACT", "NO_LIFT")) == pytest.approx(1.8)
    r = {"carry_path_m": 0.025, "place_err_m": 0.5, "grasp_dt_s": None}  # clipped to 1
    assert res.match_score(r, True, ()) == pytest.approx((3 * 0.5 + 3 * 1.0) / 6)
    assert res.match_score({}, True, ()) == 0.0
