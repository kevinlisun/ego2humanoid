# ruff: noqa: E501
import numpy as np

from contracts.real2sim import FeedbackEvent, FeedbackHint, PhaseSpan, QaSummary, Real2SimFeedback
from contracts.sim_params import SimParams
from skills.feedback_md import render_feedback_md
from skills.real2sim_report import FeedbackInputs, build_feedback, write_feedback

GOLDEN = """\
# clip 0 · sharpa · iter 3 (parent 2) · rollout-only · params 4f2a…  → FAIL 1.50 (DROPPED_IN_CARRY)
Reference is monocular: 'matches the video' = matches video estimates (UNVERIFIED_MONOCULAR).
QA: lift 6.1 cm, carry 4.2 cm (need 7.5), place err —
Residuals (normalized): carry_path 3.8 cm (0.76) · lift_ratio — · grasp_dt 0.10 s (0.33) · release_dt — · place_err — · held_track — · tactile 1-IoU 0.41 (0.41) · final_tilt — · final_drift — · final_height — · place_buried_frac — · place_outside —
Timeline:
  1.62 s close   FIRST_TOUCH held
  2.05 s carry   SLIP held 7 mm along wrist -y in 0.2 s
  2.31 s carry   DROP held z 6 cm, gap 0.30 s
Hints (affects):
  objects.held.friction 0.6 → ↑ [0.05, 3] (sim+plan)
  contact.solref [0.01, 1.0] → ↑ [0.002, 0.1] (sim)
Changed vs parent: contact.noslip_iterations 5 → 10
"""


def _golden_feedback() -> Real2SimFeedback:
    return Real2SimFeedback(
        clip="0",
        hand="sharpa",
        iter=3,
        parent_iter=2,
        params_hash="4f2a9c0de1b25577",
        qa=QaSummary(
            success=False,
            failure_reasons=("DROPPED_IN_CARRY",),
            lift_height_m=0.061,
            translation_m=0.042,
            expected_translation_m=0.075,
        ),
        residuals={"carry_path_m": 0.038, "grasp_dt_s": 0.1, "tactile_iou": 0.41},
        match_score=1.5,
        phases=(PhaseSpan(name="close", t0_s=1.5, t1_s=1.8),),
        events=(
            FeedbackEvent(t_s=1.62, phase="close", kind="FIRST_TOUCH", object="held"),
            FeedbackEvent(
                t_s=2.05, phase="carry", kind="SLIP", object="held", detail="7 mm along wrist -y in 0.2 s"
            ),
            FeedbackEvent(t_s=2.31, phase="carry", kind="DROP", object="held", detail="z 6 cm, gap 0.30 s"),
        ),
        hints=(
            FeedbackHint(
                pattern="SLIP",
                param="objects.held.friction",
                current=0.6,
                range=(0.05, 3.0),
                direction="up",
                affects=("sim", "plan"),
            ),
            FeedbackHint(
                pattern="SIM_UNSTABLE",
                param="contact.solref",
                current=[0.01, 1.0],
                range=(0.002, 0.1),
                direction="up",
                affects=("sim",),
            ),
        ),
    )


def test_feedback_md_golden() -> None:
    md = render_feedback_md(_golden_feedback(), [("contact.noslip_iterations", 5, 10)])
    assert md == GOLDEN


def test_feedback_md_is_capped_at_60_lines() -> None:
    fb = _golden_feedback()
    many = tuple(FeedbackEvent(t_s=i * 0.1, phase="carry", kind="SLIP", object=f"o{i}") for i in range(40))
    fb = fb.model_copy(update={"events": many, "hints": fb.hints * 20})
    md = render_feedback_md(fb, [("a.b", 1, 2)] * 3)
    assert len(md.splitlines()) <= 60
    assert "more events in feedback.json" in md and "more hints" in md


N = 150
T_S = np.arange(N) * 0.02


def _inputs(drop: bool) -> FeedbackInputs:
    obj = np.tile(np.eye(4), (N, 1, 1, 1))
    z = np.where((T_S > 1.0) & (T_S < 2.0), 0.08, 0.0)
    obj[:, 0, 2, 3] = z
    obj[:, 0, 0, 3] = np.clip(T_S - 1.0, 0, 1) * 0.1
    contact = np.zeros((N, 1), bool)
    contact[(T_S >= 0.5) & (T_S < (1.5 if drop else 2.0)), 0] = True
    probe = {
        "pen_robot_table_m": np.zeros(N),
        "pen_robot_object_m": np.zeros((N, 1)),
        "pen_object_support_m": np.zeros((N, 1)),
        "force_robot_object_n": np.where(contact, 4.0, 0.0),
        "obj_speed_m_s": np.zeros((N, 1)),
        "obj_ang_speed_rad_s": np.zeros((N, 1)),
        "finite": np.ones(N, bool),
        "joint_group": np.array([0, 1]),
    }
    wrist = np.tile(np.eye(4), (N, 2, 1, 1))
    wrist[:, 1, :3, 3] = obj[:, 0, :3, 3]
    arrays = {
        "timestamps_s": T_S,
        "q_command_rad": np.zeros((N, 2)),
        "q_actual_rad": np.zeros((N, 2)),
        "wrist_actual_T_sim": wrist,
        "object_T_sim": obj,
        "object_robot_contact": contact,
    }
    pp = {
        "success": not drop,
        "failure_reasons": ("DROPPED_IN_CARRY",) if drop else (),
        "lift_height_m": 0.08,
        "translation_m": 0.1,
        "place_error_m": float("nan"),
        "carry_gap_s": 0.5,
    }
    ref_t = np.arange(0, 3, 0.1)
    refs = {
        "ref_t_s": ref_t,
        "ref_human_pinch_m": np.stack(
            [np.clip(ref_t - 1.0, 0, 1) * 0.1, 0 * ref_t, np.where((ref_t > 1.0) & (ref_t < 2.0), 0.08, 0.0)],
            1,
        ),
    }
    return FeedbackInputs(
        "c0", "right", 1, 0, SimParams(), "abcd1234abcd1234", "rollout", False, 1.5, arrays, probe, ["cup"], 0,
        [False], pp, {"close": (0.4, 0.8), "hold": (0.8, 1.0), "carry": (1.0, 2.0), "release": (2.0, 2.4)}, 2.0,
        grasp_onset_s=0.5, demo_carry_m=0.1, refs=refs, sized_grip_n=4.0,
    )  # fmt: skip


def test_build_feedback_success_has_low_score_and_no_hints() -> None:
    fb = build_feedback(_inputs(drop=False))
    assert fb.qa.success and fb.match_score < 0.4
    assert fb.residuals["carry_path_m"] is not None and fb.residuals["carry_path_m"] < 0.03
    assert fb.residuals["grasp_dt_s"] == 0.0
    assert [p.name for p in fb.phases] == ["close", "hold", "carry", "release"]
    assert not fb.hints


def test_build_feedback_drop_scores_failure_and_hints(tmp_path) -> None:
    fb = build_feedback(_inputs(drop=True))
    assert not fb.qa.success and fb.match_score == 1.5
    assert any(e.kind == "DROP" for e in fb.events)
    assert any(h.pattern == "SLIP" and h.param == "objects.held.friction" for h in fb.hints)
    assert not any(h.pattern == "GRIP_FORCE_LOW" for h in fb.hints)
    j, m = write_feedback(tmp_path / "it1", fb, [("contact.noslip_iterations", 5, 10)])
    assert Real2SimFeedback.model_validate_json(j.read_text()) == fb
    assert "DROP" in m.read_text() and "Changed vs parent" in m.read_text()


def test_scene_layout_report_adds_events_and_scene_hints() -> None:
    inp = _inputs(drop=False)
    inp.scene_layout = {"source": "a", "source_by_name": "b", "source_from": "grip"}
    fb = build_feedback(inp)
    assert any(e.kind == "SCENE_ROLE_SWAP" for e in fb.events)
    assert any(h.param == "scene.source_from" for h in fb.hints)


def test_never_touched_adds_no_contact_reason() -> None:
    inp = _inputs(drop=False)
    inp.arrays["object_robot_contact"][:] = False
    fb = build_feedback(inp)
    assert not fb.qa.success and "NO_CONTACT" in fb.qa.failure_reasons
    assert fb.match_score == 1.8
    assert any(h.pattern == "NO_CONTACT" for h in fb.hints)


def test_build_feedback_flags_box_fallback_with_knob_hint() -> None:
    from dataclasses import replace

    fb = build_feedback(replace(_inputs(drop=False), held_asset={"source": "box"}))
    assert any(e.kind == "HELD_MESH_FALLBACK_BOX" for e in fb.events)
    assert any(h.param == "trajectory.allow_held_box_fallback" for h in fb.hints)
