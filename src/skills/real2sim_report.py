"""Top-level builder of the real2sim feedback (`feedback.json` + `feedback.md`), spec sections 7-9."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contracts.real2sim import FeedbackEvent, PhaseSpan, QaSummary, Real2SimFeedback
from contracts.sim_params import SimParams
from skills import residuals as res
from skills.asset_provenance import provenance_events
from skills.attribution import build_hints
from skills.feedback import EventInputs, detect_events, grip_force_stats, phase_at, touch_times
from skills.feedback_md import render_feedback_md
from skills.final_pose import final_pose
from skills.place_check import BURIED_FRAC
from skills.reachability_check import reach_events
from skills.receptacle_check import ReceptacleGeometry, check_place, check_retry
from skills.scene_layout import layout_events


@dataclass
class FeedbackInputs:
    clip: str
    hand: str  # "left" | "right"; indexes axis 1 of wrist_actual_T_sim (left = 0)
    iter: int
    parent_iter: int | None
    params: SimParams
    params_hash: str
    tier_run: str
    plan_frozen: bool
    wall_time_s: float
    arrays: dict[str, np.ndarray]
    probe: dict[str, np.ndarray]
    object_names: list[str]
    held_index: int
    static_flags: list[bool]
    pick_place: dict
    phases_t_s: dict[str, tuple[float, float]]
    release_t_s: float
    grasp_onset_s: float | None = None
    drop_point_m: tuple[float, float, float] | None = None
    demo_carry_m: float = float("nan")
    refs: dict[str, np.ndarray] = field(default_factory=dict)
    sim_pinch_contact: np.ndarray | None = None
    place_buried_frac: float | None = None  # share of the held mesh hidden under the receptacle mesh
    sized_grip_n: float | None = None
    scene_layout: dict | None = None  # report of build_scene_inventory (None: no inventory)
    held_asset: dict | None = None  # provenance of the held object's mesh (`skills.asset_provenance`)
    receptacle: dict | None = None  # ReceptacleGeometry fields + held_half_z_m (RolloutRecord.receptacle)
    place_retry: dict | None = None  # plan's place-goal retry record (bundle meta `place_retry`)
    reach: dict | None = None  # planner reachability of the grasp and place windows (reachability_check)


def _none_if_nan(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def _human_lift(refs: dict[str, np.ndarray]) -> float | None:
    if "grasp_rise_m" in refs:
        return _none_if_nan(float(np.asarray(refs["grasp_rise_m"]).reshape(-1)[0]))
    track = refs.get("ref_held_track_m")
    if track is not None and len(track) >= res.MIN_TRACK_SAMPLES:
        return float(np.max(track[:, 2]) - track[0, 2])
    return None


def _residuals(
    inp: FeedbackInputs,
    t: np.ndarray,
    held_pos: np.ndarray,
    touches: tuple[float, float] | None,
    h_grasp: float | None,
) -> dict[str, float | None]:
    refs, pp = inp.refs, inp.pick_place
    t_grasp = touches[0] if touches else None
    carry = inp.phases_t_s.get("carry")
    ref_t, pinch = refs.get("ref_t_s"), refs.get("ref_human_pinch_m")
    have_pinch = ref_t is not None and pinch is not None and h_grasp is not None
    out: dict[str, float | None] = {k: None for k in res.SCALES}
    if have_pinch and carry and t_grasp is not None:
        out["carry_path_m"] = res.carry_path_m(t, held_pos, carry, t_grasp, h_grasp, ref_t, pinch)
    out["lift_ratio"] = res.lift_ratio(float(pp.get("lift_height_m", 0.0)), _human_lift(refs))
    out["grasp_dt_s"] = res.timing_residual(t_grasp, h_grasp)
    out["release_dt_s"] = res.timing_residual(touches[1] if touches else None, inp.release_t_s)
    offset = None
    if have_pinch and t_grasp is not None:
        h_rel = res._interp(np.array([inp.release_t_s, h_grasp]), ref_t, pinch)
        offset = (held_pos[int(np.searchsorted(t, t_grasp))], h_rel[0] - h_rel[1])
    drop = None if inp.drop_point_m is None else np.asarray(inp.drop_point_m, float)
    out["place_err_m"] = res.place_err_m(held_pos, drop, offset)
    if "ref_held_track_t_s" in refs and "ref_held_track_m" in refs:
        out["held_track_m"] = res.held_track_m(
            t, held_pos, refs["ref_held_track_t_s"], refs["ref_held_track_m"]
        )
    if inp.sim_pinch_contact is not None and "ref_tactile_on" in refs and ref_t is not None:
        out["tactile_iou"] = res.tactile_residual(t, inp.sim_pinch_contact, ref_t, refs["ref_tactile_on"])
    return {k: None if v is None else round(float(v), 5) for k, v in out.items()}


def _receptacle_events(
    inp: FeedbackInputs, t: np.ndarray, held_T: np.ndarray, residuals: dict[str, float | None]
) -> list[FeedbackEvent]:
    """Place-outcome and retry events of ``skills.receptacle_check``; adds ``place_outside_m``."""
    found: list[tuple[str, str]] = []
    if inp.receptacle is not None:
        geo = dict(inp.receptacle)
        half_z = float(geo.pop("held_half_z_m"))
        k = min(int(np.searchsorted(t, inp.release_t_s)), len(t) - 1)
        chk = check_place(held_T[-1, :3, 3], held_T[k, :3, 3], half_z, ReceptacleGeometry(**geo))
        residuals["place_outside_m"] = round(chk.outside_m, 5)
        found += [(kind, chk.detail()) for kind in chk.defects()]
    retry = check_retry(inp.place_retry)
    if retry is not None:
        found.append((retry.kind, retry.detail))
    return [
        FeedbackEvent(
            t_s=round(float(t[-1]), 3),
            phase=phase_at(float(t[-1]), inp.phases_t_s),
            kind=kind,
            object=inp.object_names[inp.held_index],
            detail=detail,
        )
        for kind, detail in found
    ]


def build_feedback(inp: FeedbackInputs) -> Real2SimFeedback:
    """Events, residuals, match score and hints of one rollout.

    ``NO_CONTACT`` (held object never touched) and ``SIM_UNSTABLE`` are added to the failure reasons
    of ``pick_place`` so that scoring and attribution see them.
    """
    a = inp.arrays
    t = np.asarray(a["timestamps_s"], dtype=np.float64)
    hand_index = 0 if inp.hand.lower().startswith("l") else 1
    x = EventInputs(
        t,
        a["object_T_sim"],
        np.asarray(a["object_robot_contact"], bool),
        a["wrist_actual_T_sim"],
        a["q_command_rad"],
        a["q_actual_rad"],
        inp.probe,
        inp.object_names,
        inp.held_index,
        inp.static_flags,
        hand_index,
        inp.pick_place,
        inp.phases_t_s,
        inp.release_t_s,
        inp.sized_grip_n,
    )
    events = detect_events(x)
    touches = touch_times(x.object_robot_contact[:, inp.held_index], t)
    reasons = list(inp.pick_place.get("failure_reasons", ()))
    if touches is None:
        reasons.append("NO_CONTACT")
    if any(e.kind == "SIM_UNSTABLE" for e in events):
        reasons.append("SIM_UNSTABLE")
    success = bool(inp.pick_place.get("success", False)) and not {"NO_CONTACT", "SIM_UNSTABLE"} & set(reasons)
    reasons = tuple(dict.fromkeys(reasons))
    close = inp.phases_t_s.get("close")
    h_grasp = inp.grasp_onset_s if inp.grasp_onset_s is not None else (close[0] if close else None)
    held_pos = x.object_T_sim[:, inp.held_index, :3, 3]
    residuals = _residuals(inp, t, held_pos, touches, h_grasp)
    fp = final_pose(t, x.object_T_sim[:, inp.held_index], bare_table=inp.drop_point_m is None)
    if fp is not None:
        residuals["final_tilt_deg"] = round(fp.tilt_deg, 3)
        residuals["final_drift_m"] = round(fp.drift_m, 5)
        residuals["final_height_m"] = None if fp.height_m is None else round(fp.height_m, 5)
        events = [
            *events,
            *(
                FeedbackEvent(
                    t_s=round(float(t[-1]), 3),
                    phase=phase_at(float(t[-1]), inp.phases_t_s),
                    kind=kind,
                    object=inp.object_names[inp.held_index],
                    detail=fp.detail(),
                )
                for kind in fp.defects()
            ),
        ]
    if inp.place_buried_frac is not None:
        residuals["place_buried_frac"] = round(inp.place_buried_frac, 3)
        if inp.place_buried_frac > BURIED_FRAC:
            events.append(
                FeedbackEvent(
                    t_s=round(float(t[-1]), 3),
                    phase=phase_at(float(t[-1]), inp.phases_t_s),
                    kind="PLACE_BURIED",
                    object=inp.object_names[inp.held_index],
                    detail=f"{100 * inp.place_buried_frac:.0f}% of the object under the receptacle mesh",
                )
            )
    events = [*events, *layout_events(inp.scene_layout)]
    events.extend(provenance_events(inp.held_asset, inp.object_names[inp.held_index]))
    events.extend(_receptacle_events(inp, t, x.object_T_sim[:, inp.held_index], residuals))
    events.extend(
        reach_events(
            inp.reach,
            round(float(t[-1]), 3),
            phase_at(float(t[-1]), inp.phases_t_s),
            inp.object_names[inp.held_index],
        )
    )
    pp = inp.pick_place
    qa = QaSummary(
        success=success,
        failure_reasons=reasons,
        lift_height_m=_none_if_nan(pp.get("lift_height_m")),
        translation_m=_none_if_nan(pp.get("translation_m")),
        expected_translation_m=_none_if_nan(inp.demo_carry_m),
        place_error_m=residuals["place_err_m"],
    )
    stats = grip_force_stats(x, t)
    grip_ratio = stats[0] / inp.sized_grip_n if stats and inp.sized_grip_n else None
    grasp_sign = touches[0] - h_grasp if touches and h_grasp is not None else None
    hints = build_hints(
        inp.params,
        events,
        qa,
        residuals,
        inp.phases_t_s,
        grip_ratio=grip_ratio,
        grasp_sign_s=grasp_sign,
        has_receptacle=inp.drop_point_m is not None,
    )
    phases = tuple(
        PhaseSpan(name=n, t0_s=round(s[0], 3), t1_s=round(s[1], 3))
        for n, s in sorted(inp.phases_t_s.items(), key=lambda kv: kv[1][0])
    )
    return Real2SimFeedback(
        clip=inp.clip,
        hand=inp.hand,
        iter=inp.iter,
        params_hash=inp.params_hash,
        parent_iter=inp.parent_iter,
        tier_run=inp.tier_run,  # type: ignore[arg-type]
        plan_frozen=inp.plan_frozen,
        qa=qa,
        residuals=residuals,
        match_score=round(res.match_score(residuals, success, reasons), 4),
        phases=phases,
        events=tuple(events),
        hints=tuple(hints),
        wall_time_s=inp.wall_time_s,
    )


def write_feedback(
    out_dir: Path, fb: Real2SimFeedback, changed_vs_parent: list[tuple[str, object, object]] | None = None
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "feedback.json", out_dir / "feedback.md"
    json_path.write_text(fb.model_dump_json(indent=1))
    md_path.write_text(render_feedback_md(fb, changed_vs_parent))
    return json_path, md_path
