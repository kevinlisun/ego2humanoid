"""Attribution hints: failure patterns -> suggested `SimParams` knobs (Track RSI, spec section 8).

One rule table; each emitted hint carries the current value, hard range and re-run cost
(`affects`). A rule whose path does not exist in `SimParams` is skipped. Hints are suggestions.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from contracts.real2sim import FeedbackEvent, FeedbackHint, QaSummary
from contracts.sim_params import ObjectPhysics, SimParams
from skills.params.resolve import _affects

HELD_NAME = "perceived"  # same as replay.sim_eval.HELD_NAME (not imported: that module needs MuJoCo)
EJECT_CLOSE_MARGIN_S = 0.2
GRIP_LOW_RATIO = 0.5
GRASP_DT_HINT_S = 0.15
LIFT_RATIO_HINT = 0.25

# Ranges the model does not bound (or only per element).
RANGE_OVERRIDES: dict[str, tuple[float, float]] = {
    "contact.solref": (0.002, 0.1),  # time constant (element 0)
    "contact.condim": (1, 6),
    "pd.groups.hand.kp": (1, 50),
    "pd.groups.hand.max_torque_nm": (1, 30),
}


@dataclass(frozen=True)
class Rule:
    param: str  # dotted path; `{name}` is filled with the object selector
    direction: str
    why: str


RULES: dict[str, tuple[Rule, ...]] = {
    "SIM_UNSTABLE": (
        Rule("contact.solref", "up", "longer time constant (element 0) = softer, more stable contacts"),
        Rule("pd.timestep_s", "down", "smaller physics step"),
        Rule("contact.impratio", "down", "lower friction/normal impedance ratio"),
    ),
    "NO_CONTACT": (
        Rule("objects.held.pos_offset_m", "tune", "shift the held object towards the hand path"),
        Rule("trajectory.start_search_m", "up", "search a wider grid of start offsets"),
        Rule("trajectory.min_object_thickness_m", "up", "pinch otherwise lands on the table"),
        Rule("trajectory.hover_clearance_m", "down", "close lower over the object"),
    ),
    "EJECT_AT_CLOSE": (
        Rule("trajectory.squeeze_m", "down", "gentler squeeze at the close"),
        Rule("contact.solref", "up", "softer contact (time constant, element 0)"),
        Rule("objects.held.mass_kg", "up", "heavier object resists being squeezed out"),
        Rule("contact.condim", "set", "set to 6 for torsional and rolling friction"),
    ),
    "SLIP": (
        Rule("objects.held.friction", "up", "more grip friction"),
        Rule("contact.min_slide_friction", "up", "higher friction floor"),
        Rule("trajectory.grip_min_n", "up", "size a stronger grip"),
        Rule("trajectory.carry_accel_m_s2", "down", "gentler carry"),
        Rule("contact.noslip_iterations", "up", "stiffer friction solve"),
        Rule("pd.groups.hand.kp", "up", "stiffer fingers hold the squeeze"),
    ),
    "ROTATE_IN_GRASP": (
        Rule("contact.torsion_m", "up", "more torsional friction"),
        Rule("contact.condim", "set", "set to 6 for torsional and rolling friction"),
    ),
    "GRIP_FORCE_LOW": (
        Rule("pd.groups.hand.kp", "up", "grip force well below the sized force"),
        Rule("pd.groups.hand.max_torque_nm", "up", "finger torque may be saturating"),
        Rule("trajectory.squeeze_m", "up", "squeeze further to raise the grip force"),
    ),
    "STUCK_AFTER_RELEASE": (
        Rule("trajectory.release_open_frames", "down", "open the fingers faster"),
        Rule("trajectory.retreat_height_m", "up", "retreat higher"),
        Rule("trajectory.open_pad_m", "up", "wider opening"),
    ),
    "AIRBORNE_RELEASE": (
        Rule("trajectory.place_dwell_frames", "up", "hold still longer before letting go"),
        Rule("trajectory.bare_place_lift_m", "down", "put down lower"),
    ),
    "NOT_PLACED": (
        Rule("trajectory.place_lead_frames", "tune", "receptacle frame takes over earlier or later"),
        Rule("objects.receptacle.pos_offset_m", "tune", "move the receptacle under the drop point"),
    ),
    "DISTURBED": (
        Rule("objects.{name}.static", "set", "weld the object so it is not pushed"),
        Rule("objects.{name}.mass_kg", "up", "heavier object is pushed less"),
    ),
    "PENETRATION_ROBOT_TABLE": (
        Rule("table.top_z_m", "down", "lower the table top"),
        Rule("trajectory.table_near_edge_m", "up", "keep the table farther from the robot"),
    ),
    "GRASP_LATE": (Rule("trajectory.close_lead_frames", "up", "close earlier; sim grasps after the human"),),
    "GRASP_EARLY": (
        Rule("trajectory.close_lead_frames", "down", "close later; sim grasps before the human"),
    ),
    "LIFT_RATIO": (
        Rule("trajectory.max_hover_rise_m", "tune", "sim lift differs from the human lift"),
        Rule("trajectory.min_object_thickness_m", "tune", "object height assumption changes the lift"),
    ),
    "FINAL_TILTED": (
        Rule("trajectory.release_open_frames", "up", "open slowly so a finger does not tip the object"),
        Rule("trajectory.bare_place_lift_m", "down", "put down lower, less tipping on touchdown"),
        Rule("trajectory.place_dwell_frames", "up", "hold still longer before letting go"),
        Rule("trajectory.retreat_height_m", "up", "retreat higher so a finger does not drag the object"),
        Rule("contact.condim", "set", "set to 6 for torsional and rolling friction"),
    ),
    "FINAL_NOT_AT_REST": (
        Rule("trajectory.retreat_frames", "up", "give the object more time to settle"),
        Rule("contact.solref", "up", "softer, better damped contact (time constant, element 0)"),
        Rule("pd.timestep_s", "down", "smaller physics step"),
        Rule("contact.noslip_iterations", "up", "stiffer friction solve stops creeping"),
    ),
    "FINAL_HEIGHT": (
        Rule("trajectory.bare_place_lift_m", "down", "object rests above or below its start height"),
        Rule("trajectory.place_dwell_frames", "up", "hold still longer before letting go"),
        Rule("table.top_z_m", "tune", "support height differs from the object's start support"),
    ),
    "GRASP_PENETRATION": (
        Rule("trajectory.squeeze_m", "down", "less squeeze drives the pads less far into the object"),
        Rule("trajectory.open_margin_m", "up", "open wider so the pads do not land on the object"),
        Rule("trajectory.close_ramp_frames", "up", "close slower, softer first contact"),
        Rule("pd.groups.hand.kp", "down", "weaker fingers press less into the object"),
        Rule("contact.solref", "down", "stiffer contact (time constant, element 0) = less overlap"),
        Rule("contact.solimp", "tune", "contact impedance width bounds the overlap"),
    ),
    "START_EMBEDDED": (
        Rule("objects.{name}.pos_offset_m", "tune", "move the object out of the support it starts inside"),
        Rule("objects.{name}.size_scale", "tune", "shrink the proxy that overlaps its neighbour"),
        Rule(
            "objects.{name}.robot_collision",
            "set",
            "the container may skip the hand",
        ),
    ),
    "PLACE_BURIED": (
        Rule("trajectory.place_clearance_m", "up", "release higher above the receptacle surface"),
        Rule("trajectory.bare_place_lift_m", "up", "put the object down higher on a bare table"),
        Rule("trajectory.held_clearance_m", "up", "carry clear of the receptacle mesh"),
        Rule("objects.receptacle.pos_offset_m", "tune", "move the receptacle off the drop point"),
        Rule("objects.receptacle.size_scale", "tune", "receptacle proxy larger than its visible surface"),
        Rule("trajectory.place_lead_frames", "tune", "receptacle frame takes over earlier or later"),
    ),
    "SCENE_ROLE_SWAP": (
        Rule(
            "scene.source_from", "set", "set to `name`: trust the task's source name over the grip geometry"
        ),
        Rule("scene.max_registration_m", "up", "a name-based source can lie farther from the grip"),
    ),
    "SCENE_RECEPTACLE_REBASED": (
        Rule("scene.source_from", "set", "a swapped source/receptacle role rebases the wrong table"),
        Rule("objects.receptacle.pos_offset_m", "tune", "move the receptacle back towards the video layout"),
    ),
    "SCENE_GRIP_GAP": (
        Rule(
            "scene.max_registration_m",
            "up",
            "trust a larger grip-to-source shift instead of dropping the layout",
        ),
        Rule("scene.source_from", "set", "the other evidence (name or grip) may pick the right source"),
    ),
    "HELD_MESH_FALLBACK_BOX": (
        Rule(
            "trajectory.allow_held_box_fallback",
            "set",
            "set to false after running the held_mesh stage (needs HF_HOME): a box is not the object",
        ),
        Rule("objects.held.size_scale", "tune", "the default box is sized from the point cloud only"),
    ),
    "PLACE_OUTSIDE_RECEPTACLE": (
        Rule("trajectory.receptacle_opening_margin_m", "up", "widen the collision opening to enter"),
        Rule("objects.receptacle.pos_offset_m", "tune", "move the receptacle under the drop point"),
        Rule("trajectory.max_offset_m", "up", "let the place correction shift the hand further"),
    ),
    "RIM_BLOCKED": (
        Rule("trajectory.receptacle_opening_margin_m", "up", "the block lags its command, catches rim"),
        Rule("trajectory.place_clearance_m", "up", "release higher so the block clears the rim"),
        Rule("trajectory.place_xy_lead_frames", "up", "be over the opening before the descent"),
        Rule("objects.receptacle.size_scale", "tune", "larger or lower collision shell than the mesh"),
    ),
    "RETRY_WORSE": (
        Rule("trajectory.max_offset_m", "down", "a smaller place correction the stance can reach"),
        Rule("trajectory.place_xy_lead_frames", "tune", "place correction fades in over other frames"),
    ),
    "RETRY_BUDGET_EXHAUSTED": (
        Rule("trajectory.place_retry_budget_s", "up", "allow the place-goal retry more wall time"),
        Rule("trajectory.max_offset_m", "down", "a smaller place correction needs no retry"),
    ),
    "PLACE_UNREACHABLE": (
        Rule("trajectory.place_reach_weight_m", "up", "weigh the place window more when choosing the stance"),
        Rule("trajectory.place_stance_search", "set", "set to true to re-score the stance on the place path"),
        Rule("trajectory.max_offset_m", "up", "let the place shift pull the hand further"),
        Rule("objects.receptacle.pos_offset_m", "tune", "move the receptacle towards the arm"),
    ),
    "STANCE_GRASP_ONLY": (
        Rule("trajectory.place_stance_search", "set", "set to true to re-score the stance on the place path"),
        Rule("trajectory.place_reach_weight_m", "up", "weigh the place window more when choosing the stance"),
    ),
    "TRACKING_ERROR_HAND": (
        Rule("pd.groups.hand.kp", "up", "fingers lag the command under load"),
        Rule("pd.groups.hand.max_torque_nm", "up", "fingers lag the command under load"),
    ),
}


def _field_info(path: str) -> FieldInfo | None:
    """FieldInfo of a dotted SimParams leaf (`objects.<selector>.<field>` -> ObjectPhysics)."""
    parts = path.split(".")
    model: Any = SimParams
    if parts[0] == "objects":
        if len(parts) != 3:
            return None
        parts, model = parts[2:], ObjectPhysics
    info = None
    for part in parts:
        if not (isinstance(model, type) and issubclass(model, BaseModel)) or part not in model.model_fields:
            return None
        info = model.model_fields[part]
        model = info.annotation
    return info


def param_exists(path: str) -> bool:
    return _field_info(path) is not None


def param_affects(path: str) -> tuple[str, ...]:
    """`affects` of a dotted path as a sorted tuple, e.g. ("plan", "sim")."""
    return tuple(sorted(_affects(path), reverse=True))


def get_param(params: SimParams, path: str) -> Any:
    """Current value at a dotted path; `objects.<sel>.*` falls back to the ObjectPhysics default."""
    parts = path.split(".")
    if parts[0] == "objects":
        obj = params.objects.get(parts[1]) or ObjectPhysics()
        return getattr(obj, parts[2])
    cur: Any = params
    for part in parts:
        cur = getattr(cur, part)
    return list(cur) if isinstance(cur, tuple) else cur


def param_range(path: str) -> tuple[float, float] | None:
    if path in RANGE_OVERRIDES:
        return RANGE_OVERRIDES[path]
    info = _field_info(path)
    if info is None:
        return None
    lo = hi = None
    for m in info.metadata:
        lo = getattr(m, "ge", getattr(m, "gt", None)) if lo is None else lo
        hi = getattr(m, "le", getattr(m, "lt", None)) if hi is None else hi
    return None if lo is None or hi is None else (float(lo), float(hi))


def _make_hint(params: SimParams, pattern: str, rule: Rule, name: str | None) -> FeedbackHint | None:
    path = rule.param.format(name=name) if name else rule.param
    if not param_exists(path):
        return None
    return FeedbackHint(
        pattern=pattern,
        param=path,
        current=get_param(params, path),
        range=param_range(path),
        direction=rule.direction,  # type: ignore[arg-type]
        affects=param_affects(path),
        why=rule.why,
    )


def fired_patterns(
    events: Iterable[FeedbackEvent],
    qa: QaSummary,
    residuals: dict[str, float | None],
    phases: dict[str, tuple[float, float]],
    *,
    grip_ratio: float | None = None,
    grasp_sign_s: float | None = None,
    has_receptacle: bool = False,
) -> list[tuple[str, str | None]]:
    """(pattern, object name or None) of every firing rule; `grasp_sign_s` = t_grasp_sim - t_grasp_human."""
    events = list(events)
    reasons = set(qa.failure_reasons)
    kinds = {e.kind for e in events}
    out: list[tuple[str, str | None]] = []
    if "SIM_UNSTABLE" in kinds:
        out.append(("SIM_UNSTABLE", None))
    if "NO_CONTACT" in reasons:
        out.append(("NO_CONTACT", None))
    close = phases.get("close")
    if close and any(
        e.kind == "EJECT" and close[0] - EJECT_CLOSE_MARGIN_S <= e.t_s <= close[1] + EJECT_CLOSE_MARGIN_S
        for e in events
    ):
        out.append(("EJECT_AT_CLOSE", None))
    if "SLIP" in kinds or "DROPPED_IN_CARRY" in reasons:
        out.append(("SLIP", None))
    if "ROTATE_IN_GRASP" in kinds:
        out.append(("ROTATE_IN_GRASP", None))
    if grip_ratio is not None and grip_ratio < GRIP_LOW_RATIO:
        out.append(("GRIP_FORCE_LOW", None))
    if "STUCK_AFTER_RELEASE" in reasons:
        out.append(("STUCK_AFTER_RELEASE", None))
    if "AIRBORNE_RELEASE" in reasons:
        out.append(("AIRBORNE_RELEASE", None))
    if "NOT_PLACED" in reasons and has_receptacle:
        out.append(("NOT_PLACED", None))
    seen: set[str] = set()
    for e in events:
        if e.kind == "DISTURBED" and e.object and e.object not in seen:
            seen.add(e.object)
            out.append(("DISTURBED", e.object))
    if any(e.kind == "PENETRATION" and e.detail.startswith("robot-table") for e in events):
        out.append(("PENETRATION_ROBOT_TABLE", None))
    dt = residuals.get("grasp_dt_s")
    if dt is not None and dt > GRASP_DT_HINT_S and grasp_sign_s:
        out.append(("GRASP_LATE" if grasp_sign_s > 0 else "GRASP_EARLY", None))
    lr = residuals.get("lift_ratio")
    if qa.success and lr is not None and lr > LIFT_RATIO_HINT:
        out.append(("LIFT_RATIO", None))
    if any(e.kind == "GRASP_PENETRATION" for e in events):
        out.append(("GRASP_PENETRATION", None))
    starts = {e.object for e in events if e.kind == "START_EMBEDDED" and e.object}
    for name in sorted(starts):
        out.append(("START_EMBEDDED", "held" if name == HELD_NAME else name))
    for kind in (
        "SCENE_ROLE_SWAP",
        "SCENE_RECEPTACLE_REBASED",
        "SCENE_GRIP_GAP",
        "PLACE_BURIED",
        "PLACE_BURIED",
        "PLACE_OUTSIDE_RECEPTACLE",
        "RIM_BLOCKED",
        "RETRY_WORSE",
        "RETRY_BUDGET_EXHAUSTED",
        "PLACE_BURIED",
        "PLACE_UNREACHABLE",
        "STANCE_GRASP_ONLY",
        "FINAL_TILTED",
        "FINAL_NOT_AT_REST",
        "FINAL_HEIGHT",
    ):
        if any(e.kind == kind for e in events):
            out.append((kind, None))
    if "HELD_MESH_FALLBACK_BOX" in kinds:
        out.append(("HELD_MESH_FALLBACK_BOX", None))
    if any(e.kind == "TRACKING_ERROR" and "hand" in e.detail for e in events):
        out.append(("TRACKING_ERROR_HAND", None))
    return out


def build_hints(
    params: SimParams,
    events: Iterable[FeedbackEvent],
    qa: QaSummary,
    residuals: dict[str, float | None],
    phases: dict[str, tuple[float, float]],
    *,
    grip_ratio: float | None = None,
    grasp_sign_s: float | None = None,
    has_receptacle: bool = False,
) -> list[FeedbackHint]:
    """Hints for the patterns that fired, in rule-table order; duplicate (pattern, param) dropped."""
    hints: list[FeedbackHint] = []
    seen: set[tuple[str, str]] = set()
    fired = fired_patterns(
        events,
        qa,
        residuals,
        phases,
        grip_ratio=grip_ratio,
        grasp_sign_s=grasp_sign_s,
        has_receptacle=has_receptacle,
    )
    for pattern, name in fired:
        rule_key = pattern
        for rule in RULES[rule_key]:
            hint = _make_hint(params, pattern, rule, name)
            if hint and (pattern, hint.param) not in seen:
                seen.add((pattern, hint.param))
                hints.append(hint)
    return hints
