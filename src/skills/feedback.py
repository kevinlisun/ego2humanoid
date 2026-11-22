"""Spatio-temporal event detectors of the real2sim feedback (Track RSI, spec section 7.3/7.4).

Pure numpy on rollout arrays and ContactProbe arrays; every event carries its phase. Detector
constants live here (not in `SimParams`). Frames: ``T_sim_*`` z up, metres, seconds.
"""

import math
from dataclasses import dataclass

import numpy as np

from common.transforms import rotation_angle
from contracts.real2sim import FeedbackEvent

CONTACT_MERGE_GAP_S = 0.04
LIFT_OFF_M = 0.01
LIFT_OFF_MIN_S = 0.1
SLIP_WINDOW_S = 0.2
SLIP_TRANS_M = 0.005
SLIP_ROT_RAD = math.radians(10.0)
ROTATE_ROT_RAD = math.radians(15.0)
PENETRATION_M = 0.005
PENETRATION_MIN_S = 0.06
GRASP_PEN_M = 0.002  # finger-object overlap that already deforms the grasp (below PENETRATION_M)
GRASP_PEN_MIN_S = 0.04
START_EMBEDDED_M = 0.005  # object-support overlap present at the first frames
START_S = 0.1
TRACKING_LIMIT_RAD = {"arm": 0.05, "hand": 0.15}
TRACKING_MIN_S = 0.1
DISTURBED_M = 0.03  # same bar as scene.inventory.DISTURBED_M (not imported: that module needs MuJoCo)
UNSTABLE_SPEED_M_S = 3.0
UNSTABLE_ANG_SPEED_RAD_S = 50.0
AIRBORNE_M = 0.02
EJECT_SPEED_M_S = 0.4
RELEASE_LEAD_S = 0.2
MAX_EVENTS = 40
HOLD_CARRY = ("hold", "carry")

SEVERITY = {
    "SIM_UNSTABLE": 10,
    "EJECT": 9,
    "DROP": 9,
    "AIRBORNE_RELEASE": 8,
    "STUCK": 8,
    "SLIP": 7,
    "ROTATE_IN_GRASP": 6,
    "PENETRATION": 6,
    "GRASP_PENETRATION": 6,
    "START_EMBEDDED": 6,
    "PLACE_BURIED": 7,
    "SCENE_ROLE_SWAP": 7,
    "SCENE_RECEPTACLE_REBASED": 6,
    "SCENE_GRIP_GAP": 6,
    "HELD_MESH_FALLBACK_BOX": 6,
    "PLACE_OUTSIDE_RECEPTACLE": 7,
    "RIM_BLOCKED": 7,
    "RETRY_WORSE": 3,
    "RETRY_BUDGET_EXHAUSTED": 3,
    "PLACE_UNREACHABLE": 7,
    "STANCE_GRASP_ONLY": 6,
    "DISTURBED": 6,
    "TRACKING_ERROR": 5,
    "GRIP_FORCE": 5,
    "FIRST_TOUCH": 4,
    "LAST_TOUCH": 4,
    "LIFT_OFF": 4,
    "CONTACT_OFF": 2,
    "CONTACT_ON": 2,
}


@dataclass(frozen=True)
class EventInputs:
    timestamps_s: np.ndarray  # (T,)
    object_T_sim: np.ndarray  # (T, n, 4, 4)
    object_robot_contact: np.ndarray  # (T, n) bool
    wrist_T_sim: np.ndarray  # (T, 2, 4, 4)
    q_command_rad: np.ndarray  # (T, A)
    q_actual_rad: np.ndarray  # (T, A)
    probe: dict[str, np.ndarray]
    object_names: list[str]
    held_index: int
    static_flags: list[bool]
    hand_index: int
    pick_place: dict
    phases: dict[str, tuple[float, float]]
    release_t_s: float | None
    sized_grip_n: float | None = None


def phase_at(t_s: float, phases: dict[str, tuple[float, float]]) -> str:
    for name, (t0, t1) in sorted(phases.items(), key=lambda kv: kv[1][0]):
        if t0 <= t_s < t1:
            return name
    return "idle"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, bool).astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def contact_runs(
    mask: np.ndarray, t: np.ndarray, gap_s: float = CONTACT_MERGE_GAP_S
) -> list[tuple[int, int]]:
    """True runs of ``mask`` merged across gaps shorter than ``gap_s``."""
    merged: list[tuple[int, int]] = []
    for a, b in _runs(mask):
        if merged and t[a] - t[merged[-1][1] - 1] < gap_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def touch_times(contact: np.ndarray, t: np.ndarray) -> tuple[float, float] | None:
    """(first touch, last touch) times of a contact mask, or None if never touched."""
    runs = contact_runs(contact, t)
    if not runs:
        return None
    return float(t[runs[0][0]]), float(t[min(runs[-1][1], len(t)) - 1])


def _long_runs(mask: np.ndarray, dt: float, min_s: float) -> list[tuple[int, int]]:
    return [(a, b) for a, b in _runs(mask) if (b - a) * dt >= min_s - 1e-9]


def _ev(t: np.ndarray, i: int, ph: dict, kind: str, obj: str | None, detail: str) -> FeedbackEvent:
    ts = float(t[min(i, len(t) - 1)])
    return FeedbackEvent(t_s=round(ts, 3), phase=phase_at(ts, ph), kind=kind, object=obj, detail=detail)


def _contact_events(x: EventInputs, t: np.ndarray) -> list[FeedbackEvent]:
    out: list[FeedbackEvent] = []
    for k, name in enumerate(x.object_names):
        runs = contact_runs(x.object_robot_contact[:, k], t)
        held = k == x.held_index
        for r, (a, b) in enumerate(runs):
            if held and r == 0:
                out.append(_ev(t, a, x.phases, "FIRST_TOUCH", name, ""))
            elif not held or r > 0:
                out.append(_ev(t, a, x.phases, "CONTACT_ON", name, ""))
            if held and r == len(runs) - 1:
                out.append(_ev(t, b - 1, x.phases, "LAST_TOUCH", name, ""))
            elif not held or r < len(runs) - 1:
                out.append(_ev(t, b, x.phases, "CONTACT_OFF", name, ""))
    return out


def _lift_off(x: EventInputs, t: np.ndarray, dt: float) -> list[FeedbackEvent]:
    z = x.object_T_sim[:, x.held_index, 2, 3]
    rest = z[t <= t[0] + 0.1].mean()
    runs = _long_runs(z > rest + LIFT_OFF_M, dt, LIFT_OFF_MIN_S)
    if not runs:
        return []
    name = x.object_names[x.held_index]
    return [_ev(t, runs[0][0], x.phases, "LIFT_OFF", name, f"peak {1000 * (z.max() - rest):.0f} mm")]


def _slip_events(x: EventInputs, t: np.ndarray, dt: float) -> list[FeedbackEvent]:
    n, w = len(t), max(1, round(SLIP_WINDOW_S / dt))
    if n <= w:
        return []
    h = x.held_index
    rel = np.linalg.inv(x.wrist_T_sim[:, x.hand_index]) @ x.object_T_sim[:, h]
    p, R = rel[:, :3, 3], rel[:, :3, :3]
    drift = np.linalg.norm(p[w:] - p[:-w], axis=1)
    rot = rotation_angle(np.swapaxes(R[:-w], 1, 2) @ R[w:])
    cs = np.r_[0, np.cumsum(x.object_robot_contact[:, h])]
    held_in_contact = cs[w + 1 :] - cs[: n - w] == w + 1
    in_phase = np.array([phase_at(float(s), x.phases) in HOLD_CARRY for s in t[: n - w]])
    ok = held_in_contact & in_phase
    rotate = ok & (rot > ROTATE_ROT_RAD) & (drift < SLIP_TRANS_M)
    near_rotate = np.convolve(rotate, np.ones(2 * w + 1), mode="same") > 0  # ramp-up/down of a rotation
    slip = ok & ((drift > SLIP_TRANS_M) | ((rot > SLIP_ROT_RAD) & ~near_rotate)) & ~rotate
    name = x.object_names[h]
    out = []
    for mask, kind in ((slip, "SLIP"), (rotate, "ROTATE_IN_GRASP")):
        for a, b in _runs(mask):
            i = a + int(np.argmax(drift[a:b] if kind == "SLIP" else rot[a:b]))
            if kind == "SLIP":
                d = p[i + w] - p[i]
                ax = int(np.argmax(np.abs(d)))
                sign = "+" if d[ax] >= 0 else "-"
                detail = f"{1000 * drift[i]:.0f} mm along wrist {sign}{'xyz'[ax]} in {SLIP_WINDOW_S:g} s"
            else:
                detail = f"{math.degrees(rot[i]):.0f} deg in {SLIP_WINDOW_S:g} s, {1000 * drift[i]:.1f} mm"
            out.append(_ev(t, a, x.phases, kind, name, detail))
    return out


def _pick_place_events(x: EventInputs, t: np.ndarray) -> list[FeedbackEvent]:
    pp = x.pick_place
    reasons = set(pp.get("failure_reasons", ()))
    h = x.held_index
    name = x.object_names[h]
    z = x.object_T_sim[:, h, 2, 3]
    rest = z[t <= t[0] + 0.1].mean()
    contact = x.object_robot_contact[:, h]
    airborne = z - rest > AIRBORNE_M
    out = []
    if "EJECTED" in reasons:
        pos = x.object_T_sim[:, h, :2, 3]
        v = np.r_[np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t), 0.0]
        free = airborne & ~contact & (v > EJECT_SPEED_M_S)
        i = int(np.argmax(free)) if free.any() else int(np.argmax(airborne))
        out.append(_ev(t, i, x.phases, "EJECT", name, f"{pp.get('ejection_speed_m_s', math.nan):.2f} m/s"))
    if "DROPPED_IN_CARRY" in reasons and airborne.any():
        lift_k = int(np.argmax(airborne))
        end = x.release_t_s - RELEASE_LEAD_S if x.release_t_s is not None else float(t[-1])
        lost = np.flatnonzero((np.arange(len(t)) >= lift_k) & (t < end) & ~contact)
        i = int(lost[0]) if len(lost) else lift_k
        out.append(
            _ev(
                t,
                i,
                x.phases,
                "DROP",
                name,
                f"z {100 * (z[i] - rest):.0f} cm, gap {pp.get('carry_gap_s', 0):.2f} s",
            )
        )
    if x.release_t_s is not None:
        t_end = x.release_t_s + float(np.nan_to_num(pp.get("contact_after_release_s", 0.0)))
        i = int(np.searchsorted(t, t_end))
        if "AIRBORNE_RELEASE" in reasons:
            detail = f"let go {1000 * pp.get('release_height_m', math.nan):.0f} mm above rest"
            out.append(_ev(t, i, x.phases, "AIRBORNE_RELEASE", name, detail))
        if "STUCK_AFTER_RELEASE" in reasons:
            after = pp.get("contact_after_release_s", 0)
            drag = 1000 * pp.get("post_release_drag_m", 0)
            detail = f"contact {after:.2f} s after release, drag {drag:.0f} mm"
            out.append(_ev(t, i, x.phases, "STUCK", name, detail))
    return out


def _penetration_events(x: EventInputs, t: np.ndarray, dt: float) -> list[FeedbackEvent]:
    pr = x.probe
    series: list[tuple[str, str | None, np.ndarray]] = []
    if "pen_robot_table_m" in pr:
        series.append(("robot-table", None, pr["pen_robot_table_m"]))
    for key, label in (("pen_robot_object_m", "robot-object"), ("pen_object_support_m", "object-support")):
        for k, name in enumerate(x.object_names) if key in pr else ():
            series.append((label, name, pr[key][:, k]))
    out = []
    for label, name, pen in series:
        pen = pen[: len(t)]
        for a, b in _long_runs(pen > PENETRATION_M, dt, PENETRATION_MIN_S):
            detail = f"{label} peak {1000 * pen[a:b].max():.1f} mm for {(b - a) * dt:.2f} s"
            out.append(_ev(t, a, x.phases, "PENETRATION", name, detail))
    return out


def _grasp_penetration_events(x: EventInputs, t: np.ndarray, dt: float) -> list[FeedbackEvent]:
    """Finger-held-object overlap during approach/close/hold, from `GRASP_PEN_M` (PENETRATION needs 5 mm)."""
    pen = x.probe.get("pen_robot_object_m")
    spans = [x.phases[k] for k in ("approach", "close", "hold") if k in x.phases]
    if pen is None or not spans:
        return []
    series = pen[: len(t), x.held_index]
    window = np.zeros(len(t), bool)
    for t0, t1 in spans:
        window |= (t >= t0) & (t < t1)
    out = []
    for a, b in _long_runs((series > GRASP_PEN_M) & window, dt, GRASP_PEN_MIN_S):
        detail = f"finger-object peak {1000 * series[a:b].max():.1f} mm for {(b - a) * dt:.2f} s"
        out.append(_ev(t, a, x.phases, "GRASP_PENETRATION", x.object_names[x.held_index], detail))
    return out


def _start_embedded_events(x: EventInputs, t: np.ndarray) -> list[FeedbackEvent]:
    """An object that starts overlapping another (clip 10: the pick sits in the bowl mesh) is pushed out."""
    pen = x.probe.get("pen_object_support_m")
    if pen is None:
        return []
    first = t <= t[0] + START_S
    out = []
    for k, name in enumerate(x.object_names):
        peak = float(np.nanmax(pen[: len(t), k][first]))
        if peak > START_EMBEDDED_M:
            out.append(
                _ev(t, 0, x.phases, "START_EMBEDDED", name, f"starts {1000 * peak:.1f} mm inside a support")
            )
    return out


def _tracking_events(x: EventInputs, t: np.ndarray, dt: float) -> list[FeedbackEvent]:
    err = np.abs(x.q_actual_rad[: len(t)] - x.q_command_rad[: len(t)])
    group = np.asarray(x.probe.get("joint_group", np.zeros(err.shape[1], int)))
    out = []
    for gid, label in ((0, "arm"), (1, "hand")):
        cols = group == gid
        if not cols.any():
            continue
        peak = err[:, cols].max(axis=1)
        for a, b in _long_runs(peak > TRACKING_LIMIT_RAD[label], dt, TRACKING_MIN_S):
            detail = f"{label} joints peak {peak[a:b].max():.3f} rad for {(b - a) * dt:.2f} s"
            out.append(_ev(t, a, x.phases, "TRACKING_ERROR", None, detail))
    return out


def grip_force_stats(x: EventInputs, t: np.ndarray) -> tuple[float, float] | None:
    """(median, min) of the held-object grip force over the hold phase, or None."""
    force = x.probe.get("force_robot_object_n")
    if force is None or "hold" not in x.phases:
        return None
    t0, t1 = x.phases["hold"]
    sel = (t >= t0) & (t < t1)
    if not sel.any():
        return None
    f = force[: len(t)][sel, x.held_index]
    return float(np.median(f)), float(f.min())


def _grip_event(x: EventInputs, t: np.ndarray) -> list[FeedbackEvent]:
    stats = grip_force_stats(x, t)
    if stats is None:
        return []
    sized = "" if x.sized_grip_n is None else f" (sized {x.sized_grip_n:.1f} N)"
    detail = f"grip force median {stats[0]:.1f} N, min {stats[1]:.1f} N{sized}"
    i = int(np.searchsorted(t, x.phases["hold"][0]))
    return [_ev(t, i, x.phases, "GRIP_FORCE", x.object_names[x.held_index], detail)]


def _disturbed_events(x: EventInputs, t: np.ndarray) -> list[FeedbackEvent]:
    out = []
    for k, name in enumerate(x.object_names):
        if k == x.held_index or x.static_flags[k]:
            continue
        p = x.object_T_sim[:, k, :3, 3]
        moved = np.linalg.norm(p[:, :2] - p[0, :2], axis=1) > DISTURBED_M
        if moved.any():
            i = int(np.argmax(moved))
            detail = f"moved {1000 * np.linalg.norm(p[-1, :2] - p[0, :2]):.0f} mm (final xy)"
            out.append(_ev(t, i, x.phases, "DISTURBED", name, detail))
    return out


def _unstable_event(x: EventInputs, t: np.ndarray) -> FeedbackEvent | None:
    n = len(x.timestamps_s)
    finite = np.asarray(x.probe.get("finite", np.ones(n, bool)), bool)
    bad = ~finite
    speed = x.probe.get("obj_speed_m_s")
    ang = x.probe.get("obj_ang_speed_rad_s")
    fast = np.zeros(n, bool)
    if speed is not None:
        fast |= (np.nan_to_num(speed, nan=np.inf) > UNSTABLE_SPEED_M_S).any(axis=1)
    if ang is not None:
        fast |= (np.nan_to_num(ang, nan=np.inf) > UNSTABLE_ANG_SPEED_RAD_S).any(axis=1)
    if not (bad | fast).any():
        return None
    i = int(np.argmax(bad | fast))
    detail = "non-finite state" if bad[i] else "object speed beyond limits"
    return _ev(x.timestamps_s, i, x.phases, "SIM_UNSTABLE", None, detail)


def merge_and_cap(events: list[FeedbackEvent], cap: int = MAX_EVENTS) -> list[FeedbackEvent]:
    """Time-sorted timeline without consecutive duplicates, capped to the ``cap`` most severe."""
    ordered = sorted(events, key=lambda e: (e.t_s, -SEVERITY.get(e.kind, 0)))
    merged: list[FeedbackEvent] = []
    for e in ordered:
        if merged and (merged[-1].kind, merged[-1].object, merged[-1].phase, merged[-1].detail) == (
            e.kind,
            e.object,
            e.phase,
            e.detail,
        ):
            continue
        merged.append(e)
    if len(merged) > cap:
        keep = sorted(range(len(merged)), key=lambda i: (-SEVERITY.get(merged[i].kind, 0), i))[:cap]
        merged = [merged[i] for i in sorted(keep)]
    return merged


def detect_events(x: EventInputs) -> list[FeedbackEvent]:
    """All feedback events of one rollout, merged, time-sorted and capped."""
    unstable = _unstable_event(x, x.timestamps_s)
    n_ok = len(x.timestamps_s)
    finite = np.asarray(x.probe.get("finite", np.ones(n_ok, bool)), bool)
    if not finite.all():
        n_ok = int(np.argmin(finite))
    events = [unstable] if unstable else []
    if n_ok >= 3:
        t = np.asarray(x.timestamps_s[:n_ok], dtype=np.float64)
        dt = float(np.median(np.diff(t)))
        y = EventInputs(
            t,
            x.object_T_sim[:n_ok],
            x.object_robot_contact[:n_ok],
            x.wrist_T_sim[:n_ok],
            x.q_command_rad[:n_ok],
            x.q_actual_rad[:n_ok],
            {k: v if k == "joint_group" else v[:n_ok] for k, v in x.probe.items()},
            x.object_names,
            x.held_index,
            x.static_flags,
            x.hand_index,
            x.pick_place,
            x.phases,
            x.release_t_s,
            x.sized_grip_n,
        )
        events += _contact_events(y, t) + _lift_off(y, t, dt) + _slip_events(y, t, dt)
        events += _pick_place_events(y, t) + _penetration_events(y, t, dt) + _tracking_events(y, t, dt)
        events += _grip_event(y, t) + _disturbed_events(y, t)
        events += _grasp_penetration_events(y, t, dt) + _start_embedded_events(y, t)
    return merge_and_cap(events)
