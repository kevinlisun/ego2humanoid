"""Sim-vs-video residuals and the scalar match score (Track RSI, spec section 7.5/7.6).

Displacement-based so that monocular absolute offsets cancel. Weights and scales are code
constants, not agent-tunable. Lower is better.
"""

import numpy as np

SCALES = {
    "carry_path_m": 0.05,
    "lift_ratio": 0.5,
    "grasp_dt_s": 0.3,
    "release_dt_s": 0.3,
    "place_err_m": 0.05,
    "held_track_m": 0.08,
    "tactile_iou": 1.0,
    "final_tilt_deg": 15.0,
    "final_drift_m": 0.01,
    "final_height_m": 0.02,
    "place_buried_frac": 0.3,
    "place_outside_m": 0.05,
}
WEIGHTS = {
    "carry_path_m": 3.0,
    "lift_ratio": 1.0,
    "grasp_dt_s": 2.0,
    "release_dt_s": 1.0,
    "place_err_m": 3.0,
    "held_track_m": 1.0,
    "tactile_iou": 1.0,
    "final_tilt_deg": 2.0,
    "final_drift_m": 2.0,
    "final_height_m": 1.5,
    "place_buried_frac": 2.0,
    "place_outside_m": 1.0,
}
SEVERITY = {
    "SIM_UNSTABLE": 0.9,
    "NO_CONTACT": 0.8,
    "NO_LIFT": 0.7,
    "EJECTED": 0.6,
    "DROPPED_IN_CARRY": 0.5,
    "NOT_PLACED": 0.3,
    "AIRBORNE_RELEASE": 0.2,
    "STUCK_AFTER_RELEASE": 0.1,
}
DEFAULT_SEVERITY = 0.4  # failure reasons without a table entry (NO_TRANSLATION, NOT_SETTLED, ...)
MIN_TRACK_SAMPLES = 5


def _interp(t_q: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_q, t, v[:, k]) for k in range(v.shape[1])], axis=1)


def _finite_or_none(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def carry_path_m(
    t: np.ndarray,
    held_pos: np.ndarray,
    carry: tuple[float, float],
    t_grasp: float,
    h_grasp: float,
    ref_t: np.ndarray,
    ref_pinch: np.ndarray,
) -> float | None:
    """RMS gap between the sim object displacement and the human pinch displacement over carry."""
    sel = (t >= carry[0]) & (t <= carry[1])
    if sel.sum() < 2:
        return None
    p0 = held_pos[int(np.searchsorted(t, t_grasp))]
    h0 = _interp(np.array([h_grasp]), ref_t, ref_pinch)[0]
    human = _interp(t[sel] - t_grasp + h_grasp, ref_t, ref_pinch) - h0
    gap = np.linalg.norm((held_pos[sel] - p0) - human, axis=1)
    return float(np.sqrt(np.mean(gap**2)))


def place_err_m(
    held_pos: np.ndarray,
    drop_point_m: np.ndarray | None,
    carry_offset: tuple[np.ndarray, np.ndarray] | None,
) -> float | None:
    """xy distance of the final object to the drop point, else to the human carry displacement.

    ``carry_offset`` = (sim object position at grasp, human pinch displacement grasp -> release).
    """
    final = held_pos[-1]
    if drop_point_m is not None:
        return float(np.linalg.norm(final[:2] - np.asarray(drop_point_m, float)[:2]))
    if carry_offset is None:
        return None
    p_grasp, human_disp = carry_offset
    return float(np.linalg.norm((final[:2] - p_grasp[:2]) - human_disp[:2]))


def held_track_m(t: np.ndarray, held_pos: np.ndarray, ref_t: np.ndarray, ref_pos: np.ndarray) -> float | None:
    """Median sim-vs-reference object distance after aligning the first reference sample."""
    ok = (ref_t >= t[0]) & (ref_t <= t[-1])
    if ok.sum() < MIN_TRACK_SAMPLES:
        return None
    sim = _interp(ref_t[ok], t, held_pos)
    ref = ref_pos[ok]
    return float(np.median(np.linalg.norm((sim - sim[0]) - (ref - ref[0]), axis=1)))


def tactile_residual(t: np.ndarray, sim_contact: np.ndarray, ref_t: np.ndarray, ref_on: np.ndarray) -> float:
    """1 - IoU over time of the sim pinch contact vs the reference tactile contact."""
    idx = np.clip(np.searchsorted(ref_t, t), 0, len(ref_t) - 1)
    near = np.clip(idx - 1, 0, None)
    idx = np.where(np.abs(ref_t[near] - t) < np.abs(ref_t[idx] - t), near, idx)
    ref = np.asarray(ref_on, bool)[idx]
    sim = np.asarray(sim_contact, bool)
    union = np.count_nonzero(sim | ref)
    return 0.0 if union == 0 else 1.0 - np.count_nonzero(sim & ref) / union


def lift_ratio(lift_sim_m: float, lift_human_m: float | None) -> float | None:
    if lift_human_m is None or not np.isfinite(lift_human_m) or lift_human_m <= 1e-6:
        return None
    return abs(1.0 - lift_sim_m / lift_human_m)


def timing_residual(sim_s: float | None, ref_s: float | None) -> float | None:
    return None if sim_s is None or ref_s is None else abs(sim_s - ref_s)


def match_score(residuals: dict[str, float | None], success: bool, failure_reasons: tuple[str, ...]) -> float:
    """Failures score 1 + worst severity; successes the weighted mean of clipped residual / scale."""
    if not success:
        sev = max((SEVERITY.get(r, DEFAULT_SEVERITY) for r in failure_reasons), default=DEFAULT_SEVERITY)
        return 1.0 + sev
    num = den = 0.0
    for k, w in WEIGHTS.items():
        r = _finite_or_none(residuals.get(k))
        if r is None:
            continue
        num += w * min(r / SCALES[k], 1.0)
        den += w
    return num / den if den else 0.0
