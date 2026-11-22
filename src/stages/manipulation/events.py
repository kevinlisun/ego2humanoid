"""Grasp-lift-translate-place event detection on a rigid object trajectory.

Pure numpy; used on achieved simulation states (QA) and on HOT3D GT tracks (golden clip
selection). Frames: ``T_world_object`` with z up, metres, seconds.
"""

from dataclasses import asdict, dataclass

import numpy as np

from common.transforms import rotation_angle


@dataclass(frozen=True)
class EventThresholds:
    lift_min_m: float = 0.05
    translate_min_m: float = 0.10
    translate_demo_frac: float = 0.75  # share of the demonstrated carry that counts when it is shorter
    place_height_tol_m: float = 0.02
    settle_window_s: float = 0.3
    settle_speed_m_s: float = 0.05
    settle_ang_speed_rad_s: float = 0.5
    rest_window_s: float = 0.1
    place_radius_m: float = 0.08  # how close to a receptacle's drop point counts as placed on it
    # Hand-object checks (need the per-frame robot contact flags and the planned release time).
    airborne_m: float = 0.02  # above its rest height by this much the object is off the support
    eject_speed_m_s: float = 0.4  # horizontal speed of an airborne object nobody holds: it was thrown
    carry_drop_s: float = 0.1  # longest hand-contact gap tolerated while carrying, before the release
    release_lead_s: float = 0.2  # contact may end this long before the planned release
    release_height_m: float = 0.03  # highest the object may be above its final rest when let go
    release_grace_s: float = 0.6  # the hand must be off the object this long after the planned release
    # A later touch (e.g. a fingertip brushing it on the retreat) is only "stuck" if it drags the object.
    drag_m: float = 0.01


@dataclass(frozen=True)
class PickPlaceEvents:
    lifted: bool
    translated: bool
    placed: bool
    settled: bool
    lift_peak_index: int
    lift_height_m: float
    translation_m: float
    final_height_above_rest_m: float
    final_speed_m_s: float
    place_error_m: float
    success: bool
    failure_reasons: tuple[str, ...]
    # Hand-object behaviour; NaN when no contact flags were given.
    ejection_speed_m_s: float = float("nan")  # peak horizontal speed while airborne and untouched
    carry_gap_s: float = float("nan")  # longest loss of hand contact between lift-off and release
    release_height_m: float = float("nan")  # object above its final rest when the hand let go
    contact_after_release_s: float = float("nan")  # end of the grip contact after the planned release
    post_release_drag_m: float = float("nan")  # object displacement while re-touched after letting go

    def as_dict(self) -> dict:
        return asdict(self)


def _window(t: np.ndarray, length_s: float, from_end: bool) -> slice:
    k = max(2, int(np.searchsorted(t, t[0] + length_s)) + 1)
    return slice(-k, None) if from_end else slice(0, k)


def detect_pick_place(
    timestamps_s: np.ndarray,
    T_world_object: np.ndarray,
    th: EventThresholds | None = None,
    place_target_pos_m: np.ndarray | None = None,
    expected_translation_m: float | None = None,
    hand_contact: np.ndarray | None = None,
    release_t_s: float | None = None,
) -> PickPlaceEvents:
    """Detect lift -> translate -> place -> settle on one object trajectory (N, 4, 4).

    ``place_target_pos_m`` is where the object is supposed to end up: the drop point of the
    receptacle the task names ("place it **on the blue bowl**"). Without it, "placed" can only mean
    "came back down to the height it started at", which is right for a table-to-table transfer and
    wrong for every task with a receptacle -- a correct placement ends one receptacle higher and was
    being scored as a failure.

    ``expected_translation_m`` is how far the demonstration carries the object. A demo that moves it
    less than ``translate_min_m`` (clip 108: 9.4 cm) cannot be reproduced to that bar, so the bar
    drops to ``translate_demo_frac`` of the demonstrated carry.

    ``hand_contact`` (N,) bool, whether the robot touches the object, and ``release_t_s``, when the
    plan lets go, add the hand-object checks: an object that leaves the hand before the release
    (DROPPED_IN_CARRY), is thrown (EJECTED), is let go in mid-air (AIRBORNE_RELEASE), or is still held
    well after the release or dragged by a later touch (STUCK_AFTER_RELEASE) fails, wherever it comes
    to rest.
    """
    th = th or EventThresholds()
    t = np.asarray(timestamps_s, dtype=np.float64)
    T = np.asarray(T_world_object, dtype=np.float64)
    if len(t) < 3 or T.shape != (len(t), 4, 4) or not np.isfinite(T).all():
        return PickPlaceEvents(
            False, False, False, False, 0, 0.0, 0.0, 0.0, 0.0, float("nan"), False, ("INVALID_TRAJECTORY",)
        )

    pos = T[:, :3, 3]
    rest = pos[_window(t, th.rest_window_s, False)].mean(0)
    height = pos[:, 2] - rest[2]
    peak = int(np.argmax(height))
    lift_height = float(height[peak])
    lifted = lift_height >= th.lift_min_m

    final = pos[_window(t, th.rest_window_s, True)].mean(0)
    translation = float(np.linalg.norm(final[:2] - rest[:2]))
    translate_min = th.translate_min_m
    if expected_translation_m is not None and np.isfinite(expected_translation_m):
        translate_min = min(translate_min, th.translate_demo_frac * float(expected_translation_m))
    translated = translation >= translate_min
    final_height = float(final[2] - rest[2])
    place_error = float("nan")
    if place_target_pos_m is None:
        # Placing means coming back down to rest height after being lifted and moved.
        on_target = abs(final_height) <= th.place_height_tol_m
    else:
        drop = np.asarray(place_target_pos_m, dtype=np.float64)
        place_error = float(np.linalg.norm(final - drop))
        on_target = bool(
            np.linalg.norm(final[:2] - drop[:2]) <= th.place_radius_m
            and abs(final[2] - drop[2]) <= th.place_radius_m
        )
    placed = lifted and translated and peak < len(t) - 1 and on_target

    w = _window(t, th.settle_window_s, True)
    dt = np.diff(t[w])
    speed = np.linalg.norm(np.diff(pos[w], axis=0), axis=1) / dt
    rel = np.swapaxes(T[w][:-1, :3, :3], 1, 2) @ T[w][1:, :3, :3]
    ang = rotation_angle(rel) / dt
    settled = bool(speed.max() <= th.settle_speed_m_s and ang.max() <= th.settle_ang_speed_rad_s)

    hand = _hand_checks(t, pos, rest[2], final[2], hand_contact, release_t_s, th)
    reasons = []
    if not lifted:
        reasons.append("NO_LIFT")
    if not translated:
        reasons.append("NO_TRANSLATION")
    if lifted and translated and not placed:
        reasons.append("NOT_PLACED")
    if not settled:
        reasons.append("NOT_SETTLED")
    reasons += hand["reasons"]
    return PickPlaceEvents(
        lifted,
        translated,
        placed,
        settled,
        peak,
        lift_height,
        translation,
        final_height,
        float(speed[-1]),
        place_error,
        placed and settled and not hand["reasons"],
        tuple(reasons),
        hand["ejection_speed_m_s"],
        hand["carry_gap_s"],
        hand["release_height_m"],
        hand["contact_after_release_s"],
        hand["post_release_drag_m"],
    )


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True in ``mask``; (0, 0) if none."""
    best = (0, 0)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if b - a > best[1]:
            best = (int(a), int(b - a))
    return best


def _hand_checks(
    t: np.ndarray,
    pos: np.ndarray,
    rest_z: float,
    final_z: float,
    contact: np.ndarray | None,
    release_t_s: float | None,
    th: EventThresholds,
) -> dict:
    nan = float("nan")
    out = {
        "reasons": [],
        "ejection_speed_m_s": nan,
        "carry_gap_s": nan,
        "release_height_m": nan,
        "contact_after_release_s": nan,
        "post_release_drag_m": nan,
    }
    if contact is None:
        return out
    c = np.asarray(contact, bool).reshape(-1)
    if len(c) != len(t):
        raise ValueError("hand_contact must have one flag per frame")
    dt = float(np.median(np.diff(t)))
    airborne = pos[:, 2] - rest_z > th.airborne_m
    v_xy = np.r_[np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / np.diff(t), 0.0]
    free = airborne & ~c
    # Two frames: one fast sample is a contact-resolution jitter, a throw lasts until landing.
    fast = free & (v_xy > th.eject_speed_m_s)
    out["ejection_speed_m_s"] = float(v_xy[free].max()) if free.any() else 0.0
    if _longest_run(fast)[1] >= 2:
        out["reasons"].append("EJECTED")
    if release_t_s is None or not airborne.any():
        return out
    lift_k = int(np.argmax(airborne))
    carry = (np.arange(len(t)) >= lift_k) & (t < release_t_s - th.release_lead_s)
    gap = _longest_run(carry & ~c)[1] * dt
    out["carry_gap_s"] = gap
    if gap > th.carry_drop_s:
        out["reasons"].append("DROPPED_IN_CARRY")
    # The grip contact is the run of contact frames holding the object at the planned release; it
    # sets when the hand actually let go. Contacts after it are re-touches.
    k_rel = int(np.searchsorted(t, release_t_s, side="right"))
    held = np.flatnonzero(c[:k_rel])
    if len(held):
        end = int(held[-1])
        while end + 1 < len(c) and c[end + 1]:
            end += 1
        out["release_height_m"] = float(pos[end, 2] - final_z)
        out["contact_after_release_s"] = float(max(t[end] - release_t_s, 0.0))
        out["post_release_drag_m"] = 0.0
        if out["release_height_m"] > th.release_height_m:
            out["reasons"].append("AIRBORNE_RELEASE")
        stuck = t[end] - release_t_s > th.release_grace_s
        later = np.flatnonzero(c[end + 1 :]) + end + 1
        if len(later):
            drag = np.linalg.norm(pos[later[0] :] - pos[later[0] - 1], axis=1)
            out["post_release_drag_m"] = float(drag[: later[-1] - later[0] + 1].max())
            stuck |= out["post_release_drag_m"] > th.drag_m
        if stuck:
            out["reasons"].append("STUCK_AFTER_RELEASE")
    return out
