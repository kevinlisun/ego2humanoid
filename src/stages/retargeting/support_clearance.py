"""Keep both retargeted hands above the support surface.

The table top is set from the height at which the hand grips the object (see ``table_fit.py``), but
the monocular hand trajectory is not metrically consistent with that height: on several clips the
resting hands sit 7-13 cm *below* the grip height. Replayed as is, both robot hands start inside the
table and stay jammed under it (measured on clip 0: 200 kN table contact on the idle hand, the active
wrist never moved towards the object). A real hand rests *on* the table, so every hand point over the
table footprint is lifted to the surface plus a clearance, smoothly in time; frames where the active
hand is meant to be at the object are exempt. The robot hand is bulkier than these landmarks, so the
executed geometry is checked separately (``table_fit.table_penetration``).

The other objects on the table are supports in the same sense: a hand resting next to a bowl in the
video often lands inside it after the monocular depth (clip 104: the idle right wrist 2.7 cm inside
the welded bowl's wall for the whole clip, which jammed the contact solver and lost the grasp of the
other hand). ``lift_above_support`` therefore takes several supports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

CLEARANCE_M = 0.015  # robot finger pads are thicker than the human landmarks suggest
SMOOTH_FRAMES = 9  # the lift is widened by this many frames before being smoothed


@dataclass(frozen=True)
class Support:
    top_z_m: float
    lo_xy_m: tuple[float, float]
    hi_xy_m: tuple[float, float]

    @classmethod
    def from_table(cls, top_z_m: float, center_xy_m, half_size_xy_m) -> Support:
        c, h = np.asarray(center_xy_m, float), np.asarray(half_size_xy_m, float)
        return cls(float(top_z_m), tuple(c - h), tuple(c + h))

    @classmethod
    def from_points(cls, points_m, margin_m: float = 0.0) -> Support:
        """The top of an object (any (V, 3) world points), over its xy box grown by ``margin_m``."""
        p = np.asarray(points_m, float).reshape(-1, 3)
        lo, hi = p[:, :2].min(0) - margin_m, p[:, :2].max(0) + margin_m
        return cls(float(p[:, 2].max()), tuple(lo), tuple(hi))


def support_lift(landmarks_sim: np.ndarray, support: Support, clearance_m: float = CLEARANCE_M) -> np.ndarray:
    """(N, 21, 3) hand landmarks -> (N,) upward shift that keeps every point over the table above it."""
    lm = np.asarray(landmarks_sim, float)
    lo, hi = np.asarray(support.lo_xy_m), np.asarray(support.hi_xy_m)
    over = np.all((lm[..., :2] >= lo) & (lm[..., :2] <= hi), axis=-1)
    z = np.where(over, lm[..., 2], np.inf).min(axis=1)
    need = np.clip(support.top_z_m + clearance_m - z, 0.0, None)
    need[~np.isfinite(need)] = 0.0
    if not need.any():
        return need
    wide = maximum_filter1d(need, size=SMOOTH_FRAMES, mode="nearest")
    return np.maximum(gaussian_filter1d(wide, SMOOTH_FRAMES / 3, mode="nearest"), need)


def exempt_weight(exempt: np.ndarray | None, n: int) -> np.ndarray:
    """(N,) bool -> (N,) weight in [0, 1]: 1 inside the exempt frames, ramping smoothly to 0 outside."""
    if exempt is None:
        return np.zeros(n)
    e = np.asarray(exempt, float)
    return np.maximum(np.clip(gaussian_filter1d(e, SMOOTH_FRAMES / 3, mode="nearest"), 0.0, 1.0), e)


def lift_above_support(
    landmarks_sim: np.ndarray,
    support: Support | Sequence[Support] | None,
    exempt: np.ndarray | None = None,
) -> np.ndarray:
    """``exempt`` (N,) bool marks frames where the hand is meant to be at the object (approach and
    grasp): the fingers have to reach down to the grip there, lifting them makes the pinch close
    above the object (measured on clip 104: lift 0.20 m -> 0.02 m)."""
    supports = (support,) if isinstance(support, Support) else tuple(support or ())
    if not supports:
        return landmarks_sim
    out = np.array(landmarks_sim, float, copy=True)
    need = np.max([support_lift(out, s) for s in supports], axis=0)
    lift = need * (1.0 - exempt_weight(exempt, len(out)))
    out[..., 2] += lift[:, None]
    return out


def held_object_lift(
    center_m: np.ndarray,
    half_size_m: np.ndarray,
    supports: Sequence[Support],
    active: np.ndarray,
    clearance_m: float,
) -> np.ndarray:
    """(N,) upward shift that carries an ``half_size_m`` box at ``center_m`` (N, 3) over ``supports``.

    The hands are kept above the other objects only away from the grasp (``lift_above_support``),
    so nothing stopped what they carry from sweeping through them: clip 117's block was dragged
    across the scooper's rim 1 cm under its top and knocked out of the pinch. Only ``active`` frames
    (the fingers closed on it) are lifted; a support already under the box at the first of them is
    what it is picked from or stands beside, and is skipped.
    """
    c = np.asarray(center_m, float)
    h = np.asarray(half_size_m, float)
    act = np.asarray(active, bool)
    need = np.zeros(len(c))
    if not act.any() or not supports:
        return need
    k0 = int(np.argmax(act))
    for s in supports:
        lo, hi = np.asarray(s.lo_xy_m), np.asarray(s.hi_xy_m)
        over = np.all((c[:, :2] + h[:2] > lo) & (c[:, :2] - h[:2] < hi), axis=1)
        if over[k0]:
            continue
        need = np.maximum(need, np.where(act & over, s.top_z_m + clearance_m - (c[:, 2] - h[2]), 0.0))
    need = np.clip(need, 0.0, None)
    if not need.any():
        return need
    wide = maximum_filter1d(need, size=SMOOTH_FRAMES, mode="nearest")
    out = np.maximum(gaussian_filter1d(wide, SMOOTH_FRAMES / 3, mode="nearest"), need)
    # Smoothing must not reach back into the approach: nothing is carried before the fingers close,
    # and lifting the open hand there misses the grasp.
    out[:k0] = 0.0
    return out
