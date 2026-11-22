"""MANO-vertex tactile -> per-region force/contact, smoothing, thresholds and fusion with geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from contracts.arrays import MANO_VERTEX_COUNT, validate_arrays

REGION_NAMES = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "thumb_pad",
    "index_pad",
    "middle_pad",
    "ring_pad",
    "pinky_pad",
    "palm",
)
N_REGIONS = len(REGION_NAMES)
# Standard MANO fingertip vertex ids (manopth / smplx tip list), thumb..pinky.
FINGERTIP_VERTEX_IDS = (744, 320, 443, 554, 671)
# MANO joint order: wrist, index(1-3), middle(4-6), pinky(7-9), ring(10-12), thumb(13-15).
# Finger index: 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky.
_FINGER_OF_JOINT = {
    **dict.fromkeys((1, 2, 3), 1),
    **dict.fromkeys((4, 5, 6), 2),
    **dict.fromkeys((7, 8, 9), 4),
    **dict.fromkeys((10, 11, 12), 3),
    **dict.fromkeys((13, 14, 15), 0),
}


def region_of_vertex_from_skinning(lbs_weights: np.ndarray) -> np.ndarray:
    """(778,) region index from MANO skinning weights ``(778, 16)``.

    A vertex belongs to its dominant joint's finger pad, or to the palm when the wrist dominates;
    the five standard fingertip vertices override to the tip regions.
    """
    w = np.asarray(lbs_weights, float)
    if w.shape != (MANO_VERTEX_COUNT, 16):
        raise ValueError(f"lbs_weights must be (778, 16), got {w.shape}")
    dominant = w.argmax(axis=1)
    region = np.full(MANO_VERTEX_COUNT, REGION_NAMES.index("palm"), dtype=np.int64)
    for joint, finger in _FINGER_OF_JOINT.items():
        region[dominant == joint] = 5 + finger
    for finger, vid in enumerate(FINGERTIP_VERTEX_IDS):
        region[vid] = finger
    return region


def _check_region_map(region_of_vertex: np.ndarray) -> np.ndarray:
    r = np.asarray(region_of_vertex)
    if r.shape != (MANO_VERTEX_COUNT,) or r.dtype.kind not in "iu":
        raise ValueError(f"region_of_vertex must be an int array of shape (778,), got {r.shape} {r.dtype}")
    if r.min() < -1 or r.max() >= N_REGIONS:
        raise ValueError(f"region_of_vertex values must be in [-1, {N_REGIONS - 1}]")
    return r.astype(np.int64)


def aggregate_regions(
    force: np.ndarray,
    contact_prob: np.ndarray,
    region_of_vertex: np.ndarray,
    prob_mode: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-region force sum and contact probability (``max`` or ``mean``); both ``(N,2,R)``.

    Vertices with region ``-1`` are ignored; empty regions give 0.
    """
    if prob_mode not in ("max", "mean"):
        raise ValueError("prob_mode must be 'max' or 'mean'")
    validate_arrays("tactile", {"force": force, "contact_prob": contact_prob})
    region = _check_region_map(region_of_vertex)
    f, p = np.asarray(force, float), np.asarray(contact_prob, float)
    out_f = np.zeros((*f.shape[:2], N_REGIONS))
    out_p = np.zeros_like(out_f)
    for r in range(N_REGIONS):
        sel = region == r
        if not sel.any():
            continue
        out_f[..., r] = f[..., sel].sum(axis=-1)
        out_p[..., r] = p[..., sel].max(axis=-1) if prob_mode == "max" else p[..., sel].mean(axis=-1)
    return out_f, out_p


def smooth_time(x: np.ndarray, method: str = "median", window: int = 5, alpha: float = 0.5) -> np.ndarray:
    """Temporal smoothing along axis 0: ``median`` (odd window) or ``ema`` (causal, ``alpha``)."""
    x = np.asarray(x, float)
    if method == "median":
        if window < 1 or window % 2 == 0:
            raise ValueError("median window must be a positive odd integer")
        return median_filter(x, size=(window,) + (1,) * (x.ndim - 1), mode="nearest")
    if method == "ema":
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        out = x.copy()
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out
    raise ValueError("method must be 'median' or 'ema'")


def tactile_contact_flags(contact_prob_region: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """(N,2,R) bool: region contact probability at or above ``threshold``."""
    return np.asarray(contact_prob_region, float) >= threshold


def fuse_with_geometric(
    tactile_flags: np.ndarray,
    geometric_flags: np.ndarray,
    mode: str = "or",
) -> np.ndarray:
    """Fuse tactile region flags with geometric per-hand contact flags from ``align/contact.py``.

    ``geometric_flags`` is ``(N,2)`` bool (hand-level, e.g. the pinch flag per hand). The result is
    ``(N,2)`` bool: ``tactile_any`` uses tactile alone, ``and`` requires both, ``or`` accepts either.
    """
    t = np.asarray(tactile_flags, bool).any(axis=-1)
    g = np.asarray(geometric_flags, bool)
    if t.shape != g.shape:
        raise ValueError(f"tactile {t.shape} and geometric {g.shape} flags must share (N,2)")
    if mode == "and":
        return t & g
    if mode == "or":
        return t | g
    if mode == "tactile_any":
        return t
    raise ValueError("mode must be 'and', 'or' or 'tactile_any'")


def compute_tactile_regions(
    force: np.ndarray,
    contact_prob: np.ndarray,
    region_of_vertex: np.ndarray,
    *,
    prob_mode: str = "max",
    smoothing: str = "median",
    window: int = 5,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Aggregate, smooth and threshold; returns the ``tactile_regions`` bundle arrays."""
    f_reg, p_reg = aggregate_regions(force, contact_prob, region_of_vertex, prob_mode)
    f_reg = smooth_time(f_reg, smoothing, window)
    p_reg = np.clip(smooth_time(p_reg, smoothing, window), 0.0, 1.0)
    out = {
        "force_region": f_reg,
        "contact_prob_region": p_reg,
        "tactile_contact_flags": tactile_contact_flags(p_reg, threshold),
    }
    validate_arrays("tactile_regions", out)
    return out


def write_tactile_regions(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write ``tactile_regions.npz`` (validated) with the region names stored alongside."""
    validate_arrays("tactile_regions", arrays)
    path = Path(path)
    np.savez_compressed(path, **arrays, region_names=np.array(REGION_NAMES))
    return path
