"""Grasp detection from hand landmarks and a tagged proxy box for the grasped object.

This is NOT object perception: the box is inferred from the hand itself, so any episode built on it
must carry ``object_source="grasp_proxy"`` and cannot be accepted into the dataset.
"""

from dataclasses import dataclass

import numpy as np

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass
class GraspProxy:
    grasp_idx: int
    release_idx: int
    center_m: np.ndarray  # (3,) sim frame, resting on the table
    half_size_m: np.ndarray  # (3,)


def pinch_aperture(lm_sim: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lm_sim[:, THUMB_TIP] - lm_sim[:, INDEX_TIP], axis=1)


def detect_grasp_proxy(
    lm_sim: np.ndarray,
    timestamps_s: np.ndarray,
    table_top_z_m: float = 0.80,
    max_height_m: float = 0.10,
    min_hold_s: float = 0.3,
) -> GraspProxy | None:
    """Find the longest span where the pinch is closed near the table, then size a box around it."""
    ap = pinch_aperture(lm_sim)
    tip_mid = 0.5 * (lm_sim[:, THUMB_TIP] + lm_sim[:, INDEX_TIP])
    height = tip_mid[:, 2] - table_top_z_m
    lo, hi = np.percentile(ap, [10, 90])
    closed = (ap < lo + 0.4 * (hi - lo)) & (height < max_height_m)
    best, start = (0, 0), None
    for i, c in enumerate(np.append(closed, False)):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    g, r = best
    if r - g < 2 or timestamps_s[r - 1] - timestamps_s[g] < min_hold_s:
        return None
    width = float(np.clip(np.median(ap[g:r]) + 0.01, 0.03, 0.08))
    half = np.array([width / 2, width / 2, width / 2])
    center = tip_mid[g : g + 3].mean(axis=0)
    center[2] = table_top_z_m + half[2]
    return GraspProxy(g, r - 1, center, half)
