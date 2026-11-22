"""Fuse per-frame object candidates (MoGe camera frame) with the hand estimate (FAH camera frame)."""

from dataclasses import dataclass

import numpy as np


def hand_depth_scale(
    depth_small: np.ndarray, wrist_px: np.ndarray, wrist_z_m: np.ndarray, image_wh: tuple[int, int]
) -> float:
    """Median ratio hand-estimator depth / MoGe depth at the wrist pixel (both RGB-only, different scales)."""
    h, w = depth_small.shape[1:]
    ratios = []
    for d, p, z in zip(depth_small, wrist_px, wrist_z_m, strict=True):
        if not (np.isfinite(p).all() and np.isfinite(z) and z > 0):
            continue
        u = int(np.clip(p[0] / image_wh[0] * w, 0, w - 1))
        v = int(np.clip(p[1] / image_wh[1] * h, 0, h - 1))
        m = float(d[v, u])
        if np.isfinite(m) and m > 0:
            ratios.append(z / m)
    if not ratios:
        raise ValueError("no usable wrist depth samples")
    return float(np.median(ratios))


@dataclass
class HeldObject:
    start_xy_m: np.ndarray
    half_size_m: np.ndarray
    n_matches: int
    observed_lift_m: float
    first_idx: int  # detection indices bounding the carried (lifted) span
    last_idx: int
    pinch_z_pre_m: float  # hand pinch height just before the lift: where the object rested
    views: tuple[tuple[int, int], ...]  # (detection index, SAM slot) of matches, for mesh generation
    size_m: float  # longest observed extent, scales a generated mesh


def select_held_object(
    cand_sim: np.ndarray,  # (N, K, 3) candidate centroids in sim frame
    extent_m: np.ndarray,  # (N, K, 3)
    pinch_mid_sim: np.ndarray,  # (N, 3) human thumb-index midpoint at the same frames
    usable: np.ndarray,  # (N,) bool, detection frame lies in the reliable hand span
    table_top_z_m: float,
    max_dist_m: float = 0.12,
    lift_m: float = 0.03,
    max_extent_m: float = 0.4,
) -> HeldObject | None:
    """Follow the candidate nearest the pinch point; carried span = frames lifted above its resting height.

    The rest position is the last matches before the first lifted frame, so the object starts where the
    hand picks it up. Distractors far from the pinch are ignored by ``max_dist_m``; boxes larger than
    ``max_extent_m`` (the tray or the table top segmented as a whole, clip 117) cannot be in one hand.
    """
    picks = []
    for i in np.where(usable)[0]:
        d = np.linalg.norm(cand_sim[i] - pinch_mid_sim[i], axis=-1)
        d[~np.isfinite(d) | (np.nan_to_num(extent_m[i], nan=0.0).max(axis=-1) > max_extent_m)] = np.inf
        k = int(np.argmin(d))
        if d[k] < max_dist_m:
            picks.append((i, k))
    if len(picks) < 3:
        return None
    pos = np.array([cand_sim[i, k] for i, k in picks])
    rest_z = np.percentile(pos[:, 2], 10)
    lifted = np.where(pos[:, 2] > rest_z + lift_m)[0]
    if len(lifted) == 0:
        return None
    first = int(lifted[0])
    lo_i = max(0, first - 3)
    pre_pinch = np.array([pinch_mid_sim[i] for i, _ in picks[lo_i : max(first, 1)]])
    pre_pinch_z = float(np.median(pre_pinch[:, 2]))
    half = np.clip(np.median([extent_m[i, k] for i, k in picks], axis=0) / 2, 0.02, 0.06)
    return HeldObject(
        np.median(pre_pinch[:, :2], axis=0),  # the hand is at the object when it pinches; depth is noisier
        half,
        len(picks),
        float(pos[:, 2].max() - rest_z),
        picks[first][0],
        picks[int(lifted[-1])][0],
        pre_pinch_z,
        tuple((int(i), int(k)) for i, k in picks[: first + 1]),
        float(np.clip(np.median([extent_m[i, k].max() for i, k in picks]), 0.04, 0.25)),
    )
