"""Draw the estimated hand pose (both hands) and, when available, the estimated tactile state on a frame.

The left panel of the review video shows what the perception stack believes: 2D skeletons with their
per-hand confidence, and per-region contact force painted on the landmarks that belong to that region.
Tactile input follows the ``tactile_regions`` contract (``stages/hands/tactile.py``), so the same overlay
works for EgoTac today and for any later tactile head.
"""

from __future__ import annotations

import cv2
import numpy as np

from stages.reconstruction.hands.tactile import N_REGIONS, REGION_NAMES

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip
SIDE_NAMES = ("left", "right")
# BGR skeleton colours; the active hand is the one the retarget follows.
ACTIVE_COLOR = (80, 255, 80)
PASSIVE_COLOR = (0, 180, 255)
LOW_SCORE_COLOR = (120, 120, 120)


def landmark_region_ids() -> np.ndarray:
    """(21,) ``REGION_NAMES`` index per hand landmark: fingertips -> ``*_tip``, other finger joints ->
    ``*_pad``, wrist -> ``palm``."""
    ids = np.full(21, REGION_NAMES.index("palm"), dtype=np.int64)
    for finger in range(5):  # landmarks 1..4 thumb, 5..8 index, ... 17..20 pinky
        base = 1 + 4 * finger
        ids[base : base + 3] = 5 + finger  # pad
        ids[base + 3] = finger  # tip
    return ids


LANDMARK_REGION = landmark_region_ids()


def force_color(value: float) -> tuple[int, int, int]:
    """Normalised force/contact in [0, 1] -> BGR, blue (no contact) through yellow to red (max)."""
    v = float(np.clip(value, 0.0, 1.0))
    return _JET_BGR[int(np.uint8([[v * 255]])[0, 0])]


# One colour-map call per value took ~0.3 ms; the overlay asks for ~1000 colours a frame.
_JET_BGR = tuple(
    tuple(int(c) for c in px)
    for px in cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_JET)[0]
)


def draw_hand(
    frame: np.ndarray,
    joints2d: np.ndarray,
    color: tuple[int, int, int],
    region_force: np.ndarray | None = None,
    thickness: int = 3,
) -> None:
    """Draw one hand's skeleton in place.

    ``joints2d``: (21, 2) px; ``region_force``: (N_REGIONS,) normalised to [0, 1].
    """
    pts = np.asarray(joints2d, float)
    good = np.isfinite(pts).all(axis=1)
    for a, b in HAND_EDGES:
        if good[a] and good[b]:
            cv2.line(frame, _px(pts[a]), _px(pts[b]), color, thickness, cv2.LINE_AA)
    for i in range(21):
        if not good[i]:
            continue
        if region_force is None:
            cv2.circle(frame, _px(pts[i]), thickness, color, -1, cv2.LINE_AA)
        else:
            f = float(region_force[LANDMARK_REGION[i]])
            r = int(round(thickness + 5 * np.clip(f, 0.0, 1.0)))
            cv2.circle(frame, _px(pts[i]), r, force_color(f), -1, cv2.LINE_AA)
            cv2.circle(frame, _px(pts[i]), r, (30, 30, 30), 1, cv2.LINE_AA)


def draw_frame_overlay(
    frame: np.ndarray,
    joints2d: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    active_side: str,
    min_score: float = 0.7,
    region_force: np.ndarray | None = None,
    force_scale_n: float | None = None,
    tactile_label: str = "tactile",
) -> np.ndarray:
    """Draw both hands on a BGR frame (copied, not modified in place).

    ``joints2d`` (2, 21, 2) px, ``valid`` (2,), ``score`` (2,) for [left, right].
    ``region_force`` (2, N_REGIONS) in Newtons; ``force_scale_n`` is the force mapped to the top of the
    colour map, either one value or one per hand (defaults to the frame maximum of each hand, which keeps
    the overlay readable before calibration). Hands are scaled independently because a hand leaning on the
    table reaches forces an order of magnitude above a fingertip grasp.
    """
    out = frame.copy()
    ui = max(out.shape[0] / 480.0, 1.0)  # the panel is downscaled later, so size the overlay with the frame
    scale = None
    if region_force is not None:
        scale = np.broadcast_to(np.asarray(force_scale_n, float), (len(SIDE_NAMES),)).astype(float).copy()
        bad = ~np.isfinite(scale) | (scale <= 0)
        scale[bad] = np.maximum(np.max(np.asarray(region_force, float), axis=1), 1e-6)[bad]
    for i, side in enumerate(SIDE_NAMES):
        if not bool(valid[i]):
            continue
        reliable = float(score[i]) >= min_score
        base = ACTIVE_COLOR if side == active_side else PASSIVE_COLOR
        rf = None if region_force is None else np.asarray(region_force[i], float) / scale[i]
        width = int(round((3 if reliable else 2) * ui))
        draw_hand(out, joints2d[i], base if reliable else LOW_SCORE_COLOR, rf, width)

    _legend(
        out, valid, score, active_side, min_score, None if region_force is None else scale, ui, tactile_label
    )
    return out


def _legend(
    frame: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    active_side: str,
    min_score: float,
    force_scale_n: float | None,
    ui: float = 1.0,
    tactile_label: str = "tactile",
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th, x0 = 0.7 * ui, max(int(round(2 * ui)), 2), int(round(20 * ui))
    y, step = int(round(80 * ui)), int(round(30 * ui))
    for i, side in enumerate(SIDE_NAMES):
        role = "active" if side == active_side else "passive"
        state = "drop" if not bool(valid[i]) or float(score[i]) < min_score else "ok"
        color = ACTIVE_COLOR if side == active_side else PASSIVE_COLOR
        cv2.putText(
            frame, f"{side} ({role}) score {float(score[i]):.2f} [{state}]", (x0, y), font, fs, color, th
        )
        y += step
    if force_scale_n is None:
        cv2.putText(frame, f"{tactile_label}: off", (x0, y), font, fs, (200, 200, 200), th)
        return
    top = np.atleast_1d(np.asarray(force_scale_n, float))
    span = " / ".join(f"{v:.2f}" for v in top) + (" N (L/R)" if top.size > 1 else " N")
    cv2.putText(frame, f"{tactile_label} 0 - {span}", (x0, y), font, fs, (255, 255, 255), th)
    bar_y, w, bar_h = y + int(round(14 * ui)), int(round(180 * ui)), int(round(12 * ui))
    for k in range(w):
        cv2.line(frame, (x0 + k, bar_y), (x0 + k, bar_y + bar_h), force_color(k / (w - 1)), 1)


def _px(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def region_force_series(path, n_frames: int) -> np.ndarray | None:
    """Load a tactile npz -> (n_frames, 2, N_REGIONS) force, or ``None`` when absent.

    Accepts the human bundle (``tactile_regions.npz``, key ``force_region``) and the simulated one
    (``sim_tactile.npz``, key ``normal_force_n``); both use the same region schema. Frames are sampled
    by nearest index, so a tactile stream at a different rate still lines up.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    arr = np.load(p)
    key = "force_region" if "force_region" in arr.files else "normal_force_n"
    f = np.asarray(arr[key], float)
    if f.ndim != 3 or f.shape[1:] != (2, N_REGIONS):
        raise ValueError(f"force_region must be (N, 2, {N_REGIONS}), got {f.shape}")
    idx = np.clip(np.round(np.linspace(0, len(f) - 1, n_frames)).astype(int), 0, len(f) - 1)
    return f[idx]
