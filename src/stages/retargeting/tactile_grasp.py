"""Grasp timing from the human hand: EgoTac contact span and the lift onset of the pinch.

The object detector reports the first frame in which the object is seen *above* its rest height, but
monocular depth makes that unreliable: on clip 1 the phone is touched at 1.2 s, slid along the table
towards the body for two seconds and only picked up at 3.0 s, while the detector already fired at
1.4 s. Closing the robot fingers there grips nothing and then follows the slide down into the table.

The grasp a robot can reproduce is the *pick*: the instant the pinch starts its main rise. It is the
start of the largest upward excursion of the human pinch height, searched only while the human hand
is in tactile contact (EgoTac) so the rise of the empty hand before or after the grasp is ignored.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

# MANO vertex ids at the centre of each tactile pad (EgoTac contact maps are per MANO vertex).
PAD_VERTICES = {
    "thumb": (762, 740, 757, 741),
    "index": (347, 328, 281, 237),
    "middle": (462, 436, 454, 441),
}
CONTACT_PROB = 0.25  # a pad is in contact above this mean contact probability
SMOOTH_FRAMES = 5


def pinch_contact(contact_prob: np.ndarray, side_index: int) -> np.ndarray:
    """(N,) bool: the thumb and the index or middle pad of one hand are in contact (a pinch).

    ``contact_prob`` is EgoTac's (N, 2, 778) per-vertex contact probability; hand axis (left, right).
    """
    p = np.asarray(contact_prob, float)[:, side_index]
    pad = {k: p[:, list(v)].mean(axis=1) for k, v in PAD_VERTICES.items()}
    on = (pad["thumb"] > CONTACT_PROB) & ((pad["index"] > CONTACT_PROB) | (pad["middle"] > CONTACT_PROB))
    return median_filter(on.astype(np.uint8), size=SMOOTH_FRAMES, mode="nearest").astype(bool)


def contact_spans(frame_idx: np.ndarray, on: np.ndarray, n_frames: int) -> list[tuple[int, int]]:
    """All runs of contact, as inclusive (first, last) ranges of the ``n_frames`` clip frames.

    ``frame_idx`` maps each tactile sample to its clip frame; samples outside the clip are dropped.
    """
    fi = np.asarray(frame_idx, int)
    keep = (fi >= 0) & (fi < n_frames)
    fi, on = fi[keep], np.asarray(on, bool)[keep]
    spans, start = [], None
    for i in range(len(on) + 1):
        if i < len(on) and on[i]:
            start = i if start is None else start
        elif start is not None:
            spans.append((int(fi[start]), int(fi[i - 1])))
            start = None
    return spans


def contact_span(frame_idx: np.ndarray, on: np.ndarray, n_frames: int) -> tuple[int, int] | None:
    """Longest run of contact (see ``contact_spans``), or None without contact."""
    spans = contact_spans(frame_idx, on, n_frames)
    return max(spans, key=lambda s: s[1] - s[0]) if spans else None


def pick_span(
    spans: list[tuple[int, int]], pinch_z: np.ndarray, lo: int, hi: int
) -> tuple[tuple[int, int], int, float] | None:
    """(span clipped to ``[lo, hi]``, onset, rise_m) of the contact run in which the pinch rises most.

    A pick is contact *followed by a lift*. The longest contact is not always it: on clip 117 the hand
    holds the block on the scooper for 1.6 s after placing it, longer than the 0.7 s pick, but the
    pinch only rises (9 cm) in the pick.
    """
    best = None
    for a, b in spans:
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        onset, rise = grasp_onset(pinch_z, a, b)
        if best is None or rise > best[2]:
            best = ((a, b), onset, rise)
    return best


def grasp_onset(pinch_z: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """(frame, rise_m): start of the largest upward excursion of ``pinch_z`` within ``[lo, hi]``."""
    seg = np.asarray(pinch_z, float)[lo : hi + 1]
    low = np.minimum.accumulate(seg)
    rise = seg - low
    peak = int(np.argmax(rise))
    onset = int(np.argmin(seg[: peak + 1]))
    return lo + onset, float(rise[peak])


def tactile_release_index(
    span: tuple[int, int] | None, hold: np.ndarray, timestamps_s: np.ndarray, min_early_s: float = 0.5
) -> int:
    """Last hold frame: the end of the tactile span when it ends > ``min_early_s`` before the pinch release.

    The pinch-geometry release can lag the real one by seconds (clips 100/101: the fingers stay
    close together after letting go); the tactile contact ends when the object is let go.
    """
    release = int(hold[-1])
    if span is None or span[1] <= hold[0] or span[1] >= release:
        return release
    return int(span[1]) if timestamps_s[release] - timestamps_s[span[1]] > min_early_s else release
