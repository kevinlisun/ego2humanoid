"""Grip-window refinement from tactile contact and the object's rest point (``retarget_estimated.py``)."""

import dataclasses
import os

import numpy as np

from stages.retargeting.tactile_grasp import (
    contact_spans,
    grasp_onset,
    pick_span,
    pinch_contact,
    tactile_release_index,
)

TACTILE_RELEASE = os.environ.get("E2H_TACTILE_RELEASE", "0") == "1"  # release at the tactile contact end


def refine_grip_window(
    clip, t, first, side_i, human_pinch, hold_k, held, close_lead, min_pick_rise_m, reach_tol_m
):
    """Return ``(tactile_timing, hold_k, held)`` with the grip moved to where the pinch is on the object."""
    # Monocular depth fires the detector early when the object is first pushed or slid (clip 1:
    # 2 s too early, the fingers then follow the slide down into the table). The robot reproduces
    # the *pick*: the start of the pinch's main rise while the human pinch is in tactile contact.
    tactile_timing = None
    tac_file = clip / "tactile_egotac.npz"
    if tac_file.exists():
        tac = np.load(tac_file)
        on = pinch_contact(tac["contact_prob"], side_i)
        spans = contact_spans(tac["frame_idx"] - first, on, len(t))
        lo_k, hi_k = int(hold_k[0]), int(hold_k[-1])
        span, picked = None, None
        if spans:
            picked = pick_span(spans, human_pinch[:, 2], lo_k, hi_k)
            span = None if picked is None else picked[0]
        elif hi_k > lo_k:
            picked = ((lo_k, hi_k), *grasp_onset(human_pinch[:, 2], lo_k, hi_k))
        if picked is not None:
            _, onset, rise = picked
            tactile_timing = {
                "tactile_span_s": None if span is None else [float(t[i] - t[0]) for i in span],
                "grasp_onset_s": float(t[onset] - t[0]),
                "grasp_rise_m": rise,
            }
            if rise >= min_pick_rise_m and onset - close_lead > hold_k[0]:
                hold_k = np.arange(onset - close_lead, hold_k[-1] + 1)
                near = slice(max(onset - 2, 0), onset + 1)
                held = dataclasses.replace(
                    held,
                    start_xy_m=np.median(human_pinch[near, :2], axis=0),
                    pinch_z_pre_m=float(np.median(human_pinch[near, 2])),
                )
                tactile_timing["hold_moved"] = True
        if TACTILE_RELEASE and tactile_timing is not None:
            end_k = tactile_release_index(span, hold_k, t)
            if end_k < hold_k[-1]:
                tactile_timing["release_moved_s"] = float(t[hold_k[-1]] - t[end_k])
                hold_k = hold_k[hold_k <= end_k]
        print("tactile grasp timing:", tactile_timing)
    # Tactile onset at the perceived hold start already confirms it (clip 104: both 2.3 s; moving
    # the grip off it left the block rocking in the bowl, NOT_SETTLED).
    onset_s = (tactile_timing or {}).get("grasp_onset_s")
    tactile_agrees = onset_s is not None and abs(onset_s - (t[hold_k[0]] - t[0])) <= 0.1
    if not (tactile_timing or {}).get("hold_moved") and not tactile_agrees:
        # The object detections are sparse, so the first one seen lifted can come well after the
        # pick (clip 114: ~1 s, the hand already 17 cm away carrying it); the robot then closes on
        # air. The grip is where the human pinch last was at the object's rest point.
        rest = np.array([*held.start_xy_m, held.pinch_z_pre_m])
        grip = int(hold_k[0]) + close_lead
        dist = np.linalg.norm(human_pinch - rest, axis=1)
        if grip < len(t) and dist[grip] > reach_tol_m:
            at_obj = np.flatnonzero(dist[:grip] <= reach_tol_m)
            if len(at_obj):
                hold_k = np.arange(max(int(at_obj[-1]) - close_lead, 0), hold_k[-1] + 1)
                print(f"grip moved {t[at_obj[-1]] - t[grip]:+.2f} s to the last frame at the object")
    return tactile_timing, hold_k, held
