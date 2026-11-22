"""Is the place target inside the arm's workspace at the chosen stance? (Track RSI, BF.9)

The standing spot is searched on the human path, which ends at the human's release point; a receptacle
that lies further away leaves the place window out of reach (clip 117). The planner records the share
of the IK targets that stay more than ``UNREACHABLE_POS_M`` away in the grasp and place windows; this
module turns those shares into events the attribution maps onto the stance knobs.
"""

import numpy as np

from contracts.real2sim import FeedbackEvent

UNREACHABLE_POS_M = 0.03  # same bar as stages.locomotion.base_placement (not imported: needs MuJoCo)
PLACE_UNREACH_FRAC = 0.2  # unreachable share of the place window that fires PLACE_UNREACHABLE


def window_unreachable_fraction(
    pos_err_m: np.ndarray, timestamps_s: np.ndarray, window_s: tuple[float, float]
) -> float | None:
    """Share of the frames inside ``window_s`` whose IK position error exceeds the bar (None: no frame)."""
    t = np.asarray(timestamps_s, float)
    inside = (t >= window_s[0]) & (t <= window_s[1])
    if not inside.any():
        return None
    return float((np.asarray(pos_err_m, float)[inside] > UNREACHABLE_POS_M).mean())


def reach_events(reach: dict | None, t_s: float, phase: str, object_name: str | None) -> list[FeedbackEvent]:
    """``reach``: ``place_unreach_frac``, ``grasp_unreach_frac``, ``stance_scored_on_place`` of the plan."""
    if not reach or reach.get("place_unreach_frac") is None:
        return []
    place = float(reach["place_unreach_frac"])
    if place <= PLACE_UNREACH_FRAC:
        return []
    grasp = reach.get("grasp_unreach_frac")
    detail = f"{100 * place:.0f}% of the place window is out of the arm's reach at this stance"
    events = [
        FeedbackEvent(t_s=t_s, phase=phase, kind="PLACE_UNREACHABLE", object=object_name, detail=detail)
    ]
    if not reach.get("stance_scored_on_place") and (grasp is None or grasp <= PLACE_UNREACH_FRAC):
        events.append(
            FeedbackEvent(
                t_s=t_s,
                phase=phase,
                kind="STANCE_GRASP_ONLY",
                object=object_name,
                detail="stance scored on the human path only: the grasp is reachable, the place is not",
            )
        )
    return events
