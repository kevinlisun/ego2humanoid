"""Is the registered scene layout the one the human demonstration shows? (Track RSI, BF.6)

Pure dict logic on the report of ``build_scene_inventory`` (stored in the sim bundle as
``scene_layout``). Three defects are detected; each maps to ``scene.*`` knobs in ``attribution.py``:

* ``SCENE_ROLE_SWAP``: the object under the grip overrode the task's named source (look-alike stands
  with swapped detector names, clip 108).
* ``SCENE_RECEPTACLE_REBASED``: the receptacle was moved far onto the release point, so the layout
  around it no longer matches the video.
* ``SCENE_GRIP_GAP``: the grip lies far from the source/held object (the inventory is dropped past
  ``scene.max_registration_m``).

Thresholds are code constants, not agent-tunable.
"""

import math

from contracts.real2sim import FeedbackEvent

GRIP_GAP_M = 0.08  # grip-to-source shift beyond which the registration is suspect
REBASE_M = 0.10  # receptacle shift beyond which the layout around it is suspect


def layout_events(report: dict | None) -> list[FeedbackEvent]:
    """Events of the inventory report; empty for a clip without an inventory."""
    if not report:
        return []
    out: list[FeedbackEvent] = []

    def add(kind: str, obj: str | None, detail: str) -> None:
        out.append(FeedbackEvent(t_s=0.0, phase="setup", kind=kind, object=obj, detail=detail))

    if report.get("source_from") == "grip" and report.get("source_by_name"):
        add(
            "SCENE_ROLE_SWAP",
            report.get("source"),
            f"source taken from under the grip ({report.get('source')}), "
            f"the task names {report['source_by_name']}",
        )
    shift = report.get("receptacle_shift_m")
    if report.get("receptacle_rebased") and shift is not None and math.hypot(*shift[:2]) > REBASE_M:
        add(
            "SCENE_RECEPTACLE_REBASED",
            report.get("receptacle"),
            f"receptacle moved {100 * math.hypot(*shift[:2]):.0f} cm onto the release point",
        )
    gap = report.get("grip_gap_m")
    if report.get("unregistered") or (gap is not None and gap > GRIP_GAP_M):
        gap_txt = "n/a" if gap is None else f"{100 * gap:.0f} cm"
        state = "inventory dropped" if report.get("unregistered") else "layout shifted to the grip"
        add(
            "SCENE_GRIP_GAP", report.get("source") or report.get("held_match"), f"grip gap {gap_txt}: {state}"
        )
    return out
