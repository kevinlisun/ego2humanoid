"""Where did the held object's mesh come from? (Track RSI)

A clip without its own crop mesh (``held_mesh`` stage never ran) used to be simulated silently as a
default box (clip 101: a red box instead of the tape). ``retarget_estimated.py`` now records the
provenance of the asset (``{"source": "crop" | "scene" | "box", ...}``, from
``stages.retargeting.held_mesh.resolve_held_asset``) and this skill turns it into events:

* ``HELD_MESH_FROM_SCENE``: the matching scene-inventory mesh stands in for the crop mesh (info).
* ``HELD_MESH_FALLBACK_BOX``: the default box was simulated because
  ``trajectory.allow_held_box_fallback`` is true; the hint is to run ``held_mesh`` and turn it off.

Without an allowed fallback a missing mesh aborts the plan (``HELD_MESH_MISSING``), so no run exists.
"""

from contracts.real2sim import FeedbackEvent

FALLBACK_BOX = "HELD_MESH_FALLBACK_BOX"
FROM_SCENE = "HELD_MESH_FROM_SCENE"


def provenance_events(held_asset: dict | None, held_name: str) -> list[FeedbackEvent]:
    """Events for a non-default asset source (nothing for the clip's own crop mesh or an unknown source)."""
    asset = held_asset or {}
    source = asset.get("source")
    if source == "box":
        detail = (
            "no mesh for the held object; simulated as a default box (trajectory.allow_held_box_fallback)"
        )
        return [FeedbackEvent(t_s=0.0, phase="setup", kind=FALLBACK_BOX, object=held_name, detail=detail)]
    if source == "scene":
        detail = f"mesh of the matching scene object ({asset.get('mesh', '?')}); no crop mesh"
        return [FeedbackEvent(t_s=0.0, phase="setup", kind=FROM_SCENE, object=held_name, detail=detail)]
    return []
