"""Apply `SimParams.objects` scene edits (enabled / static / robot_collision / pose / yaw / scale)."""

import math
from dataclasses import replace

import numpy as np

from contracts.sim_params import ObjectPhysics, SimParams
from stages.manipulation.mjcf_scene import BoxObject

ALIASES = ("held", "receptacle", "source")


def _selected(params: SimParams, name: str, roles: dict[str, str]) -> list[ObjectPhysics]:
    """Overrides for `name`: `*` first, then role aliases, then the exact name (most specific last)."""
    out = []
    for key in ("*", *[r for r, n in roles.items() if n == name], name):
        if key in params.objects:
            out.append(params.objects[key])
    return out


def check_selectors(params: SimParams, names: list[str], roles: dict[str, str]) -> None:
    valid = {"*", *ALIASES, *names}
    unknown = [k for k in params.objects if k not in valid]
    if unknown:
        raise ValueError(f"unknown object selector(s) {unknown}; valid: {sorted(valid)}")


def _rotate_z(v: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return v @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _edit(obj: BoxObject, o: ObjectPhysics) -> BoxObject:
    changes: dict = {}
    scale = np.asarray(o.size_scale, float)
    yaw = math.radians(o.yaw_offset_deg)
    if (scale != 1.0).any() or yaw != 0.0:
        if obj.mesh_vertices is None and yaw != 0.0:
            raise ValueError(f"object {obj.name!r} is a box: yaw_offset_deg needs a mesh object")
        changes["half_size_m"] = tuple(np.asarray(obj.half_size_m, float) * scale)

        def tf(v):
            return _rotate_z(np.asarray(v, float) * scale, yaw)

        if obj.mesh_vertices is not None:
            changes["mesh_vertices"] = tf(obj.mesh_vertices)
        if obj.collision_pieces:
            changes["collision_pieces"] = tuple((tf(v), f) for v, f in obj.collision_pieces)
    if any(o.pos_offset_m):
        changes["pos_m"] = tuple(np.asarray(obj.pos_m, float) + np.asarray(o.pos_offset_m, float))
    if o.static is not None:
        changes["static"] = o.static
    if o.robot_collision is not None:
        changes["robot_collision"] = o.robot_collision
    return replace(obj, **changes) if changes else obj


def apply_scene_overrides(
    objects: list[BoxObject], params: SimParams, roles: dict[str, str] | None = None
) -> list[BoxObject]:
    """`objects` with the configured edits; disabled objects are dropped. Unknown selectors raise."""
    roles = roles or {}
    check_selectors(params, [o.name for o in objects], roles)
    out = []
    for obj in objects:
        edits = _selected(params, obj.name, roles)
        if any(not e.enabled for e in edits):
            continue
        for e in edits:
            obj = _edit(obj, e)
        out.append(obj)
    return out
