"""Visual parameters of the Blender digital twin that the appearance loop may tune (Track BT, BT.23).

Only how the twin *looks* (shape class, size, colour, light) is covered; MuJoCo physics parameters live
in ``sim_params``. Every leaf has a numeric range, so an edit proposed by an LLM is clamped to it and an
unknown path or value type is rejected instead of silently applied.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

TWIN_APPEARANCE_VERSION = "twin_appearance/1"

# Inclusive (low, high) per leaf name; applied element-wise to tuples.
RANGES: dict[str, tuple[float, float]] = {
    "size_m": (0.004, 0.6),
    "wall_m": (0.002, 0.05),
    "color_rgb": (0.0, 1.0),
    "roughness": (0.05, 1.0),
    "offset_m": (-0.05, 0.05),
    "yaw_deg": (-180.0, 180.0),
    "scale": (0.5, 2.0),
    "sun_energy": (0.0, 8.0),
    "sun_elevation_deg": (10.0, 89.0),
    "sun_azimuth_deg": (-180.0, 180.0),
    "world_strength": (0.0, 4.0),
    "world_rgb": (0.0, 1.0),
    "wall_rgb": (0.0, 1.0),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ObjectAppearance(_Model):
    shape: Literal["mesh", "box", "tray", "cylinder"] = "mesh"
    size_m: tuple[float, float, float] | None = None  # full extents in the object frame (None: mesh as is)
    wall_m: float = 0.01  # tray wall thickness
    texture: Literal["atlas", "flat", "projected"] = "atlas"  # atlas: generated mesh texture
    color_rgb: tuple[float, float, float] = (0.6, 0.55, 0.5)  # sRGB, used by `flat` and as the fallback
    roughness: float = 0.6
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # visual only; the physics body is unchanged
    yaw_deg: float = 0.0  # visual only
    scale: float = 1.0  # mesh shape only


class TableAppearance(_Model):
    color_rgb: tuple[float, float, float] = (0.72, 0.66, 0.58)
    roughness: float = 0.7


class LightAppearance(_Model):
    sun_energy: float = 2.0
    sun_elevation_deg: float = 55.0
    sun_azimuth_deg: float = 15.0
    world_rgb: tuple[float, float, float] = (1.0, 0.97, 0.92)
    world_strength: float = 0.8


class RoomAppearance(_Model):
    walls: bool = False
    wall_rgb: tuple[float, float, float] = (0.8, 0.76, 0.7)


class TwinAppearance(_Model):
    version: str = TWIN_APPEARANCE_VERSION
    objects: dict[str, ObjectAppearance] = {}
    table: TableAppearance = TableAppearance()
    light: LightAppearance = LightAppearance()
    room: RoomAppearance = RoomAppearance()


def _clamp(leaf: str, value: Any) -> Any:
    lo_hi = RANGES.get(leaf)
    if lo_hi is None:
        return value
    lo, hi = lo_hi

    def one(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"{leaf}: numbers expected")
        return min(max(float(x), lo), hi)

    return [one(x) for x in value] if isinstance(value, (list, tuple)) else one(value)


def apply_edits(app: TwinAppearance, edits: list[dict]) -> tuple[TwinAppearance, list[str]]:
    """Apply ``[{"path": "objects.scene_box.size_m", "value": [...]}, ...]``.

    Returns the new appearance and one message per rejected edit (unknown path, bad type or value).
    Numbers are clamped to ``RANGES``; an object that is not listed yet starts from its defaults.
    """
    data = app.model_dump()
    rejected: list[str] = []
    for edit in edits:
        try:
            path = str(edit["path"]).split(".")
            value = edit["value"]
            if path[0] == "objects":
                if len(path) != 3:
                    raise KeyError("objects.<name>.<field>")
                group = data["objects"].get(path[1]) or ObjectAppearance().model_dump()
                leaf = path[2]
            elif path[0] in ("table", "light", "room") and len(path) == 2:
                group, leaf = data[path[0]], path[1]
            else:
                raise KeyError("path")
            if leaf not in group:
                raise KeyError(leaf)
            candidate = {**group, leaf: _clamp(leaf, value)}
            if path[0] == "objects":
                ObjectAppearance.model_validate(candidate)
            else:
                type(getattr(app, path[0])).model_validate(candidate)
            group.update(candidate)
            if path[0] == "objects":
                data["objects"][path[1]] = group
        except (KeyError, ValueError, TypeError, ValidationError) as e:
            rejected.append(f"{edit.get('path') if isinstance(edit, dict) else edit}: {e}")
    return TwinAppearance.model_validate(data), rejected
