"""L.8: sample static manipulation stances around the object (annulus x heading x height x waist pitch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Supported postures from configs/dwbc_profile.json (L.3).
HEIGHTS_M = (0.6, 0.7, 0.74)
WAIST_PITCH_RAD = (0.0, 0.2, 0.4, 0.5)
RADII_M = (0.3, 0.35, 0.4, 0.45, 0.55, 0.65)
BEARING_STEP_DEG = 30.0
YAW_OFFSETS_RAD = (0.0, 0.15, -0.15, 0.3, -0.3)


@dataclass(frozen=True)
class Stance:
    x: float
    y: float
    yaw: float
    height_m: float
    waist_pitch_rad: float


def sample_candidates(
    object_xy, grid, extent, res: float, focus_xy=None
) -> tuple[list[Stance], dict[str, int]]:
    """Stances on the annulus around the object, facing ``focus_xy`` (default: the object).

    Cells blocked in ``grid`` are rejected (footprint).
    """
    ox, oy = object_xy
    fx, fy = focus_xy if focus_xy is not None else object_xy
    kept, rejected = [], {"footprint_blocked": 0, "out_of_map": 0}
    for bearing in np.radians(np.arange(0.0, 360.0, BEARING_STEP_DEG)):
        for r in RADII_M:
            x, y = ox + r * np.cos(bearing), oy + r * np.sin(bearing)
            row, col = int((y - extent[1][0]) / res), int((x - extent[0][0]) / res)
            if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
                rejected["out_of_map"] += 1
                continue
            if grid[row, col]:
                rejected["footprint_blocked"] += 1
                continue
            facing = float(np.arctan2(fy - y, fx - x))
            for dyaw in YAW_OFFSETS_RAD:
                for h in HEIGHTS_M:
                    for wp in WAIST_PITCH_RAD:
                        kept.append(Stance(float(x), float(y), facing + dyaw, h, wp))
    return kept, rejected
