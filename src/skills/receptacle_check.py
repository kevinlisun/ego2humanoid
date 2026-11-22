"""Did the held object end up inside the receptacle, and did the place retry help? (Track RSI)

``check_place`` compares where the held object came to rest with the footprint of the receptacle's
collision (inner rim and outer wall, world frame): ``PLACE_OUTSIDE_RECEPTACLE`` when it rests beyond the
wall, ``RIM_BLOCKED`` when it rests on the wall ring or was caught by the rim above the floor.
``check_retry`` reads the plan's place-goal retry record (``scripts/retarget_estimated.py``) and flags a
retry that was thrown away because it was worse (``RETRY_WORSE``) or cut by the wall-time budget
(``RETRY_BUDGET_EXHAUSTED``). Both are pure functions on arrays and dicts, no simulator needed.
"""

from dataclasses import dataclass

import numpy as np

RIM_HEIGHT_FRAC = 0.5  # resting above this share of the wall height inside the rim = held up by the rim


@dataclass(frozen=True)
class ReceptacleGeometry:
    """Collision footprint of a welded receptacle in the world frame."""

    centre_xy_m: tuple[float, float]
    inner_half_xy_m: tuple[float, float]  # half extents of the inner opening at the rim
    outer_half_xy_m: tuple[float, float]  # half extents of the outer wall
    rim_z_m: float
    floor_z_m: float


@dataclass(frozen=True)
class PlaceCheck:
    outside: bool  # rests beyond the outer wall
    rim_blocked: bool  # rests on the wall ring, or hung on the rim at the release frame
    outside_m: float  # horizontal distance of the rest position beyond the inner rim (0 when inside)
    released_inside: bool  # the object was let go over the opening
    release_bottom_m: float  # height of the object's underside above the floor at the release frame
    rim_height_m: float  # rim above the floor

    def defects(self) -> tuple[str, ...]:
        return (
            *(("PLACE_OUTSIDE_RECEPTACLE",) if self.outside else ()),
            *(("RIM_BLOCKED",) if self.rim_blocked else ()),
        )

    def detail(self) -> str:
        if not self.defects():
            return "rests inside the receptacle"
        rest = f"{100 * self.outside_m:.1f} cm outside the rim" if self.outside_m > 0 else "on the wall"
        how = "over the opening" if self.released_inside else "outside the opening"
        return (
            f"rests {rest}; at the release it was {how}, underside {100 * self.release_bottom_m:.1f} cm "
            f"above the floor (rim {100 * self.rim_height_m:.1f} cm)"
        )


def _radius(xy: np.ndarray, centre: tuple[float, float], half: tuple[float, float]) -> float:
    d = (np.asarray(xy, float) - np.asarray(centre, float)) / np.maximum(np.asarray(half, float), 1e-6)
    return float(np.hypot(d[0], d[1]))


def check_place(
    end_pos_m: np.ndarray, release_pos_m: np.ndarray, held_half_z_m: float, g: ReceptacleGeometry
) -> PlaceCheck:
    """``end_pos_m`` / ``release_pos_m``: held object centre at rest and at the release frame."""
    end, rel = np.asarray(end_pos_m, float), np.asarray(release_pos_m, float)
    r_in = _radius(end[:2], g.centre_xy_m, g.inner_half_xy_m)
    r_out = _radius(end[:2], g.centre_xy_m, g.outer_half_xy_m)
    wall = max(g.rim_z_m - g.floor_z_m, 1e-6)
    rel_bottom = float(rel[2] - held_half_z_m - g.floor_z_m)
    rel_in = _radius(rel[:2], g.centre_xy_m, g.inner_half_xy_m) <= 1.0
    # let go at rim height while over the opening: the rim, not the hand, holds the object up
    hung = _radius(rel[:2], g.centre_xy_m, g.outer_half_xy_m) <= 1.0 and rel_bottom > RIM_HEIGHT_FRAC * wall
    return PlaceCheck(
        outside=r_out > 1.0,
        rim_blocked=1.0 < r_in and r_out <= 1.0 or hung,
        outside_m=max(r_in - 1.0, 0.0) * float(np.mean(g.inner_half_xy_m)),
        released_inside=rel_in,
        release_bottom_m=rel_bottom,
        rim_height_m=wall,
    )


@dataclass(frozen=True)
class RetryCheck:
    kind: str  # "RETRY_WORSE" | "RETRY_BUDGET_EXHAUSTED"
    detail: str


def check_retry(record: dict | None) -> RetryCheck | None:
    """``record``: the plan's ``place_retry`` dict (``outcome``, ``budget_s``, ``elapsed_s``, ``passes``)."""
    if not record:
        return None
    passes = record.get("passes", [])
    summary = "; ".join(
        f"pass {p['pass']}: grasp {p['frac']:.2f} place {p['place_frac']:.2f} unreachable {p['bad']:.3f}"
        for p in passes
    )
    outcome = record.get("outcome")
    if outcome == "restored_worse":
        return RetryCheck("RETRY_WORSE", f"retry rejected, first pass kept ({summary})")
    if outcome == "budget_exhausted":
        return RetryCheck(
            "RETRY_BUDGET_EXHAUSTED",
            f"{record.get('elapsed_s', 0):.0f} s of a {record.get('budget_s', 0):.0f} s budget, first pass "
            f"({summary})",
        )
    return None
