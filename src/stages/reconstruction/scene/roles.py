"""Task roles of the inventory objects: held, source (stand / container) and receptacle.

Split out of ``build_scene_inventory`` (see ``inventory.py``). The roles come from the task phrases
and from geometry around the grip. Both can disagree: the detector's names for look-alike stands
are sometimes swapped (clip 108: the "round table" the pen is picked from lies 21 cm from the grip,
the "square table" under it). ``SimParams.scene.source_from`` selects which evidence wins and
``scene.max_registration_m`` caps the grip-to-source shift that is still trusted
(``skills/scene_layout.py`` reports the cases and names these knobs).
"""

from __future__ import annotations

import colorsys
from dataclasses import replace

import numpy as np

from stages.reconstruction.objects.place_target import _SYNONYM_OF, match_object_name

CONTAINER_WORDS = (
    "bowl", "tray", "lid", "tupperware", "scoop", "scooper", "basket", "container", "plate", "box", "case",
)  # fmt: skip
RECEPTACLE_MATCH_M = 0.20
SOURCE_INNER_FRAC = 0.6  # floor of a source container, as a fraction of its footprint
COLOUR_SWAP_HUE_DEG = 70.0  # a task colour overrides the detector's category only this far off
COLOUR_SWAP_MIN_SAT = 0.3
RELEASE_INSIDE_FRAC = 0.9  # an unnamed receptacle is a container the late carry passes over ...
RELEASE_MIN_FRAMES = 5  # ... for at least this many frames
SUPPORT_MIN_TOP_M = 0.03  # an object under the grip at least this tall is a stand, not the table
HELD_FALLBACK_M = 0.04  # a named but undetected held object: the fallback must lie this close to the grip
HELD_MAX_AREA_RATIO = 4.0  # a container around the grip this much larger than the held box holds it
_COLOUR_WORDS = frozenset(_SYNONYM_OF)
_HUES = frozenset({"red", "orange", "yellow", "green", "blue", "purple"})


def _hsv(rgb) -> tuple[float, float, float]:
    h, sat, val = colorsys.rgb_to_hsv(*(np.asarray(rgb, float) / 255.0))
    return h * 360.0, sat, val


def _hue_gap(a, b) -> float:
    d = abs(_hsv(a)[0] - _hsv(b)[0]) % 360.0
    return min(d, 360.0 - d)


def colour_name(rgb) -> str:
    """Coarse colour word of a median RGB, in the vocabulary of the task descriptions."""
    hue, sat, val = _hsv(rgb)
    if val < 0.2:
        return "black"
    if sat < 0.18:
        return "white" if val > 0.75 else "grey"
    for name, upper in (("red", 15), ("orange", 40), ("yellow", 70), ("green", 170), ("blue", 255)):
        if hue < upper:
            return name
    return "purple" if hue < 345 else "red"


def is_container(label: str) -> bool:
    return any(w in label.lower().split("_") or w in label.lower().split() for w in CONTAINER_WORDS)


def inside(fp, xy: np.ndarray, shrink: float = 1.0) -> bool:
    d = xy - fp.center_xy
    u = np.array([d @ fp.axis_xy, d @ np.array([-fp.axis_xy[1], fp.axis_xy[0]])])
    return bool((np.abs(u) <= shrink * fp.half_xy).all())


def clamp_inside(fp, xy: np.ndarray) -> np.ndarray:
    side = np.array([-fp.axis_xy[1], fp.axis_xy[0]])
    d = xy - fp.center_xy
    u = np.clip([d @ fp.axis_xy, d @ side], -fp.half_xy, fp.half_xy)
    return fp.center_xy + u[0] * fp.axis_xy + u[1] * side


class PhraseMatcher:
    """Task phrase -> inventory id by name, with the colour correction of the detector's categories."""

    def __init__(self, names: dict, labels: dict, colours: dict, rgbs: dict, fps: dict):
        self.names, self.labels, self.colours, self.rgbs, self.fps = names, labels, colours, rgbs, fps
        self.relabelled: dict[str, str] = {}

    def __call__(self, phrase: str | None, exclude: set[int]) -> int | None:
        if not phrase:
            return None
        names, labels, colours, rgbs, fps = self.names, self.labels, self.colours, self.rgbs, self.fps
        cands = {f"{labels[i]} {names[i]}": i for i in names if i not in exclude}
        hit = match_object_name(list(cands), phrase)
        # Open-vocabulary labels are generic ("box" for "box lid"): fall back to the modifier nouns,
        # head first, never to a bare colour.
        nouns = [t for t in phrase.lower().split() if t not in _COLOUR_WORDS]
        for noun in reversed(nouns[:-1]):
            if hit is not None:
                break
            hit = match_object_name(list(cands), noun)
        if hit is None:
            return None
        # Open-vocabulary detectors confuse categories more than colours (clip 10: the yellow bottle
        # was called "scoop", the purple scoop "spatula"). A named colour that contradicts the hit
        # moves the phrase to the one object of that colour, which then takes the phrase's noun.
        # Only hues are trusted: lighting turns black into grey and white into beige.
        want = {_SYNONYM_OF[t] for t in phrase.lower().split() if t in _SYNONYM_OF} & _HUES
        # Guards against noisy colour names (clip 116: the pale green bowl reads "yellow" and the
        # one "green" object is the small block): the hues must be far apart and the replacement
        # comparable in size.
        h = cands[hit]
        if want and colours[h] not in want:
            area = np.prod(fps[h].half_xy)
            same = [
                i
                for i in cands.values()
                if colours[i] in want
                and _hue_gap(rgbs[i], rgbs[h]) > COLOUR_SWAP_HUE_DEG
                and _hsv(rgbs[i])[1] >= COLOUR_SWAP_MIN_SAT
                and np.prod(fps[i].half_xy) >= 0.5 * area
            ]
            if len(same) == 1:
                labels[same[0]] = nouns[-1] if nouns else labels[same[0]]
                self.relabelled[names[same[0]]] = labels[same[0]]
                return same[0]
        return cands[hit]


def find_held(
    match: PhraseMatcher,
    held_xy: np.ndarray,
    held_phrase: str,
    place_phrase: str | None,
    source_phrase: str | None,
    held_half_xy: np.ndarray | None,
    held_match_m: float,
) -> int | None:
    """The held object: by name near the grip point, else simply the nearest object to it."""
    fps, labels = match.fps, match.labels
    near_held = [i for i in fps if np.linalg.norm(fps[i].center_xy - held_xy) < held_match_m]
    held_id = match(held_phrase, set(fps) - set(near_held))
    if held_id is not None:
        return held_id
    # Not detected by name (a plush in a bowl): never fall back onto the named containers.
    named = {match(source_phrase, set()), match(place_phrase, set())}
    if held_half_xy is not None:
        # Nor onto the container the grip is in, under another name (clip 117: the "silver
        # tray" was detected as "tupperware" and became the held block, so it never existed).
        area = 4 * float(np.prod(held_half_xy))
        named |= {
            i
            for i in near_held
            if is_container(labels[i])
            and inside(fps[i], held_xy)
            and 4 * float(np.prod(fps[i].half_xy)) > HELD_MAX_AREA_RATIO * area
        }
    near_held = [i for i in near_held if i not in named]
    gap = {i: float(np.linalg.norm(held_xy - clamp_inside(fps[i], held_xy))) for i in near_held}
    if held_phrase:
        # A named object that no detection matches was likely not seen at all; a generic task
        # ("pick up an object") names nothing, so the nearest object is the best evidence.
        near_held = [i for i in near_held if gap[i] < HELD_FALLBACK_M]
    return min(near_held, key=gap.get) if near_held else None


def pick_source(
    match: PhraseMatcher,
    fps: dict,
    held_id: int | None,
    held_xy: np.ndarray,
    held_half_xy: np.ndarray | None,
    source_phrase: str | None,
    source_from: str,
    report: dict,
) -> int | None:
    """Source container / stand: named in the task, or the container the held object stands in."""
    labels, names = match.labels, match.names
    src_id = match(source_phrase, {held_id})
    if src_id is None and source_phrase:
        holding = [
            i for i in fps if i != held_id and is_container(labels[i]) and inside(fps[i], held_xy, 1.1)
        ]
        src_id = max(holding, key=lambda i: fps[i].half_xy.prod(), default=None)
    if src_id is None or inside(fps[src_id], held_xy, 1.1):
        return src_id
    report["source_by_name"] = names[src_id]
    if source_from == "name":
        report["source_from"] = "name"
        return src_id
    # The detector's names for look-alike stands can be swapped (clip 108): by default the geometry
    # under the grip wins over the name.
    under = [
        i
        for i in fps
        if i not in (held_id, src_id)
        and fps[i].top_m >= SUPPORT_MIN_TOP_M
        and inside(fps[i], held_xy, 1.1)
        and (held_half_xy is None or fps[i].half_xy.prod() > np.prod(held_half_xy))
    ]
    if under:
        src_id = min(under, key=lambda i: float(np.linalg.norm(fps[i].center_xy - held_xy)))
        report["source_from"] = "grip"
    return src_id


def source_shift(fp, held_xy: np.ndarray, held_half_xy: np.ndarray | None) -> np.ndarray:
    """Translation that puts the grip on the source's inner floor (never inside its wall)."""
    # The plush hidden in clip 10's bowl is not detected at all, and clip 114's duck matched with an
    # offset too large to trust and started on the tray's wall (launched at the first step).
    reach = np.zeros(2) if held_half_xy is None else np.full(2, float(np.max(held_half_xy)))
    inner = replace(fp, half_xy=np.maximum(SOURCE_INNER_FRAC * fp.half_xy - reach, 0.0))
    return held_xy - clamp_inside(inner, held_xy)


def pick_receptacle(
    match: PhraseMatcher,
    held_id: int | None,
    src_id: int | None,
    held_xy: np.ndarray,
    place_phrase: str | None,
    place_hint_xy: np.ndarray | None,
    carry_xy: np.ndarray | None,
    report: dict,
) -> int | None:
    fps, labels = match.fps, match.labels
    rec_id = match(place_phrase, {held_id, src_id})
    if rec_id is None and place_phrase and place_hint_xy is not None:
        near = [
            i
            for i in fps
            if i not in (held_id, src_id)
            and np.linalg.norm(fps[i].center_xy - place_hint_xy) < RECEPTACLE_MATCH_M
        ]
        rec_id = min(near, key=lambda i: np.linalg.norm(fps[i].center_xy - place_hint_xy), default=None)
    if rec_id is None and not place_phrase and carry_xy is not None:
        # A generic task ("place it somewhere else"): the container the human pinch lingers over in
        # the second half of the carry (the detected release can come late: clip 101 lets go in the
        # case at 6.2 s, the object track ends at 7.6 s with the hand already away).
        dwell = {
            i: sum(inside(fps[i], xy, RELEASE_INSIDE_FRAC) for xy in np.asarray(carry_xy, float))
            for i in fps
            if i not in (held_id, src_id) and is_container(labels[i]) and not inside(fps[i], held_xy)
        }
        best = max(dwell, key=dwell.get, default=None)
        if best is not None and dwell[best] >= RELEASE_MIN_FRAMES:
            rec_id = best
            report["receptacle_from"] = "carry"
    return rec_id
