"""Object mass / friction estimation, following SimFoundry stage 11.

SimFoundry asks a VLM for ``{"mass": kg, "friction": mu}`` given the object image, its category phrase,
bounding box and volume. ``vlm_prompt`` and ``parse_physics_json`` implement that contract so any VLM or
LLM can be plugged in; ``heuristic_physics`` is the offline fallback (category keyword -> density table).
"""

import json
import re

import numpy as np

# (keywords, density kg/m^3 of the solid-equivalent object, friction); first match wins.
_DENSITY_TABLE = (
    (("plush", "stuffed", "toy", "doll", "puppy", "bear"), 120.0, 0.9),
    (("sponge", "foam"), 60.0, 1.0),
    (("cloth", "towel", "napkin", "sock", "shirt"), 150.0, 0.9),
    (("stapler", "tape", "scissor", "tool"), 450.0, 0.6),
    (("bottle", "cup", "mug", "bowl", "can", "jar"), 200.0, 0.5),
    (("block", "cube", "brick", "wood"), 600.0, 0.5),
    (("book", "box", "card"), 400.0, 0.5),
)
_DEFAULT = (300.0, 0.6)
# Below ~30 g the object is so light against a 0.8-1.8 kg hand that a contact impulse flings it
# (six clips sat at the old 10 g floor; clip 114: 13 g ejected at 3x the wrist speed).
MIN_MASS_KG = 0.03
_MAX_MASS_KG = 1.0  # a Dex3 pinch cannot lift more


def vlm_prompt(phrase: str, bbox_cm: np.ndarray, volume_cm3: float) -> str:
    bbox = " x ".join(f"{v:.2f}cm" for v in bbox_cm)
    return (
        f"Shown is a picture of a {phrase}. Please give the estimated mass (kg) and friction (unitless) of "
        f"the shown object, given that its bounding box dimensions are {bbox} and its volume is "
        f"{volume_cm3:.2f}cm^3.\nAnswer in JSON: " + '{"mass": <value>, "friction": <value>}'
    )


def parse_physics_json(text: str) -> dict[str, float]:
    match = re.search(r"\{.*?\}", text, re.S)
    if match is None:
        raise ValueError("no JSON object in response")
    data = json.loads(match.group(0))
    mass, friction = float(data["mass"]), float(data["friction"])
    if not (np.isfinite(mass) and np.isfinite(friction) and mass > 0 and friction > 0):
        raise ValueError(f"invalid physics values: {data}")
    return {
        "mass": float(np.clip(mass, MIN_MASS_KG, _MAX_MASS_KG)),
        "friction": float(np.clip(friction, 0.2, 2.0)),
    }


def heuristic_physics(phrase: str, volume_m3: float) -> dict[str, float]:
    density, friction = _DEFAULT
    words = phrase.lower()
    for keys, d, f in _DENSITY_TABLE:
        if any(k in words for k in keys):
            density, friction = d, f
            break
    mass = float(np.clip(density * volume_m3, MIN_MASS_KG, _MAX_MASS_KG))
    return {"mass": mass, "friction": friction}
