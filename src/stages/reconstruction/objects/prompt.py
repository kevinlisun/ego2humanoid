"""Derive SAM 3 text prompts for the manipulated object and the place target from the task description."""

import re

_PICK = re.compile(
    r"pick(?:s)? up (?:a |an |the )?(?P<obj>.+?)(?: from | with | and | to | in | on |\.|$)", re.I
)
# "place it on the blue bowl", "put the cup into the sink", "drop it in the box".
_PLACE = re.compile(
    r"(?:place|put|drop|set)(?:s)? (?:it|them|the [\w\- ]+?) "
    r"(?:on top of|onto|into|on|in|inside|over|next to|beside) "
    r"(?:a |an |the )?(?P<dst>.+?)(?: with | and | using |\.|$)",
    re.I,
)


def object_prompt(task_description: str, default: str = "object") -> str:
    m = _PICK.search(task_description or "")
    obj = m.group("obj").strip() if m else ""
    return obj if obj and obj.lower() not in {"an object", "object"} else default


def place_target_prompt(task_description: str) -> str | None:
    """The receptacle the object is placed on/in, e.g. "blue bowl", or None for a bare transfer.

    The hand has to be accurate about this object at release in exactly the way it is accurate about
    the grasped object at closure, so it is resolved from the same sentence the grasp prompt is.
    """
    m = _PLACE.search(task_description or "")
    dst = m.group("dst").strip().rstrip(".") if m else ""
    return dst or None


_SOURCE = re.compile(r"\bfrom (?:a |an |the )?(?P<src>.+?)(?: with | and | to |\.|$)", re.I)


def source_prompt(task_description: str) -> str | None:
    """The support the object is picked from, e.g. "blue bowl", or None when it is just the table."""
    m = _SOURCE.search(task_description or "")
    src = m.group("src").strip().rstrip(".") if m else ""
    return src if src and src.lower() != "table" else None
