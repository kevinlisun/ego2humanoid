"""Collect the per-clip retarget metrics into one reviewable table.

Usage:
    uv run --frozen python scripts/summarize_batch.py [root] [out.md]

Reads ``retarget_rgb.json`` (Dex3) and ``retarget_rgb_sharpa.json`` (Sharpa) from every clip under
``root`` and emits a markdown table plus the path of the review video for each row. The numbers are
the ones the simulator measured during the rollout, not predictions:

* ``success``   -- the object was lifted, translated, put down and came to rest.
* ``lift``      -- peak height of the object above its resting height, in metres.
* ``carry``     -- horizontal distance the object travelled, in metres.
* ``contact``   -- number of control steps with at least one hand/object contact.
* ``selfcol``   -- number of control steps with an arm/body self-collision.
* ``ik err``    -- mean wrist position error of the retargeting IK, in millimetres.
* ``valid``     -- fraction of frames where the hand estimate passed the score gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HANDS = (("dex3", "retarget_rgb.json", ""), ("sharpa", "retarget_rgb_sharpa.json", "_sharpa"))
COLUMNS = (
    "clip",
    "hand",
    "success",
    "lift m",
    "carry m",
    "contact",
    "selfcol",
    "ik err mm",
    "valid",
    "grid video",
)


def _row(clip: Path, hand: str, meta: dict, suffix: str) -> list[str]:
    po = meta.get("perceived_object") or {}
    ev = po.get("sim_events") or {}
    grid = clip / "review" / f"result_grid{suffix}.mp4"
    return [
        clip.name.replace("basic_pick_place_", ""),
        hand,
        "yes" if ev.get("success") else "no",
        f"{ev.get('lift_height_m', float('nan')):.3f}",
        f"{ev.get('translation_m', float('nan')):.3f}",
        str(po.get("contact_steps", "-")),
        str(meta.get("arm_collision_steps", "-")),
        f"{meta.get('ik_pos_err_mean_mm', float('nan')):.1f}",
        f"{meta.get('valid_frames', float('nan')):.2f}",
        str(grid) if grid.exists() else "-",
    ]


def collect(root: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for clip in sorted(root.iterdir(), key=lambda p: int(p.name.rsplit("_", 1)[-1])):
        for hand, name, suffix in HANDS:
            meta_path = clip / name
            if not meta_path.exists():
                continue
            rows.append(_row(clip, hand, json.loads(meta_path.read_text()), suffix))
    return rows


def to_markdown(rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    n = sum(1 for r in rows if r[2] == "yes")
    out += ["", f"{n}/{len(rows)} rollouts succeeded."]
    return "\n".join(out)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/derived/egodex")
    text = to_markdown(collect(root))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
