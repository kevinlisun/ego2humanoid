"""Clip selector: pick single-active-hand pick-and-place style clips from GT hand motion.

GT is used only to choose clips (dataset curation), never as a pipeline input.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from egodex.clip_reader import EgoDexClip, list_clips, load_clip

SIDES = ("left", "right")


@dataclass(frozen=True)
class ClipScore:
    clip_id: str
    path: Path
    active: str
    travel_m: float
    other_travel_m: float
    duration_s: float
    accepted: bool


def _travel(wrist_xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(wrist_xyz, axis=0), axis=1).sum())


def score_clip(
    clip: EgoDexClip, path: Path, min_travel=0.3, max_other=0.15, min_s=3.0, max_s=15.0
) -> ClipScore:
    travel = [_travel(clip.T_world_wrist[:, i, :3, 3]) for i in range(2)]
    hinted = clip.active_hand()
    active = hinted or SIDES[int(np.argmax(travel))]
    a = SIDES.index(active)
    dur = clip.n_frames / 30.0
    ok = travel[a] >= min_travel and travel[1 - a] <= max_other and min_s <= dur <= max_s
    return ClipScore(clip.clip_id, path, active, travel[a], travel[1 - a], dur, ok)


def select_clips(root: str | Path, **kw) -> list[ClipScore]:
    scores = [score_clip(load_clip(p), p, **kw) for p in list_clips(root)]
    return sorted(scores, key=lambda s: (not s.accepted, -s.travel_m))
