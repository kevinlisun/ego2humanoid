"""Rebuild ``result_grid<suffix>.mp4`` from the cached panels, re-doing only the ego panel.

The bottom-left panel is the one that needs CUDA (the background splat is rasterised in another
environment), so the expensive retarget + rollout is not repeated: the other three panels are read
back from ``review/panels/`` and the ego panel is recomposited from the MuJoCo capture and the
freshly rendered background.

Usage: python scripts/make_review_grid.py <clip_dir> [suffix]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from stages.qa.grid_video import compose_grid
from stages.qa.review_paths import existing_panels_dir
from stages.reconstruction.scene.neural_ego import composite_neural_ego

EGO_PANEL = "robot ego view (Unitree G1 head D435, 3DGS background)"


def read_video(path: str | Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main(clip: Path, suffix: str) -> Path:
    panels = existing_panels_dir(clip)
    hand = suffix.lstrip("_") or "dex3"
    bg = panels / f"bg_rgbd{suffix}.npz"
    ego = composite_neural_ego(panels / f"ego_d435{suffix}.npz", bg if bg.exists() else None)
    return compose_grid(
        [
            ("RGB + estimated hands (+tactile)", read_video(panels / f"tl{suffix}.mp4")),
            (f"robot in MuJoCo ({hand})", read_video(panels / f"tr{suffix}.mp4")),
            (EGO_PANEL if bg.exists() else EGO_PANEL + " [sim only]", ego),
            ("task-space trajectories", read_video(panels / f"br{suffix}.mp4")),
        ],
        clip / "review" / f"result_grid{suffix}.mp4",
    )


if __name__ == "__main__":
    print("grid video:", main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else ""))
