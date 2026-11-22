"""Review video for SimFoundry stage 5 (decompose scene): detection, mask, and object removal per iteration.

Usage: python scripts/viz_stage5_removal.py data/scenes/<scene> [out.mp4]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from stages.reconstruction.scene.viz import write_slideshow

PANELS = (
    ("detected_phrases", "detected"),
    ("removal_mask", "removal mask"),
    ("pre_object_removal", "before removal"),
    ("post_object_removal", "after removal"),
)


def build_slides(s5: Path) -> list[list[tuple[str, np.ndarray]]]:
    slides = []
    iters = sorted({int(p.stem.split("_")[1]) for p in (s5 / "pre_object_removal").glob("iter_*.png")})
    for k in iters:
        row = []
        for sub, title in PANELS:
            f = s5 / sub / f"iter_{k}.png"
            if f.exists():
                row.append((f"iter {k}: {title}", cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)))
        if row:
            slides.append(row)
    return slides


def main() -> None:
    scene = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else scene / "review" / "stage5_removal.mp4"
    slides = build_slides(scene / "s5_scene")
    if not slides:
        raise SystemExit("no stage-5 removal iterations found")
    write_slideshow(out, slides, panel_hw=(360, 480), fps=10, hold_s=2.5)
    print(out)


if __name__ == "__main__":
    main()
