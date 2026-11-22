"""Review video for SimFoundry stage 6: per-object crop (padded) -> FLUX upsampled -> background removed.

Usage: python scripts/viz_stage6_upsample.py data/scenes/<scene> [out.mp4]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from stages.reconstruction.scene.viz import write_slideshow


def _load(f: Path) -> np.ndarray:
    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:  # composite RGBA on grey so the cutout is visible
        a = img[..., 3:4] / 255.0
        img = (img[..., :3] * a + 128 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_slides(s6: Path) -> list[list[tuple[str, np.ndarray]]]:
    slides = []
    for padded in sorted((s6 / "padded").glob("iter_*.png")):
        k = padded.stem
        row = [(f"{k}: input crop", _load(padded))]
        for sub, name, title in (
            ("upsampled", f"{k}.png", "FLUX upsampled"),
            ("upsampled", f"{k}_transparent.png", "background removed"),
        ):
            f = s6 / sub / name
            if f.exists():
                row.append((f"{k}: {title}", _load(f)))
        slides.append(row)
    return slides


def main() -> None:
    scene = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else scene / "review" / "stage6_upsample.mp4"
    slides = build_slides(scene / "s6_upsample")
    if not slides:
        raise SystemExit("no stage-6 outputs found")
    write_slideshow(out, slides, panel_hw=(400, 400), fps=10, hold_s=2.5)
    print(out)


if __name__ == "__main__":
    main()
