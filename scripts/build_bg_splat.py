"""Build the background Gaussian splat of a reconstructed scene. See ``scene/bg_splat.py``.

Usage: python scripts/build_bg_splat.py <scene_dir> <clip_dir> [out_npz]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stages.reconstruction.scene.bg_splat import build_background_splat

if __name__ == "__main__":
    scene, clip = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else scene / "bg_splat.npz"
    splat, report = build_background_splat(scene, clip)
    splat.save(out)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2))
    print("wrote", out, f"({report['n_gaussians']} gaussians)")
