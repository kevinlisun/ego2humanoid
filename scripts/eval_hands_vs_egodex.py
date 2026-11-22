"""Compare FAH estimates (hands_fah*.npz) with EgoDex ARKit GT for one exported clip."""

import json
import sys
from pathlib import Path

import numpy as np

from evaluation.hand_metrics import evaluate_hands

clip = Path(sys.argv[1])
gt = np.load(clip / "gt" / "egodex_gt.npz")
report = {}
for est_file in sorted(clip.glob("hands_fah*.npz")):
    est = np.load(est_file)
    for i, side in enumerate(("left", "right")):
        n = len(est["valid"])
        valid = est["valid"][:, i] & (gt["confidence"][:n, i] > 0.5)
        if valid.any():
            report[f"{est_file.stem}/{side}"] = evaluate_hands(
                est["joints3d"][:, i], gt["joints_camera"][:n, i], valid
            )
print(json.dumps(report, indent=1))
(clip / "hand_eval.json").write_text(json.dumps(report, indent=1))
