"""Compare EgoTac contact on the active hand with the perceived hold window of each clip.

Usage: eval_tactile_vs_hold.py <egodex_derived_dir> [out.json]. Needs only numpy; reads
``tactile_egotac.npz`` and ``retarget_rgb.json`` per clip. No ground truth is used.
"""

import glob
import json
import os
import sys

import numpy as np

root = sys.argv[1]
rows = []
for clip in sorted(glob.glob(os.path.join(root, "basic_pick_place_*"))):
    tp, rp = os.path.join(clip, "tactile_egotac.npz"), os.path.join(clip, "retarget_rgb.json")
    if not (os.path.exists(tp) and os.path.exists(rp)):
        continue
    r = json.load(open(rp))
    po = r.get("perceived_object") or {}
    hold, side = po.get("hold_s"), r.get("side")
    cp = np.load(tp)["contact_prob"].max(2)
    on = cp[:, 0 if side == "left" else 1] > 0.5
    row = {
        "clip": os.path.basename(clip),
        "side": side,
        "tactile_on_frames": int(on.sum()),
        "frames": len(on),
    }
    if hold:
        t = np.arange(len(on)) / 30.0
        g = (t >= hold[0]) & (t <= hold[1])
        row["iou"] = float((on & g).sum() / max(1, (on | g).sum()))
        row["recall"] = float((on & g).sum() / max(1, g.sum()))
    row["success"] = (po.get("sim_events") or {}).get("success")
    rows.append(row)
    print(row)
if len(sys.argv) > 2:
    json.dump(rows, open(sys.argv[2], "w"), indent=2)
