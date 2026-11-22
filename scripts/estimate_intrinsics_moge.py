"""Estimate pinhole focal length from RGB only (MoGe-2, median over sampled frames).

Run inside the FAH venv (needs ``moge``). Writes ``<clip_dir>/intrinsics_est.json``.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

clip = Path(sys.argv[1])
n_samples = 8
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
cap = cv2.VideoCapture(str(clip / "video.mp4"))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fs = []
for i in np.linspace(0, n - 1, min(n_samples, n)).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    img = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32).permute(2, 0, 1)
    K = model.infer(img.cuda())["intrinsics"].cpu().numpy()
    fs.append(float(K[0, 0] * w))
out = {"focal_px": float(np.median(fs)), "width": w, "height": h, "per_frame": fs, "source": "moge2"}
(clip / "intrinsics_est.json").write_text(json.dumps(out, indent=1))
print(out["focal_px"])
