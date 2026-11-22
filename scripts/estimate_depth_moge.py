"""Per-frame metric depth with MoGe-2 (run inside the FAH venv).

Writes ``<clip_dir>/depth_moge.npz`` (``depth``: T x h x w float16, half resolution).
Uses the clip focal from ``intrinsics_est.json`` so depth and VO share intrinsics.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

clip = Path(sys.argv[1])
fov_x = None
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
intr = json.loads((clip / "intrinsics_est.json").read_text())
cap = cv2.VideoCapture(str(clip / "video.mp4"))
out = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    img = (
        torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .cuda()
    )
    fov = float(np.degrees(2 * np.arctan(w / (2 * intr["focal_px"]))))
    with torch.no_grad():
        d = model.infer(img, fov_x=fov)["depth"].cpu().numpy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    out.append(cv2.resize(d, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST).astype(np.float16))
np.savez_compressed(clip / "depth_moge.npz", depth=np.stack(out))
print("depth", len(out), out[0].shape)
