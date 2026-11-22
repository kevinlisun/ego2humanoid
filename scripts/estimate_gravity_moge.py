"""Estimate the gravity-up direction (camera frame, OpenCV) from RGB via MoGe-2 normals.

Tabletop scenes are dominated by horizontal surfaces: the up vector is the dominant surface normal found
by a RANSAC-style vote over the lower image half, averaged over sampled frames. Run in the FAH venv.
Writes ``<clip_dir>/gravity_est.json``.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel


def dominant_normal(
    normals: np.ndarray, iters: int = 300, thresh_deg: float = 8.0, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cand = normals[rng.integers(0, len(normals), iters)]
    cos_t = np.cos(np.radians(thresh_deg))
    score = np.array([(np.abs(normals @ c) > cos_t).sum() for c in cand])
    best = cand[score.argmax()]
    inl = normals[np.abs(normals @ best) > cos_t]
    inl = inl * np.sign(inl @ best)[:, None]
    m = inl.mean(0)
    return m / np.linalg.norm(m)


clip = Path(sys.argv[1])
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
cap = cv2.VideoCapture(str(clip / "video.mp4"))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ups = []
for i in np.linspace(0, n - 1, min(8, n)).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, frame = cap.read()
    if not ok:
        continue
    img = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32).permute(2, 0, 1)
    out = model.infer(img.cuda())
    nrm = out["normal"].cpu().numpy()
    mask = out["mask"].cpu().numpy()
    h = nrm.shape[0]
    sel = np.zeros_like(mask)
    sel[h // 2 :] = True
    pts = nrm[mask & sel]
    pts = pts[np.linalg.norm(pts, axis=1) > 0.5]
    pts = pts[:: max(1, len(pts) // 20000)]
    u = dominant_normal(pts)
    ups.append(u if u[1] < 0 else -u)  # OpenCV camera: up is -y
m = np.mean(ups, axis=0)
m /= np.linalg.norm(m)
(clip / "gravity_est.json").write_text(
    json.dumps({"up_in_camera": m.tolist(), "per_frame": [u.tolist() for u in ups]})
)
print(m)
