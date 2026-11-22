"""Estimate per-frame ego camera poses of a (walking) clip with lightweight VO.

Needs ``depth_moge.npz`` (scripts/estimate_depth_moge.py), ``intrinsics_est.json`` and
``hands_fah_f645.npz`` (hand pixels are masked out). Writes ``<clip_dir>/camera_poses.npz``
(``poses`` T_world_cam, frame 0 = identity; ``ok``; ``inliers``) and a top-down plot in ``review/``.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from stages.reconstruction.align.ego_vo import estimate_poses

clip = Path(sys.argv[1])
intr = json.loads((clip / "intrinsics_est.json").read_text())
cap = cv2.VideoCapture(str(clip / "video.mp4"))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
frames = np.stack(frames)
T, H, W = frames.shape[:3]
depths = np.load(clip / "depth_moge.npz")["depth"].astype(np.float32)
K = np.array([[intr["focal_px"], 0, W / 2], [0, intr["focal_px"], H / 2], [0, 0, 1.0]])

masks = np.zeros((T, H, W), dtype=bool)
hands_path = clip / "hands_fah_f645.npz"
if hands_path.exists():
    hz = np.load(hands_path)
    for t in range(min(T, len(hz["valid"]))):
        for s in range(2):
            if hz["valid"][t, s]:
                pts = hz["joints2d"][t, s].astype(np.int32)
                hull = cv2.convexHull(pts)
                m = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(m, hull, 1)
                masks[t] |= cv2.dilate(m, np.ones((61, 61), np.uint8)).astype(bool)

stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
res = estimate_poses(frames, K, depths, masks, stride=stride)
np.savez(clip / "camera_poses.npz", poses=res.poses, ok=res.ok, inliers=res.inliers)
pos = res.poses[:, :3, 3]
print(f"frames={T} ok={res.ok.mean():.2%} median_inliers={np.median(res.inliers[1:]):.0f}")
print(
    "path length m:",
    float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
    "net displacement m:",
    pos[-1].round(2),
)

canvas = np.full((500, 500, 3), 255, np.uint8)
xz = pos[:, [0, 2]]
span = max(1.0, np.abs(xz).max() * 1.2)
px = ((xz / span) * 240 + 250).astype(int)
for a, b in zip(px[:-1], px[1:]):
    cv2.line(canvas, tuple(a), tuple(b), (200, 0, 0), 2)
cv2.putText(canvas, f"top-down x/z, +-{span:.1f} m", (10, 20), 0, 0.6, (0, 0, 0), 1)
(clip / "review").mkdir(exist_ok=True)
cv2.imwrite(str(clip / "review/camera_trajectory.png"), canvas)
