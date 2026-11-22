"""Run FAH (video mode) on an exported clip's video.mp4 and save per-frame hand estimates.

Run inside the FAH venv. Pure RGB by default (K=None); ``--use-gt-k`` is an ablation only.
Output: ``<clip_dir>/hands_fah.npz`` with joints3d (N,2,21,3) [left,right], valid (N,2).
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--model", default="fah")
    ap.add_argument("--use-gt-k", action="store_true")
    ap.add_argument(
        "--focal", type=float, default=0.0, help="assumed focal in px (pinhole, centre principal point)"
    )
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    import fah

    K = None
    if a.use_gt_k:
        K = np.load(a.clip_dir / "gt" / "egodex_gt.npz")["intrinsics"]
    if a.focal:
        h, w = (
            cv2.VideoCapture(str(a.clip_dir / "video.mp4")).get(4),
            cv2.VideoCapture(str(a.clip_dir / "video.mp4")).get(3),
        )
        K = np.array([[a.focal, 0, w / 2], [0, a.focal, h / 2], [0, 0, 1.0]])
    model = fah.load(a.model, mode="video")
    model.reset()
    cap = cv2.VideoCapture(str(a.clip_dir / "video.mp4"))
    J, J2, S, valid = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and len(J) >= a.max_frames):
            break
        j = np.full((2, 21, 3), np.nan)
        j2 = np.full((2, 21, 2), np.nan)
        sc = np.zeros(2)
        v = np.zeros(2, bool)
        for h in model(frame, K=K):
            side = 1 if bool(h["is_right"]) else 0
            if not v[side]:
                j[side], j2[side], sc[side], v[side] = (
                    np.asarray(h["joints3d"]),
                    np.asarray(h["joints2d"]),
                    h["score"],
                    True,
                )
        J.append(j)
        J2.append(j2)
        S.append(sc)
        valid.append(v)
    suffix = "_gtk" if a.use_gt_k else (f"_f{round(a.focal)}" if a.focal else "")
    np.savez_compressed(
        a.clip_dir / f"hands_fah{suffix}.npz",
        joints3d=np.stack(J),
        joints2d=np.stack(J2),
        score=np.stack(S),
        image_wh=np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]),
        valid=np.stack(valid),
    )
    print("frames", len(J), "valid", np.stack(valid).mean(0), "FAH_HOME", os.environ.get("FAH_HOME"))


if __name__ == "__main__":
    main()
