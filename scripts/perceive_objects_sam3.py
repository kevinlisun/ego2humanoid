"""RGB-only object perception: SAM 3 text-prompted masks + MoGe-2 point map -> object 3D track.

Run inside the FAH venv (needs transformers>=5 and moge). Usage:
  perceive_objects_sam3.py <clip_dir> "<text prompt>" [stride] [threshold]
Writes ``<clip_dir>/objects_sam3.npz``: frame_idx, score (N,K), centroid_cam (N,K,3), extent_cam (N,K,3),
depth_small.
No ground truth is used. The camera frame is OpenCV (x right, y down, z forward).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from stages.reconstruction.objects.hf_weights import require_model

clip = Path(sys.argv[1])
prompt = sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3

require_model("facebook/sam3", "SAM3")
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()

TOPK = 5
cap = cv2.VideoCapture(str(clip / "video.mp4"))
frames, cents, exts, scores, depths = [], [], [], [], []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % stride == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        inputs = proc(images=Image.fromarray(rgb), text=prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = sam(**inputs)
        res = proc.post_process_instance_segmentation(out, threshold=thr, target_sizes=[(h, w)])[0]
        img = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
        with torch.no_grad():
            pts = moge.infer(img)["points"].cpu().numpy()
        c = np.full((TOPK, 3), np.nan)
        e = np.full((TOPK, 3), np.nan)
        sc = np.zeros(TOPK)
        for slot, k in enumerate(np.argsort(-res["scores"].cpu().numpy())[:TOPK]):
            mask = res["masks"][k].cpu().numpy().astype(bool)
            p = pts[mask & np.isfinite(pts).all(-1)]
            if len(p) > 50:
                lo, hi = np.percentile(p, [5, 95], axis=0)
                c[slot], e[slot], sc[slot] = np.median(p, 0), hi - lo, float(res["scores"][k])
        frames.append(i)
        cents.append(c)
        exts.append(e)
        scores.append(sc)
        depths.append(
            cv2.resize(pts[..., 2], (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float16)
        )
    i += 1
np.savez_compressed(
    clip / "objects_sam3.npz",
    frame_idx=np.array(frames),
    score=np.stack(scores),  # (N, K) instances sorted by SAM score
    centroid_cam=np.stack(cents),  # (N, K, 3)
    extent_cam=np.stack(exts),  # (N, K, 3)
    depth_small=np.stack(depths),  # (N, H/8, W/8) MoGe z-depth for scale alignment with the hand estimator
)
print(f"frames={len(frames)} with_detection={np.mean([np.isfinite(c).any() for c in cents]):.2f}")
