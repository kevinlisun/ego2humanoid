"""Cut the held object out of an RGB frame (SAM 3 mask) as a background-free RGBA image for 3D generation.

Run inside the FAH venv. Usage: crop_object_sam3.py <clip_dir> "<text prompt>"
Reads ``retarget_rgb.json`` or ``retarget_rgb_sharpa.json`` (``perceived_object.views``) and writes
``<clip_dir>/object_crop.png``. Only the video frame is used; no ground truth.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from stages.reconstruction.objects.crop_select import select_crop_mask
from stages.reconstruction.objects.hf_weights import require_model

clip = Path(sys.argv[1])
prompt = sys.argv[2]
report = next(p for p in (clip / "retarget_rgb.json", clip / "retarget_rgb_sharpa.json") if p.exists())
info = json.loads(report.read_text())["perceived_object"]
require_model("facebook/sam3", "SAM3")
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
cap = cv2.VideoCapture(str(clip / "video.mp4"))
views = info["views"]
views = [views[i] for i in np.unique(np.linspace(0, len(views) - 1, min(len(views), 10)).astype(int))]
cands = []  # (rgb, mask) per view; the cleanest whole mask is chosen below
for frame_idx, slot in views:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    inputs = proc(images=Image.fromarray(rgb), text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = sam(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.1, target_sizes=[(h, w)])[0]
    order = np.argsort(-res["scores"].cpu().numpy())
    if slot < len(order):
        cands.append((rgb, res["masks"][order[slot]].cpu().numpy().astype(bool)))
pick = select_crop_mask([m for _, m in cands])
assert pick is not None, "object not found in any whole (unfragmented) view"
rgb, mask = cands[pick]
print(f"crop_source=video_view {pick} of {len(cands)}")
h, w = rgb.shape[:2]
ys, xs = np.where(mask)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
pad = int(0.15 * max(y1 - y0, x1 - x0))
y0, y1, x0, x1 = max(y0 - pad, 0), min(y1 + pad, h), max(x0 - pad, 0), min(x1 + pad, w)
rgba = np.dstack([rgb, mask.astype(np.uint8) * 255])[y0:y1, x0:x1]
side = max(rgba.shape[:2])  # square canvas, transparent padding
canvas = np.zeros((side, side, 4), np.uint8)
oy, ox = (side - rgba.shape[0]) // 2, (side - rgba.shape[1]) // 2
canvas[oy : oy + rgba.shape[0], ox : ox + rgba.shape[1]] = rgba
Image.fromarray(canvas).resize((512, 512), Image.LANCZOS).save(clip / "object_crop.png")
print(f"mask_px={int(mask.sum())} crop_side={side}")
