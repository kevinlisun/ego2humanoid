"""RGB-only object perception with public weights: OWLv2 boxes -> SAM masks + MoGe-2 point map.

Fallback for ``perceive_objects_sam3.py`` when SAM 3 (``facebook/sam3``, gated on the Hub) is unavailable.
OWLv2 (``google/owlv2-base-patch16-ensemble``) finds text-prompted boxes, SAM (``facebook/sam-vit-base``)
turns each box into a mask, and the per-mask MoGe-2 statistics are computed exactly as in the SAM 3 script.
Run inside a venv with torch, transformers and moge. Usage:
  perceive_objects_owl.py <clip_dir> "<text prompt>" [stride] [threshold]
Writes ``<clip_dir>/objects_sam3.npz`` (the name the pipeline reads) with the same arrays as the SAM 3
script: frame_idx, score (N,K), centroid_cam (N,K,3), extent_cam (N,K,3), depth_small, plus a string
``detector`` naming the models. Scores are OWLv2 confidences, which run lower than SAM 3 scores, hence the
lower default threshold. No ground truth is used. The camera frame is OpenCV (x right, y down, z forward).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from PIL import Image
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

clip = Path(sys.argv[1])
prompt = sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

OWL, SAM, MOGE = "google/owlv2-base-patch16-ensemble", "facebook/sam-vit-base", "Ruicheng/moge-2-vitl-normal"
owl_proc = Owlv2Processor.from_pretrained(OWL)
owl = Owlv2ForObjectDetection.from_pretrained(OWL).cuda().eval()
sam_proc = SamProcessor.from_pretrained(SAM)
sam = SamModel.from_pretrained(SAM).cuda().eval()
moge = MoGeModel.from_pretrained(MOGE).cuda().eval()

TOPK = 5
NMS_IOU = 0.5  # OWLv2 emits overlapping boxes per object; SAM 3 returns de-duplicated instances
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
        pil = Image.fromarray(rgb)
        inputs = owl_proc(images=pil, text=[[prompt]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = owl(**inputs)
        # OWLv2 pads the image to a square at the bottom/right, so boxes are in the padded square's pixels.
        s = max(h, w)
        det = owl_proc.post_process_grounded_object_detection(out, threshold=thr, target_sizes=[(s, s)])[0]
        boxes, conf = det["boxes"].float(), det["scores"].float()
        keep = nms(boxes, conf, NMS_IOU)[:TOPK]
        boxes, conf = boxes[keep], conf[keep]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
        masks = np.zeros((0, h, w), bool)
        if len(boxes):
            si = sam_proc(images=pil, input_boxes=[boxes.cpu().tolist()], return_tensors="pt").to("cuda")
            with torch.no_grad():
                so = sam(**si, multimask_output=False)
            masks = sam_proc.post_process_masks(
                so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu()
            )[0][:, 0].numpy()
        img = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
        with torch.no_grad():
            pts = moge.infer(img)["points"].cpu().numpy()
        c = np.full((TOPK, 3), np.nan)
        e = np.full((TOPK, 3), np.nan)
        sc = np.zeros(TOPK)
        for slot, k in enumerate(np.argsort(-conf.cpu().numpy())[:TOPK]):
            p = pts[masks[k].astype(bool) & np.isfinite(pts).all(-1)]
            if len(p) > 50:
                lo, hi = np.percentile(p, [5, 95], axis=0)
                c[slot], e[slot], sc[slot] = np.median(p, 0), hi - lo, float(conf[k])
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
    score=np.stack(scores),  # (N, K) instances sorted by OWLv2 score
    centroid_cam=np.stack(cents),  # (N, K, 3)
    extent_cam=np.stack(exts),  # (N, K, 3)
    depth_small=np.stack(depths),  # (N, H/8, W/8) MoGe z-depth for scale alignment with the hand estimator
    detector=np.array("owlv2+sam-vit-base+moge2"),
)
print(f"frames={len(frames)} with_detection={np.mean([np.isfinite(c).any() for c in cents]):.2f}")
