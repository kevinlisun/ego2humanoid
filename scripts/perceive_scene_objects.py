"""RGB-only inventory of every object on the table: OWLv2 boxes -> SAM masks -> MoGe-2 points.

The held object and the receptacle are tracked by ``perceive_objects_*.py``; this script lists the
whole tabletop so the simulation can reproduce the scene, not just the two objects the task names.
Run inside the perception venv (transformers >= 5, moge). Usage:

  perceive_scene_objects.py <clip_dir> [n_keyframes] [span_s]

Keyframes are taken from the first ``span_s`` seconds, before the hand reaches the object, and an
object must be seen in at least half of them (a single-frame detection is usually a hand or a
shadow). Each object keeps its least occluded observation. Writes ``<clip>/scene/``:
``inventory.json`` (names, scores, frames, crop files), ``points.npz`` (camera-frame MoGe points per
object, unscaled, OpenCV axes), ``crop_<i>.png`` (RGBA, the mesh-generation input) and
``inventory.jpg`` (masks over the keyframe, for review). No ground truth is used.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stages.reconstruction.objects.prompt import object_prompt, place_target_prompt, source_prompt

OWL, SAM, MOGE = "google/owlv2-base-patch16-ensemble", "facebook/sam-vit-huge", "Ruicheng/moge-2-vitl-normal"
# Generic tabletop vocabulary; the task's own phrases are added in front. Hand/arm queries are only
# used to reject detections on the person.
VOCAB = [
    "bowl",
    "plate",
    "cup",
    "mug",
    "bottle",
    "jar",
    "can",
    "box",
    "box lid",
    "tray",
    "basket",
    "container",
    "tupperware",
    "scoop",
    "spoon",
    "fork",
    "knife",
    "tongs",
    "spatula",
    "pen",
    "marker",
    "pencil",
    "scissors",
    "pliers",
    "screwdriver",
    "hammer",
    "stapler",
    "tape",
    "phone",
    "remote",
    "card",
    "book",
    "toy",
    "plush toy",
    "rubber duck",
    "ball",
    "block",
    "cube",
    "rubik's cube",
    "dice",
    "lego brick",
    "sponge",
    "towel",
    "cloth",
    "bag",
    "glasses",
    "watch",
    "key",
    "clip",
    "peg",
    "lid",
    "case",
    "stand",
    "small table",
    "figurine",
    "fruit",
    "orange",
    "apple",
    "banana",
    "brush",
    "comb",
    "toothbrush",
    "bottle opener",
    "fidget toy",
    "pop it toy",
    "package",
    "wallet",
    "coaster",
    "paper",
]
PERSON = ["human hand", "arm", "sleeve", "finger", "sweater", "jacket"]
DET_THR = 0.12
NMS_IOU = 0.5
MASK_DUP_IOU = 0.6
MIN_POINTS = 80
MIN_HEIGHT_M, MAX_HEIGHT_M = 0.003, 0.45  # object top above the table plane
MAX_FOOTPRINT_M = 0.6
MAX_BOTTOM_M = 0.05  # lowest point above the table plane; contents of a tray or lid still pass
HAND_OVERLAP = 0.4
TASK_BONUS = 0.1
PART_SCORE = 0.6
CONFIRM_IOU = 0.3
MIN_SOLIDITY = 0.25  # mask area / box area: thin outlines are shadows and edges, not objects


def keyframes(video: Path, n: int, span_s: float) -> tuple[list[int], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sorted({int(round(x)) for x in np.linspace(0, min(span_s * fps, total - 1), n)})
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return idx, frames


def table_basis(up: np.ndarray) -> np.ndarray:
    """Rows e1, e2, up: a right-handed frame whose third axis is gravity-up (camera x projected first)."""
    e1 = np.array([1.0, 0.0, 0.0]) - up * up[0]
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])


def table_plane(pts: np.ndarray, normals: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    """(normal, offset) of the table, ``normal @ p = offset``: the largest near-horizontal surface.

    Seeded by the height histogram of upward-facing points in the lower image, then refined by an
    iterated least-squares plane fit: the gravity estimate is a few degrees off and MoGe bends the far
    field, so a plane through ``up`` alone puts thin objects (pliers) below the table.
    """
    h, _ = pts.shape[:2]
    p, nrm = pts[h // 3 :].reshape(-1, 3), normals[h // 3 :].reshape(-1, 3)
    ok = np.isfinite(p).all(1) & np.isfinite(nrm).all(1)
    flat = p[ok & (np.abs(nrm @ up) > 0.95)]
    hv = flat @ up
    hist, edges = np.histogram(hv, bins=400)
    k = int(np.argmax(hist))
    n, d = up.copy(), float(0.5 * (edges[k] + edges[k + 1]))
    for tol in (0.03, 0.015, 0.008):
        sel = flat[np.abs(flat @ n - d) < tol]
        c = sel.mean(0)
        n_new = np.linalg.svd(sel - c, full_matrices=False)[2][-1]
        n = n_new if n_new @ up > 0 else -n_new
        d = float(n @ c)
    return n, d


def detect(owl_proc, owl, pil, queries):
    inputs = owl_proc(images=pil, text=[queries], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = owl(**inputs)
    w, h = pil.size
    s = max(h, w)  # OWLv2 pads to a square at the bottom/right
    det = owl_proc.post_process_grounded_object_detection(out, threshold=DET_THR, target_sizes=[(s, s)])[0]
    boxes, conf, lab = det["boxes"].float().cpu(), det["scores"].float().cpu(), det["labels"].cpu()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    return boxes.numpy(), conf.numpy(), lab.numpy()


def segment(sam_proc, sam, pil, boxes):
    if len(boxes) == 0:
        return np.zeros((0, pil.size[1], pil.size[0]), bool)
    si = sam_proc(images=pil, input_boxes=[boxes.tolist()], return_tensors="pt").to("cuda")
    with torch.no_grad():
        so = sam(**si, multimask_output=False)
    return (
        sam_proc.post_process_masks(
            so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu()
        )[0][:, 0]
        .numpy()
        .astype(bool)
    )


def box_nms(boxes, conf):
    from torchvision.ops import nms

    if len(boxes) == 0:
        return np.zeros(0, int)
    return nms(
        torch.tensor(boxes, dtype=torch.float32), torch.tensor(conf, dtype=torch.float32), NMS_IOU
    ).numpy()


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def frame_detections(pil, queries, n_task, models):
    """Boxes of non-person detections plus the union mask of the person (hands, arms)."""
    owl_proc, owl, sam_proc, sam = models
    boxes, conf, lab = detect(owl_proc, owl, pil, queries + PERSON)
    # Task phrases are what the simulation needs by name; boost them over the generic vocabulary.
    conf = conf + TASK_BONUS * (lab < n_task)
    keep = box_nms(boxes, conf)
    boxes, conf, lab = boxes[keep], conf[keep], lab[keep]
    person = lab >= len(queries)
    hand = np.zeros(pil.size[::-1], bool)
    for m in segment(sam_proc, sam, pil, boxes[person]):
        hand |= m
    dets = [
        dict(box=boxes[k], label=queries[int(lab[k])], score=float(conf[k]), task=bool(lab[k] < n_task))
        for k in np.flatnonzero(~person)
    ]
    return dets, hand


def homography(src: np.ndarray, dst: np.ndarray, mask_src: np.ndarray, mask_dst: np.ndarray) -> np.ndarray:
    """src -> dst image homography from ORB matches off the person; identity if it cannot be fitted."""
    orb = cv2.ORB_create(4000)
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    k1, d1 = orb.detectAndCompute(g1, (~mask_src).astype(np.uint8) * 255)
    k2, d2 = orb.detectAndCompute(g2, (~mask_dst).astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return np.eye(3)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 20:
        return np.eye(3)
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
    return H if H is not None and inl.sum() >= 15 else np.eye(3)


def warp_box(box, H):
    x0, y0, x1, y1 = box
    c = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), H)[0]
    return np.array([*c.min(0), *c.max(0)])


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def object_geometry(m, box, hand, pts, basis, th, named=False):
    """Filtered mask and its MoGe points, or None if it is not a plausible object resting on the table.

    ``named``: the task names the object. A thin one (a pen, 0.8 cm wide) has a low mask/box ratio and
    loses its core to the erosion, so the solidity test is skipped and the erosion is lighter.
    """
    area = int(m.sum())
    if area < MIN_POINTS or (m & hand).sum() > HAND_OVERLAP * area:
        return None
    x0, y0, x1, y1 = box
    if not named and area < MIN_SOLIDITY * max((x1 - x0) * (y1 - y0), 1.0):
        return None
    n_cc, cc = cv2.connectedComponents(m.astype(np.uint8))
    if n_cc > 2:  # keep the largest piece; SAM leaks into look-alike patches elsewhere
        sizes = np.bincount(cc.ravel())[1:]
        m2 = cc == 1 + int(np.argmax(sizes))
        if m2.sum() < 0.7 * area:
            return None
        m = m2
    k = 1 if named else 5
    core = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~hand
    p = pts[core & np.isfinite(pts).all(-1)]
    if len(p) < MIN_POINTS:
        return None
    q = p @ basis.T  # (e1, e2, table normal) coordinates
    height = np.percentile(q[:, 2], 95) - th
    lo, hi = np.percentile(q[:, :2], [3, 97], axis=0)
    if not (MIN_HEIGHT_M <= height <= MAX_HEIGHT_M) or (hi - lo).max() > MAX_FOOTPRINT_M:
        return None
    if np.percentile(q[:, 2], 5) - th > MAX_BOTTOM_M:
        return None  # floating: a wall socket or shelf item, not resting on the table
    return m, p.astype(np.float32)


def crop_rgba(rgb, mask, pad=12):
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    out[..., :3] = rgb[y0:y1, x0:x1]
    out[..., 3] = mask[y0:y1, x0:x1] * 255
    side = max(out.shape[:2])  # square, centred: the mesh generator expects a centred object
    sq = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - out.shape[0]) // 2, (side - out.shape[1]) // 2
    sq[oy : oy + out.shape[0], ox : ox + out.shape[1]] = out
    return cv2.resize(sq, (512, 512), interpolation=cv2.INTER_AREA)


def main():
    clip = Path(sys.argv[1])
    n_key = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    span_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    task = json.loads((clip / "episode.json").read_text())["task_description"]
    phrases = [object_prompt(task, default=""), place_target_prompt(task) or "", source_prompt(task) or ""]
    task_q = [p for p in dict.fromkeys(phrases) if p]
    queries = task_q + [v for v in VOCAB if v not in task_q]
    task_nouns = {p.split()[-1] for p in task_q}  # "blue pen" also names the generic "pen" detections
    up = np.asarray(json.loads((clip / "gravity_est.json").read_text())["up_in_camera"], float)
    up /= np.linalg.norm(up)
    models = (
        Owlv2Processor.from_pretrained(OWL),
        Owlv2ForObjectDetection.from_pretrained(OWL).cuda().eval(),
        SamProcessor.from_pretrained(SAM),
        SamModel.from_pretrained(SAM).cuda().eval(),
    )
    moge = MoGeModel.from_pretrained(MOGE).cuda().eval()
    idx, frames = keyframes(clip / "video.mp4", n_key, span_s)
    per_frame = []
    for fi, rgb in zip(idx, frames, strict=True):
        dets, hand = frame_detections(Image.fromarray(rgb), queries, len(task_q), models)
        per_frame.append(dict(frame=fi, rgb=rgb, dets=dets, hand=hand))
        print(f"frame {fi}: {len(dets)} detections, hand px {int(hand.sum())}")
    # One reference frame carries all the geometry: the pipeline maps the scene with a single
    # camera-to-sim transform, so mixing frames of a moving head camera would smear the layout. The
    # other keyframes are registered to it by a homography (the tabletop is near-planar and the head
    # mostly rotates) and only vote on which detections are persistent objects.
    ref = min(per_frame, key=lambda d: (int(d["hand"].sum()), d["frame"]))
    clusters: list[dict] = []
    for d in per_frame:
        H = np.eye(3) if d is ref else homography(d["rgb"], ref["rgb"], d["hand"], ref["hand"])
        for det in sorted(d["dets"], key=lambda x: -x["score"]):
            bx = warp_box(det["box"], H)
            best = max(clusters, key=lambda c: box_iou(c["box"], bx), default=None)
            if (
                best is not None
                and box_iou(best["box"], bx) > CONFIRM_IOU
                and d["frame"] not in best["frames"]
            ):
                best["frames"].add(d["frame"])
                best["labels"].append((det["label"], det["score"], det["task"]))
                if d is ref or (ref["frame"] not in best["frames"] and det["score"] > best["score"]):
                    best["box"], best["score"] = bx, det["score"]
            else:
                clusters.append(
                    dict(
                        box=bx,
                        score=det["score"],
                        frames={d["frame"]},
                        labels=[(det["label"], det["score"], det["task"])],
                    )
                )
    need = max(1, (len(per_frame) + 1) // 2)
    clusters = [c for c in clusters if len(c["frames"]) >= need]
    rgb = ref["rgb"]
    img = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
    with torch.no_grad():
        g = moge.infer(img)
    pts, normals = g["points"].cpu().numpy(), g["normal"].cpu().numpy()
    up, th = table_plane(pts, normals, up)
    basis = table_basis(up)
    h_img, w_img = rgb.shape[:2]
    boxes = np.array([np.clip(c["box"], 0, [w_img - 1, h_img - 1, w_img - 1, h_img - 1]) for c in clusters])
    masks = segment(models[2], models[3], Image.fromarray(rgb), boxes.reshape(-1, 4))
    kept, taken = [], []
    for c, bx, m in sorted(zip(clusters, boxes, masks, strict=True), key=lambda x: -x[0]["score"]):
        named = any(x[2] or x[0] in task_nouns for x in c["labels"])
        geo = object_geometry(m, bx, ref["hand"], pts, basis, th, named)
        print(
            "  cluster",
            c["labels"][0][0],
            round(c["score"], 2),
            sorted(c["frames"]),
            "kept" if geo else "rejected",
        )
        if geo is None or any(mask_iou(geo[0], t) > MASK_DUP_IOU for t in taken):
            continue
        taken.append(geo[0])
        kept.append(
            dict(
                box=bx,
                mask=geo[0],
                points=geo[1],
                area=int(geo[0].sum()),
                labels=c["labels"],
                score=c["score"],
            )
        )
    # A weak detection mostly inside a stronger one is a part of it (one cell row of a pop-it toy); a
    # confident one is content (the hammer in the box lid) and is cut out of the container's mask.
    parts = set()
    for a_ in kept:
        for j, b_ in enumerate(kept):
            inside = (a_["mask"] & b_["mask"]).sum() > 0.7 * b_["area"]
            if a_ is not b_ and a_["area"] > b_["area"] and inside and b_["score"] < PART_SCORE * a_["score"]:
                parts.add(j)
    kept = [o for j, o in enumerate(kept) if j not in parts]
    for a_ in kept:
        for b_ in kept:
            if (
                a_ is not b_
                and a_["area"] > b_["area"]
                and (a_["mask"] & b_["mask"]).sum() > 0.8 * b_["area"]
            ):
                a_["mask"] = a_["mask"] & ~b_["mask"]
    kept = [(o, o["labels"]) for o in kept]
    out_dir = clip / "scene"
    out_dir.mkdir(exist_ok=True)
    for f in [*out_dir.glob("crop_*.png"), *out_dir.glob("mesh_*.glb")]:
        f.unlink()
    entries, arrays, used = [], {}, {}
    kept.sort(key=lambda x: -x[0]["area"])
    for i, (o, labels) in enumerate(kept):
        scores: dict[str, float] = {}
        for label, sc, is_task in labels:
            scores[label] = scores.get(label, 0.0) + sc + (0.5 if is_task else 0.0)
        label = max(scores, key=scores.get)
        used[label] = used.get(label, 0) + 1
        name = label.replace(" ", "_").replace("'", "") + ("" if used[label] == 1 else f"_{used[label]}")
        cv2.imwrite(
            str(out_dir / f"crop_{i}.png"), cv2.cvtColor(crop_rgba(rgb, o["mask"]), cv2.COLOR_RGBA2BGRA)
        )
        arrays[f"points_{i}"] = o["points"]
        entries.append(
            dict(
                id=i,
                name=name,
                label=label,
                task_phrase=any(x[2] for x in labels),
                score=float(np.mean([x[1] for x in labels])),
                seen=len(labels),
                box_xyxy=[round(float(v), 1) for v in o["box"]],
                rgb_median=[int(v) for v in np.median(rgb[o["mask"]], axis=0)],
                crop=f"crop_{i}.png",
                points=f"points_{i}",
            )
        )
    d = pts[..., 2]
    np.savez_compressed(
        out_dir / "points.npz",
        # Same layout as ``objects_sam3.npz`` depth_small, so the pipeline's depth scale can be
        # transferred to this frame by a depth ratio.
        depth_small=cv2.resize(
            np.nan_to_num(d, nan=0.0, posinf=0.0),
            (d.shape[1] // 8, d.shape[0] // 8),
            interpolation=cv2.INTER_NEAREST,
        ),
        **arrays,
    )
    inv = dict(
        task=task,
        task_phrases=task_q,
        frame=int(ref["frame"]),
        keyframes=idx,
        # The table plane in the reference camera frame (unscaled MoGe units): normal @ p = offset.
        table_normal_cam=up.tolist(),
        table_offset_cam=th,
        detector="owlv2-ensemble+sam-vit-huge+moge2",
        objects=entries,
    )
    (out_dir / "inventory.json").write_text(json.dumps(inv, indent=1))
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rng = np.random.default_rng(0)
    for e, (o, _) in zip(entries, kept, strict=True):
        col = rng.integers(60, 255, 3).tolist()
        m = o["mask"]
        vis[m] = (0.5 * vis[m] + 0.5 * np.array(col)).astype(np.uint8)
        x0, y0 = int(e["box_xyxy"][0]), int(e["box_xyxy"][1])
        cv2.putText(vis, f"{e['id']}:{e['name']}", (x0, max(y0 - 4, 12)), 0, 0.7, col, 2)
    cv2.imwrite(str(out_dir / "inventory.jpg"), vis)
    print(f"reference frame {ref['frame']}")
    print(json.dumps([(e["id"], e["name"], round(e["score"], 2), e["seen"]) for e in entries]))


if __name__ == "__main__":
    main()
