"""Run WiLoR on an exported clip's video.mp4 and save per-frame hand estimates (FAH-compatible npz).

Fallback for ``estimate_hands_fah.py`` when FAH's gated SAM 3D Body config is not available. Run inside the
WiLoR venv (``WILOR_REPO`` = checkout with ``pretrained_models/{wilor_final.ckpt,detector.pt,model_config.yaml}``
and ``mano_data/{MANO_RIGHT.pkl,mano_mean_params.npz}``). The crop camera is converted to the full image with a
pinhole of ``--focal`` px (centre principal point), as in ``third_party/FAH/bench/methods/wilor.py``.
Output: ``<clip_dir>/hands_fah_f<focal>.npz`` (same keys as the FAH script) with joints3d (N,2,21,3)
[left,right] in the camera frame, joints2d, score, image_wh, valid.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# The official WiLoR checkpoints are full pickles (ultralytics / lightning objects); torch>=2.6 refuses them by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch  # noqa: E402


def load(repo: Path, device: str):
    sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    os.chdir(repo)  # WiLoR resolves ./mano_data and ./pretrained_models relative to the CWD
    try:
        from ultralytics import YOLO
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor

        model, cfg = load_wilor(
            checkpoint_path="./pretrained_models/wilor_final.ckpt",
            cfg_path="./pretrained_models/model_config.yaml",
        )
        detector = YOLO("./pretrained_models/detector.pt")
    finally:
        os.chdir(cwd)
    return model.to(device).eval(), detector.to(device), cfg, ViTDetDataset


@torch.no_grad()
def hands_in_frame(frame, K, model, detector, cfg, ds_cls, device, det_conf=0.3):
    det = detector(frame, conf=det_conf, iou=0.5, verbose=False)[0]
    if len(det.boxes) == 0:
        return []
    boxes = det.boxes.xyxy.cpu().numpy()
    right = det.boxes.cls.cpu().numpy()
    scores = det.boxes.conf.cpu().numpy()
    loader = torch.utils.data.DataLoader(
        ds_cls(cfg, frame, boxes, right, rescale_factor=2.0), batch_size=16, shuffle=False, num_workers=0
    )
    f, cx0, cy0 = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    out_hands = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        mult = 2 * batch["right"] - 1
        cam = out["pred_cam"].float().clone()
        cam[:, 1] = mult * cam[:, 1]
        bc, bsz = batch["box_center"].float(), batch["box_size"].float()
        bs = bsz * cam[:, 0] + 1e-9
        cam_t = (
            torch.stack(
                [cam[:, 1] + 2 * (bc[:, 0] - cx0) / bs, cam[:, 2] + 2 * (bc[:, 1] - cy0) / bs, 2 * f / bs],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        joints = out["pred_keypoints_3d"].float().cpu().numpy()
        m = mult.cpu().numpy()
        for n in range(joints.shape[0]):
            j = joints[n].copy()
            j[:, 0] *= m[n]
            j3 = j + cam_t[n]
            uv = j3[:, :2] / np.clip(j3[:, 2:], 1e-6, None) * f + np.array([cx0, cy0])
            i = len(out_hands)
            out_hands.append((bool(right[i] > 0.5), float(scores[i]), j3, uv))
    return out_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--focal", type=float, required=True, help="assumed focal in px (centre principal point)")
    ap.add_argument("--repo", type=Path, default=Path(os.environ.get("WILOR_REPO", str(Path(__file__).resolve().parents[1] / "third_party/WiLoR"))))
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, detector, cfg, ds_cls = load(a.repo, device)
    cap = cv2.VideoCapture(str(a.clip_dir / "video.mp4"))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    K = np.array([[a.focal, 0, w / 2], [0, a.focal, h / 2], [0, 0, 1.0]])
    J, J2, S, V = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and len(J) >= a.max_frames):
            break
        j = np.full((2, 21, 3), np.nan)
        j2 = np.full((2, 21, 2), np.nan)
        sc, v = np.zeros(2), np.zeros(2, bool)
        for is_right, score, j3, uv in hands_in_frame(frame, K, model, detector, cfg, ds_cls, device):
            side = 1 if is_right else 0
            if score > sc[side]:
                j[side], j2[side], sc[side], v[side] = j3, uv, score, True
        J.append(j), J2.append(j2), S.append(sc), V.append(v)
        if len(J) % 100 == 0:
            print("frame", len(J), "valid", np.stack(V).mean(0), flush=True)
    np.savez_compressed(
        a.clip_dir / f"hands_fah_f{round(a.focal)}.npz",
        joints3d=np.stack(J),
        joints2d=np.stack(J2),
        score=np.stack(S),
        image_wh=np.array([w, h]),
        valid=np.stack(V),
    )
    print("frames", len(J), "valid", np.stack(V).mean(0))


if __name__ == "__main__":
    main()
