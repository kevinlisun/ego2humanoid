"""Run EgoTac on an exported clip's video.mp4 and save per-frame tactile force and contact.

Run inside the EgoTac venv (torch, timm, hydra, scipy, ...) from any directory. Pure RGB, no ground truth.
Output: ``<clip_dir>/tactile_egotac.npz`` with ``force (N,2,778)`` and ``contact_prob (N,2,778)``
(hand order [left, right], matching the FAH hands bundle), plus ``frame_idx`` and ``source``.
Usage: estimate_tactile_egotac.py <clip_dir> [--checkpoint PATH] [--max-frames N] [--stride S]
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ECO = ROOT / "third_party" / "EgoTac"
DEFAULT_CKPT = ROOT / "data" / "weights" / "egotac" / "egotac_base_h4.pt"
N_VERTS = 778


def split_hands(rows: np.ndarray, n_verts: int = N_VERTS) -> np.ndarray:
    """(N, 2*V) or (N, V, 2) or (N, 2, V) prediction rows -> (N, 2, V) in [left, right] order."""
    rows = np.asarray(rows)
    n = rows.shape[0]
    flat = rows.reshape(n, -1)
    if flat.shape[1] != 2 * n_verts:
        raise ValueError(f"expected {2 * n_verts} values per frame, got {flat.shape[1]}")
    if rows.ndim == 3 and rows.shape[1:] == (n_verts, 2):
        return np.moveaxis(rows, -1, 1)
    return flat.reshape(n, 2, n_verts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="run every S-th frame; others copy the last result")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(ECO))
    import cv2
    import torch

    sys.argv = [sys.argv[0]]  # infer.py parses argv at import time only inside main(); keep it clean
    from infer import load_model  # noqa: E402

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.checkpoint, device)
    keys = list(model.obs_encoder.rgb_keys)
    shape = cfg.shape_meta.obs[keys[0]].shape
    height, width = int(shape[1]), int(shape[2])
    horizon = int(cfg.shape_meta.obs[keys[0]].horizon)

    cap = cv2.VideoCapture(str(a.clip_dir / "video.mp4"))
    window: deque = deque(maxlen=horizon)
    force, contact, frame_idx = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and i >= a.max_frames):
            break
        rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        chw = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
        if not window:
            window.extend([chw] * (horizon - 1))
        window.append(chw)
        if i % a.stride == 0 or not force:
            tensor = torch.from_numpy(np.stack(window)[None]).to(device)
            with torch.inference_mode():
                pred = model.predict_tactile({k: tensor for k in keys})
            f = pred["tactile_pred"][0, -1].detach().cpu().numpy()
            c = torch.sigmoid(pred["contact_logits"][0, -1]).detach().cpu().numpy()
        force.append(f)
        contact.append(c)
        frame_idx.append(i)
        i += 1
    if not force:
        raise SystemExit(f"no frames decoded from {a.clip_dir / 'video.mp4'}")
    out = a.clip_dir / "tactile_egotac.npz"
    np.savez_compressed(
        out,
        force=split_hands(np.stack(force)).astype(np.float32),
        contact_prob=split_hands(np.stack(contact)).astype(np.float32),
        frame_idx=np.array(frame_idx),
        source="egotac_base_h4",
    )
    print(f"{out}: {len(force)} frames, raw force row {np.stack(force).shape[1:]}")


if __name__ == "__main__":
    main()
