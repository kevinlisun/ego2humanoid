"""Rasterise the background Gaussian splat from the robot's head camera (gsplat, CUDA).

Runs in the ``simfoundry`` conda environment, which is the one with torch and a CUDA toolchain; the
rest of the pipeline has neither. It is therefore a file-to-file step:

    ego_d435<suffix>.npz  (camera poses, written by EgoNeuralRecorder)
  + bg_splat.npz          (the background Gaussians, written by build_bg_splat.py)
  -> bg_rgbd<suffix>.npz  (RGB, expected depth and alpha at every frame)

Usage:
    CUDA_HOME=/mnt/nvme_hot/envs/cuda_home PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0 \
    <conda>/envs/simfoundry/bin/python scripts/render_bg_splat.py <ego_npz> <splat_npz> <out_npz>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

BATCH = 16
# MuJoCo cameras look along -z with +y up; gsplat expects OpenCV axes (+z forward, +y down).
MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def render(ego_npz: Path, splat_npz: Path, out_npz: Path) -> Path:
    ego = np.load(ego_npz)
    g = np.load(splat_npz)
    h, w = (int(x) for x in ego["size"])
    fovy = float(ego["fovy_deg"])
    fy = h / 2 / np.tan(np.radians(fovy) / 2)
    k = np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], np.float32)

    dev = "cuda"
    means = torch.as_tensor(g["means"], dtype=torch.float32, device=dev)
    quats = torch.as_tensor(g["quats"], dtype=torch.float32, device=dev)
    scales = torch.as_tensor(g["scales"], dtype=torch.float32, device=dev)
    opac = torch.as_tensor(g["opacities"], dtype=torch.float32, device=dev)
    colors = torch.as_tensor(g["colors"], dtype=torch.float32, device=dev)

    cam = np.asarray(ego["cam_T_world"], np.float64).copy()
    cam[:, :3, :3] = cam[:, :3, :3] @ MJ_TO_CV
    viewmats = np.linalg.inv(cam).astype(np.float32)

    rgb_out, depth_out, alpha_out = [], [], []
    for i in range(0, len(viewmats), BATCH):
        v = torch.as_tensor(viewmats[i : i + BATCH], device=dev)
        ks = torch.as_tensor(k, device=dev)[None].expand(len(v), 3, 3).contiguous()
        with torch.no_grad():
            img, alpha, _ = rasterization(
                means, quats, scales, opac, colors, v, ks, w, h, render_mode="RGB+ED"
            )
        rgb_out.append((img[..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
        depth_out.append(img[..., 3].cpu().numpy())
        alpha_out.append(alpha[..., 0].cpu().numpy())
        print(f"  frames {i}-{i + len(v) - 1}", flush=True)

    depth = np.concatenate(depth_out).astype(np.float32)
    alpha = np.concatenate(alpha_out).astype(np.float32)
    # The expected depth of a ray that hit nothing is meaningless; push it behind everything.
    depth = np.where(alpha > 1e-3, depth, 1e4)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, rgb=np.concatenate(rgb_out), depth=depth, alpha=alpha)
    return out_npz


if __name__ == "__main__":
    ego, splat, out = (Path(a) for a in sys.argv[1:4])
    print("wrote", render(ego, splat, out))
