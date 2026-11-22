"""Turntable review video of the stage-7 textured meshes (point-splat renderer)."""

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

from synthetic.video import write_mp4

PANEL = 320
N_FRAMES = 90
N_POINTS = 60000


def _splat(points, colors, yaw, size=PANEL):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    p = points @ rot.T
    img = np.full((size, size, 3), 245, np.uint8)
    u = (p[:, 0] * 0.42 + 0.5) * size
    v = (0.5 - p[:, 1] * 0.42) * size
    order = np.argsort(-p[:, 2])
    for dx in (0, 1):
        for dy in (0, 1):
            x = np.clip((u[order] + dx).astype(int), 0, size - 1)
            y = np.clip((v[order] + dy).astype(int), 0, size - 1)
            img[y, x] = colors[order]
    return img


def _base_color_image(path):
    # trimesh drops the texture when KHR_materials_specular is present, so read it directly.
    d = Path(path).read_bytes()
    jlen = struct.unpack("<I", d[12:16])[0]
    meta = json.loads(d[20 : 20 + jlen])
    bin_start = 20 + jlen + 8
    view = meta["bufferViews"][meta["images"][0]["bufferView"]]
    start = bin_start + view.get("byteOffset", 0)
    raw = np.frombuffer(d[start : start + view["byteLength"]], np.uint8)
    return cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _load(path):
    scene = trimesh.load(path, force="mesh")
    pts, face_idx = scene.sample(N_POINTS, return_index=True)
    tri = scene.triangles[face_idx]
    bary = trimesh.triangles.points_to_barycentric(tri, pts)
    uv = (scene.visual.uv[scene.faces[face_idx]] * bary[..., None]).sum(1)
    img = _base_color_image(path)
    h, w = img.shape[:2]
    x = np.clip((uv[:, 0] % 1.0 * (w - 1)).astype(int), 0, w - 1)
    y = np.clip(((1 - uv[:, 1] % 1.0) * (h - 1)).astype(int), 0, h - 1)
    col = img[y, x]
    pts = pts - (scene.bounds[0] + scene.bounds[1]) / 2
    pts = pts / (np.abs(pts).max() + 1e-9)
    return pts, np.asarray(col, np.uint8)


def main(scene_dir: str) -> None:
    root = Path(scene_dir)
    meshes = sorted((root / "s7_mesh/textured_mesh/hunyuan").glob("iter_*_mesh.glb"))
    loaded = [_load(m) for m in meshes]
    frames = []
    for i in range(N_FRAMES):
        yaw = 2 * np.pi * i / N_FRAMES
        frames.append(np.concatenate([_splat(p, c, yaw) for p, c in loaded], axis=1))
    out = root / "review/stage7_meshes.mp4"
    out.parent.mkdir(exist_ok=True)
    write_mp4(out, np.stack(frames), fps=30)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/scenes/egodex_104")
