"""Colour a generated (shape-only) object mesh from the RGBA crop it was generated from.

The generated mesh is in an arbitrary pose, so the camera pose of the crop is recovered by matching the
mesh silhouette to the crop alpha over the 24 axis-aligned views. Texels are then coloured by projecting
their 3D point into the crop (back faces reuse the mirrored front colour). Image-only; no ground truth.
"""

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

_SIL_RES = 160


@dataclass(frozen=True)
class MeshTexture:
    vertices: np.ndarray  # (V,3) xatlas-split vertices (same frame as the input mesh)
    faces: np.ndarray  # (F,3)
    uv: np.ndarray  # (V,2) in [0,1], v measured from the image top
    image: np.ndarray  # (H,W,3) uint8
    view_R: np.ndarray  # (3,3) mesh -> view rotation chosen by the silhouette match
    silhouette_iou: float


def _axis_rotations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sg) in enumerate(zip(perm, signs, strict=True)):
                m[row, col] = sg
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


def _fit_to_alpha(p_xy: np.ndarray, alpha: np.ndarray, res: int) -> tuple[float, np.ndarray]:
    """Uniform scale and offset mapping projected points onto the mask bbox."""
    ys, xs = np.nonzero(alpha)
    a_lo, a_hi = np.array([xs.min(), ys.min()], float), np.array([xs.max() + 1, ys.max() + 1], float)
    lo, hi = p_xy.min(axis=0), p_xy.max(axis=0)
    scale = float(np.min((a_hi - a_lo) / np.maximum(hi - lo, 1e-9)))
    offset = (a_lo + a_hi) / 2 - scale * (lo + hi) / 2
    return scale, offset


def _silhouette(p_xy: np.ndarray, scale: float, offset: np.ndarray, res: int) -> np.ndarray:
    pix = np.clip(np.round(p_xy * scale + offset).astype(int), 0, res - 1)
    sil = np.zeros((res, res), np.uint8)
    sil[pix[:, 1], pix[:, 0]] = 1
    return cv2.dilate(sil, np.ones((3, 3), np.uint8)).astype(bool)  # closes gaps between surface samples


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_view(points: np.ndarray, rgba: np.ndarray) -> tuple[np.ndarray, float]:
    """View (axis-aligned rotation plus an in-plane angle) whose silhouette best matches the crop alpha (IoU).

    The generator returns the object upright and axis-aligned, while the crop shows it at any in-plane angle.
    """
    alpha = cv2.resize(rgba[..., 3], (_SIL_RES, _SIL_RES), interpolation=cv2.INTER_AREA) > 127

    def score(rot: np.ndarray) -> float:
        p = points @ rot.T
        scale, offset = _fit_to_alpha(p[:, :2], alpha, _SIL_RES)
        sil = _silhouette(p[:, :2], scale, offset, _SIL_RES)
        return float((sil & alpha).sum() / max((sil | alpha).sum(), 1))

    best = (np.eye(3), -1.0, 0.0, np.eye(3))
    for base in _axis_rotations():
        for deg in range(0, 360, 10):
            iou = score(_rot_z(deg) @ base)
            if iou > best[1]:
                best = (_rot_z(deg) @ base, iou, deg, base)
    for deg in np.arange(best[2] - 8, best[2] + 9, 2.0):
        iou = score(_rot_z(deg) @ best[3])
        if iou > best[1]:
            best = (_rot_z(deg) @ best[3], iou, deg, best[3])
    return best[0], best[1]


def _filled_rgb(rgba: np.ndarray) -> np.ndarray:
    """RGB where the transparent border and a 2 px rim (hand fringe) take the nearest interior colour."""
    from scipy import ndimage

    inside = ndimage.binary_erosion(rgba[..., 3] > 127, iterations=2)
    if inside.sum() < 16:
        inside = rgba[..., 3] > 127
    idx = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return rgba[..., :3][idx[0], idx[1]]


def bake_texture(vertices: np.ndarray, faces: np.ndarray, rgba: np.ndarray, size: int = 512) -> MeshTexture:
    import xatlas

    vmap, new_faces, uv = xatlas.parametrize(vertices.astype(np.float32), faces.astype(np.uint32))
    verts = vertices[vmap]
    new_faces = new_faces.astype(np.int64)
    import trimesh

    rot, iou = fit_view(trimesh.Trimesh(verts, new_faces, process=False).sample(20000), rgba)
    side = rgba.shape[0]
    alpha = rgba[..., 3] > 127
    p = verts @ rot.T
    scale, offset = _fit_to_alpha(p[:, :2], alpha, side)
    rgb = _filled_rgb(rgba)

    tex = np.zeros((size, size, 3), np.uint8)
    filled = np.zeros((size, size), bool)
    px = uv * (size - 1)
    for tri in new_faces:
        a = px[tri]  # (3,2) texel coordinates (x, y from top)
        lo = np.floor(a.min(axis=0)).astype(int)
        hi = np.ceil(a.max(axis=0)).astype(int)
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        q = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
        d = (a[1, 1] - a[2, 1]) * (a[0, 0] - a[2, 0]) + (a[2, 0] - a[1, 0]) * (a[0, 1] - a[2, 1])
        if abs(d) < 1e-12:
            continue
        w0 = ((a[1, 1] - a[2, 1]) * (q[:, 0] - a[2, 0]) + (a[2, 0] - a[1, 0]) * (q[:, 1] - a[2, 1])) / d
        w1 = ((a[2, 1] - a[0, 1]) * (q[:, 0] - a[2, 0]) + (a[0, 0] - a[2, 0]) * (q[:, 1] - a[2, 1])) / d
        w = np.stack([w0, w1, 1 - w0 - w1], axis=1)
        ok = (w >= -0.05).all(axis=1) & (q[:, 0] < size) & (q[:, 1] < size) & (q[:, 0] >= 0) & (q[:, 1] >= 0)
        if not ok.any():
            continue
        pt = w[ok] @ p[tri]
        src = np.clip(np.round(pt[:, :2] * scale + offset).astype(int), 0, side - 1)
        qi = q[ok].astype(int)
        tex[qi[:, 1], qi[:, 0]] = rgb[src[:, 1], src[:, 0]]
        filled[qi[:, 1], qi[:, 0]] = True
    tex = cv2.inpaint(tex, (~filled).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return MeshTexture(verts, new_faces, uv, tex, rot, iou)
