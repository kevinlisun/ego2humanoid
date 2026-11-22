"""Procedural mug (open cylinder shell with a ring handle), used as the visual of a held mug.

Image-to-3D meshes of a small, hand-occluded mug come out fragmentary; the render only needs a clean
mug of the right size, so the shape is built from the perceived size instead.
"""

from __future__ import annotations

import numpy as np


def _revolve(profile: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Surface of revolution about z of a (K, 2) (r, z) polyline."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.concatenate([np.stack([r * np.cos(ang), r * np.sin(ang), np.full(n, z)], 1) for r, z in profile])
    f = []
    for k in range(len(profile) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            f += [(a, b, a + n), (b, b + n, a + n)]
    return v, np.array(f)


def _ring(centre: np.ndarray, major: float, minor: float, n: int = 24, m: int = 10):
    """Torus in the xz plane around ``centre``."""
    u, w = np.linspace(0, 2 * np.pi, n, endpoint=False), np.linspace(0, 2 * np.pi, m, endpoint=False)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    x = (major + minor * np.cos(ww)) * np.cos(uu)
    z = (major + minor * np.cos(ww)) * np.sin(uu)
    y = minor * np.sin(ww)
    v = np.stack([x, y, z], -1).reshape(-1, 3) + centre
    idx = np.arange(n * m).reshape(n, m)
    f = []
    for i in range(n):
        for j in range(m):
            a, b = idx[i, j], idx[(i + 1) % n, j]
            c, d = idx[(i + 1) % n, (j + 1) % m], idx[i, (j + 1) % m]
            f += [(a, b, c), (a, c, d)]
    return v, np.array(f)


def procedural_mug(radius_m: float = 0.037, height_m: float = 0.09, wall_m: float = 0.004):
    """Vertices (V, 3) centred on the body's bounding box and faces (F, 3); handle on +x."""
    R, H, t = radius_m, height_m, wall_m
    body_v, body_f = _revolve(np.array([[0, 0], [R, 0], [R, H], [R - t, H], [R - t, t], [0, t]]), 36)
    handle_v, handle_f = _ring(np.array([R + 0.004, 0.0, 0.55 * H]), 0.022, 0.004)
    v = np.concatenate([body_v, handle_v])
    f = np.concatenate([body_f, handle_f + len(body_v)])
    v = v - (v.max(0) + v.min(0)) / 2
    return v, f
