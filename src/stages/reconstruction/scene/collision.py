"""Convex decomposition (CoACD) of reconstructed meshes for faithful MuJoCo collision."""

from __future__ import annotations

import coacd
import numpy as np
import trimesh


def convex_pieces(
    vertices: np.ndarray, faces: np.ndarray, threshold: float = 0.05, max_pieces: int = 16, seed: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Approximate convex decomposition; falls back to the single convex hull for non-watertight input."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not mesh.is_watertight:
        hull = mesh.convex_hull
        return ((np.asarray(hull.vertices), np.asarray(hull.faces)),)
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold,
        max_convex_hull=max_pieces,
        seed=seed,
    )
    return tuple((np.asarray(v), np.asarray(f)) for v, f in parts)
