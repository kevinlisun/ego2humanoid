"""Per-frame hand-object contact flags from estimated landmarks and object geometry (sim frame)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

THUMB_TIP, INDEX_TIP = 4, 8
CONTACT_DIST_M = 0.02


def _min_dist(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """(N,3) points to (M,3) cloud -> (N,) nearest distance."""
    return np.linalg.norm(points[:, None, :] - cloud[None, :, :], axis=-1).min(axis=1)


def contact_flags(
    landmarks_sim: np.ndarray,
    object_points_sim: np.ndarray,
    threshold_m: float = CONTACT_DIST_M,
) -> dict[str, np.ndarray]:
    """Flags per frame for a static object point cloud (use one call per pose for moving objects).

    ``thumb``/``index``: that fingertip is within ``threshold_m`` of the object; ``pinch``: both are.
    ``landmarks_sim`` is (T,21,3); ``object_points_sim`` is (M,3) or (T,M,3).
    """
    lm = np.asarray(landmarks_sim, float)
    pts = np.asarray(object_points_sim, float)
    out = {}
    for name, idx in (("thumb", THUMB_TIP), ("index", INDEX_TIP)):
        if pts.ndim == 2:
            d = _min_dist(lm[:, idx], pts)
        else:
            d = np.array([_min_dist(lm[t, idx][None], pts[t])[0] for t in range(len(lm))])
        out[name] = d < threshold_m
    out["pinch"] = out["thumb"] & out["index"]
    return out


def write_alignment_bundle(
    out_dir: str | Path,
    T_sim_camera: np.ndarray,
    table_top_z_m: float,
    scale: float,
    flags: dict[str, np.ndarray] | None = None,
) -> Path:
    """``alignment/`` bundle: frame transform, table height, scale, optional contact flags."""
    root = Path(out_dir) / "alignment"
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "T_sim_camera": np.asarray(T_sim_camera, float).tolist(),
        "table_top_z_m": float(table_top_z_m),
        "scale": float(scale),
    }
    (root / "alignment.json").write_text(json.dumps(meta, indent=2))
    if flags:
        np.savez_compressed(root / "contact_flags.npz", **flags)
    return root
