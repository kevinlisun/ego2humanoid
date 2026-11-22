"""Background 3D Gaussian Splatting model of a reconstructed scene, in the sim frame.

The clip's camera barely moves (EgoDex 104: 2.2 cm of translation and 2.9 deg of rotation over the
whole take, and DA3 recovers the same), so there is no baseline to optimise a splat from many
views: multi-view training would only reproduce what one view plus monocular depth already says.
The background is therefore *built* rather than trained, from the stage-2 DA3 reconstruction:

1. every DA3 view is warped into the anchor view, and the temporal spread of the stack marks the
   human: the camera is static, so whatever changes colour is the moving arm;
2. those pixels, together with the pixels of the objects that SimFoundry removed (they are rebuilt
   as meshes in MuJoCo and must not appear twice), are replaced by the FLUX clean plate -- which
   repaints the scene, so it is first colour-corrected to the real frame with a spatially varying
   gain fitted on the pixels both images agree on;
3. their depth is replaced by a ray cast against the two planes fitted to the scene, the table and
   the backdrop, because that is what the removed things were standing in front of;
4. each surviving pixel becomes one flat Gaussian: centre at the back-projected point, disc normal
   along the local surface normal, radius equal to the pixel footprint at that depth.

The result is a genuine Gaussian splat -- it is rasterised by gsplat from the robot's own camera and
depth-composited with MuJoCo -- but it is a single-view-equivalent one. Novel views that are far
from the anchor open disocclusion holes; the Unitree head camera is 0.19 m away from it, which is
small enough for the holes to stay at object silhouettes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ANCHOR_RE = "image_*_cam2world.npy"


@dataclass(frozen=True)
class BackgroundSplat:
    """Flat Gaussians in the sim frame: ``means`` (N,3), ``quats`` wxyz (N,4), ``scales`` (N,3),
    ``opacities`` (N,), ``colors`` (N,3) in 0..1."""

    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            means=self.means.astype(np.float32),
            quats=self.quats.astype(np.float32),
            scales=self.scales.astype(np.float32),
            opacities=self.opacities.astype(np.float32),
            colors=self.colors.astype(np.float32),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> BackgroundSplat:
        d = np.load(path)
        return BackgroundSplat(**{k: d[k] for k in ("means", "quats", "scales", "opacities", "colors")})


def _anchor_index(scene_dir: Path) -> tuple[int, np.ndarray]:
    """Index of the DA3 view SimFoundry anchored the world frame to, and its cam2world."""
    matches = sorted((scene_dir / "s4_frame").glob(ANCHOR_RE))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {ANCHOR_RE} in {scene_dir / 's4_frame'}, got {matches}")
    return int(matches[0].name.split("_")[1]), np.load(matches[0])


def removal_mask(scene_dir: Path, hw: tuple[int, int], dilate_px: int = 6) -> np.ndarray:
    """Union of the stage-5 object-removal masks, resampled to ``hw`` (bool, True = removed)."""
    h, w = hw
    mask = np.zeros((h, w), bool)
    for path in sorted((scene_dir / "s5_scene/removal_mask").glob("*.png")):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        mask |= cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    if dilate_px:
        k = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    return mask


def match_colors(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    quantile: float = 50.0,
) -> np.ndarray:
    """Recolour ``source`` to ``target`` one surface at a time, by the ratio of each class.

    FLUX repaints the plate -- in this clip a beige table becomes pink while the backdrop stays
    white -- so a single global curve cannot undo it: the surfaces move in different directions.
    They are corrected separately instead, and with a median rather than a mean, because the target
    frame still contains the human and only a quantile survives that. The gain of a class is
    measured on its ``valid`` pixels but applied to all of them, so the areas hidden behind the
    human still get the colour of the surface they belong to. Before the human is known, ``valid``
    is everything and the human covers more than half of the table, so the median is not robust
    enough and a higher ``quantile`` of the target is used instead: the occluders here -- a black
    sleeve and shadowed skin -- are darker than the surfaces they hide.
    """
    out = source.astype(np.float32).copy()
    for label in np.unique(labels):
        fit = (labels == label) & valid
        if fit.sum() < 50:
            continue
        gain = np.percentile(target.astype(np.float32)[fit], quantile, axis=0) / np.maximum(
            np.percentile(source.astype(np.float32)[fit], quantile, axis=0), 1e-3
        )
        m = labels == label
        out[m] = source.astype(np.float32)[m] * np.clip(gain, 0.2, 5.0)
    return np.clip(out, 0, 255)


def surface_labels(plate: np.ndarray, n_surfaces: int = 4) -> np.ndarray:
    """Cluster the clean plate into its few flat surfaces (table, backdrop, side wall, frame).

    The plate is object- and human-free and each of its surfaces is close to uniform, so plain
    colour k-means segments it; the labels are what lets every surface be recoloured on its own.
    """
    data = plate.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(data, n_surfaces, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(plate.shape[:2])


def fill_smooth(values: np.ndarray, mask: np.ndarray, iters: int = 400) -> np.ndarray:
    """Harmonic (Laplace) extrapolation of ``values`` into ``mask``.

    The depth under the human has to come from its surroundings, and the surroundings are smooth
    surfaces, so the smoothest interpolant that agrees with them on the boundary is the right guess.
    Repeated box blurring of only the masked pixels converges to exactly that.
    """
    out = values.astype(np.float32).copy()
    known = ~mask
    if not known.any():
        return out
    out[mask] = float(np.median(values[known]))
    for _ in range(iters):
        blurred = cv2.blur(out, (5, 5))
        out[mask] = blurred[mask]
    return out


def _normals(points: np.ndarray) -> np.ndarray:
    """Unit normals of an organised point grid (H,W,3) from its cross-differences."""
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > 1e-9, n / np.maximum(norm, 1e-9), np.array([0.0, 0.0, 1.0]))


def _quats_from_normals(normals: np.ndarray) -> np.ndarray:
    """wxyz quaternions of frames whose third axis is ``normals`` (N,3)."""
    n = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    helper = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    x = np.cross(helper, n)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    y = np.cross(n, x)
    r = np.stack([x, y, n], axis=-1)
    # Shepperd's method, branchless enough for the trace-positive case plus a fallback.
    t = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    q = np.empty((len(r), 4))
    pos = t > 0
    s = np.sqrt(np.maximum(t[pos] + 1.0, 1e-12)) * 2
    q[pos] = np.stack(
        [
            0.25 * s,
            (r[pos, 2, 1] - r[pos, 1, 2]) / s,
            (r[pos, 0, 2] - r[pos, 2, 0]) / s,
            (r[pos, 1, 0] - r[pos, 0, 1]) / s,
        ],
        -1,
    )
    if (~pos).any():
        from scipy.spatial.transform import Rotation

        q[~pos] = Rotation.from_matrix(r[~pos]).as_quat()[:, [3, 0, 1, 2]]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def build_background_splat(
    scene_dir: str | Path,
    clip_dir: str | Path,
    upsample: int = 3,
    disc_thickness: float = 0.2,
    radius_scale: float = 0.75,
    foreground_thresh: float = 35.0,
) -> tuple[BackgroundSplat, dict]:
    """Build the background splat of ``scene_dir`` in the sim frame of ``clip_dir``."""
    scene, clip = Path(scene_dir), Path(clip_dir)
    da3 = dict(np.load(scene / "s2_da/da/exports/npz/results.npz"))
    anchor, cam2world = _anchor_index(scene)
    align = json.loads((clip / "alignment/alignment.json").read_text())
    t_sim_world = np.asarray(align["T_sim_camera"], float) @ np.linalg.inv(cam2world)
    t_sim_world[2, 3] = align["table_top_z_m"]  # world z=0 is the table; drop the depth-scale residue

    frame = da3["image"][anchor].astype(np.float32)
    h0, w0 = frame.shape[:2]
    depth0 = da3["depth"][anchor].astype(np.float32)
    t_sim_cam = t_sim_world @ cam2world

    # The human is in nearly every frame of this clip, so no temporal statistic can remove it: the
    # FLUX clean plate, which has neither the arms nor the objects, is the only background source.
    plate = cv2.cvtColor(cv2.imread(str(scene / "plate.png")), cv2.COLOR_BGR2RGB)
    plate = cv2.resize(plate, (w0, h0), interpolation=cv2.INTER_AREA)
    labels = surface_labels(plate)
    color = match_colors(plate, frame, labels, np.ones((h0, w0), bool), quantile=75.0)

    # What the recoloured plate cannot explain is exactly what it removed: the human and the
    # objects. Their depth belongs to whatever stood behind them, so it is diffused in from around.
    residual = np.abs(color - frame).max(-1)
    foreground = cv2.dilate(
        (residual > foreground_thresh).astype(np.uint8), np.ones((9, 9), np.uint8)
    ).astype(bool)
    foreground |= removal_mask(scene, (h0, w0)) | (depth0 <= 1e-3)
    # Refit now that the human is known: its dark sleeve was dragging the first fit's medians down.
    color = match_colors(plate, frame, labels, ~foreground)
    depth = fill_smooth(depth0, foreground)

    h, w = h0 * upsample, w0 * upsample
    color = cv2.resize(color.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    k = da3["intrinsics"][anchor].astype(np.float64).copy()
    k[:2] *= upsample

    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    cam = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1) * depth[..., None]
    t_sim_cam = t_sim_world @ cam2world
    pts = cam @ t_sim_cam[:3, :3].T + t_sim_cam[:3, 3]
    normals = _normals(pts)

    keep = (depth > 1e-3) & np.isfinite(pts).all(-1)
    means = pts[keep]
    radius = depth[keep] / k[0, 0] * radius_scale
    scales = np.stack([radius, radius, radius * disc_thickness], -1)
    splat = BackgroundSplat(
        means=means,
        quats=_quats_from_normals(normals[keep]),
        scales=scales,
        opacities=np.ones(len(means)),
        colors=color[keep].astype(np.float32) / 255.0,
    )
    report = {
        "anchor_view": anchor,
        "n_gaussians": int(len(means)),
        "source_hw": [int(h0), int(w0)],
        "splat_hw": [int(h), int(w)],
        "foreground_fraction": float(foreground.mean()),
        "T_sim_anchor_camera": t_sim_cam.tolist(),
        "anchor_fovy_deg": float(np.degrees(2 * np.arctan(k[1, 2] / k[1, 1]))),
        "mean_radius_mm": float(radius.mean() * 1000),
    }
    return splat, report
