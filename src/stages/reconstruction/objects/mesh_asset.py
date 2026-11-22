"""Turn a generated (unit-scale, arbitrary-pose) object mesh into a table-resting sim asset."""

from dataclasses import dataclass

import numpy as np
import trimesh

MIN_HEIGHT_M = 0.015


def rest_transform(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_extent_m: float,
    pinch_axis_xy: np.ndarray | None = None,
    min_height_m: float = MIN_HEIGHT_M,
) -> np.ndarray:
    """4x4 similarity: rest the mesh on its most stable face, scale it, centre its bounding box (z up).

    The longest side is scaled to ``max_extent_m``; a mesh flatter than ``min_height_m`` is then stretched
    vertically to that height. If ``pinch_axis_xy`` is given, the mesh is yawed so its thinnest horizontal
    direction lies along that axis (the thumb-to-index direction), i.e. the pinch closes across the
    object's narrow side.
    """
    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, int), process=True)
    poses, probs = mesh.compute_stable_poses(n_samples=1, threshold=0.0)
    T = poses[int(np.argmax(probs))].copy()
    mesh.apply_transform(T)
    scale = max_extent_m / mesh.extents.max()
    S = np.diag([scale, scale, scale, 1.0])
    T = S @ T
    v = np.asarray(mesh.vertices) * scale
    if pinch_axis_xy is not None:
        xy = v[:, :2] - v[:, :2].mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov(xy.T))
        minor = vecs[:, 0]
        target = np.asarray(pinch_axis_xy, float)
        yaw = np.arctan2(target[1], target[0]) - np.arctan2(minor[1], minor[0])
        c, s = np.cos(yaw), np.sin(yaw)
        Y = np.eye(4)
        Y[:3, :3] = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]).T  # row-vector convention v @ M
        T = Y @ T
        v = v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    height = float(v[:, 2].max() - v[:, 2].min())
    if height < min_height_m:  # image-to-3D tends to flatten the unseen depth; keep the object graspable
        Z = np.diag([1.0, 1.0, min_height_m / max(height, 1e-6), 1.0])
        T = Z @ T
        v = v @ Z[:3, :3]
    C = np.eye(4)
    C[:3, 3] = -(v.min(axis=0) + v.max(axis=0)) / 2
    return C @ T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, float) @ T[:3, :3].T + T[:3, 3]


def prepare_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_extent_m: float,
    pinch_axis_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh vertices resting on the table in the sim frame (see ``rest_transform``); faces are unchanged."""
    T = rest_transform(vertices, faces, max_extent_m, pinch_axis_xy)
    return transform_points(T, vertices), np.asarray(faces)


@dataclass(frozen=True)
class MeshAsset:
    vertices: np.ndarray  # sim-frame, resting on z = 0 after the caller places the body
    faces: np.ndarray
    uv: np.ndarray | None
    texture: np.ndarray | None
    texture_iou: float | None


def build_mesh_asset(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_extent_m: float,
    pinch_axis_xy: np.ndarray | None,
    crop_rgba: np.ndarray | None,
) -> MeshAsset:
    """Table-resting, scaled mesh; coloured from the generation crop when one is given."""
    T = rest_transform(vertices, faces, max_extent_m, pinch_axis_xy)
    if crop_rgba is None:
        return MeshAsset(transform_points(T, vertices), np.asarray(faces), None, None, None)
    from stages.reconstruction.objects.mesh_texture import bake_texture

    tex = bake_texture(np.asarray(vertices), np.asarray(faces), crop_rgba)
    return MeshAsset(transform_points(T, tex.vertices), tex.faces, tex.uv, tex.image, tex.silhouette_iou)


def prism_hull(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex prism over the mesh footprint, spanning its full height: (vertices, faces).

    A reconstructed top surface that was stretched to a graspable height keeps its rounded edges, so
    its convex hull narrows towards the top (clip 1: 51 mm wide at the base, 28 mm 11 mm higher).
    Fingers closing on those slanted sides squeeze the object down instead of holding it. The real
    objects are prisms (phones, boxes), so this shape is what collides; the visual mesh is unchanged.
    """
    from scipy.spatial import ConvexHull

    v = np.asarray(vertices, float)
    ring = v[ConvexHull(v[:, :2]).vertices, :2]
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    pts = np.r_[np.c_[ring, np.full(len(ring), z0)], np.c_[ring, np.full(len(ring), z1)]]
    hull = trimesh.convex.convex_hull(pts)
    return np.asarray(hull.vertices), np.asarray(hull.faces)


def slab_prism(vertices: np.ndarray, axis_xy: np.ndarray, width_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Box with two parallel faces ``width_m`` apart across ``axis_xy``, spanning the footprint's
    length and the mesh height: (vertices, faces), centred on the footprint's bounding box.

    A thin footprint hull tapers by a few millimetres along its length (clip 116: 19 mm at one end,
    11 mm 3 cm further), so the two pinching pads squeeze a wedge and shoot the light object out
    towards its wide end. The pinched faces of an invented prism are made parallel instead.
    """
    v = np.asarray(vertices, float)
    u = np.asarray(axis_xy, float)[:2] / np.linalg.norm(axis_xy[:2])
    p = np.array([-u[1], u[0]])
    c = (v[:, :2].max(0) + v[:, :2].min(0)) / 2
    a_c = float(c @ u)
    along = v[:, :2] @ p
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    pts = [
        np.r_[(a_c + sa * width_m / 2) * u + b * p, z]
        for sa in (-1, 1)
        for b in (along.min(), along.max())
        for z in (z0, z1)
    ]
    hull = trimesh.convex.convex_hull(np.array(pts))
    return np.asarray(hull.vertices), np.asarray(hull.faces)


def footprint_chord_width(
    vertices: np.ndarray, axis_xy: np.ndarray, center_xy: np.ndarray | None = None, band_m: float = 0.01
) -> float:
    """Width of the footprint hull along ``axis_xy`` on lines through ``center_xy`` (the pinch point).

    The full extent along the pinch axis overstates the width under the fingertips of a curved or
    tapered object (clip 100: 61 mm extent, 50-53 mm at the centre), so a gap of extent minus squeeze
    barely touches it. Lines offset by up to ``band_m`` across the axis (the fingertip pad width) are
    included and the widest is returned. ``center_xy`` defaults to the bounding-box centre.
    """
    from scipy.spatial import ConvexHull

    xy = np.asarray(vertices, float)[:, :2]
    axis = np.asarray(axis_xy, float) / np.linalg.norm(axis_xy)
    perp = np.array([-axis[1], axis[0]])
    c = (xy.max(0) + xy.min(0)) / 2 if center_xy is None else np.asarray(center_xy, float)
    ring = xy[ConvexHull(xy).vertices] - c
    nxt = np.roll(ring, -1, axis=0)
    best = 0.0
    for s in np.linspace(-band_m, band_m, 9):
        a, b = ring @ perp - s, nxt @ perp - s
        hit = (a * b <= 0) & (a != b)
        w = (a[hit] / (a[hit] - b[hit]))[:, None]
        u = ((1 - w) * ring[hit] + w * nxt[hit]) @ axis
        if len(u) >= 2:
            best = max(best, float(u.max() - u.min()))
    return best if best > 0 else float(np.ptp(xy @ axis))


def _chord_normals(ring: np.ndarray, axis: np.ndarray) -> tuple[float, float] | None:
    """Width and wedge ``tan(alpha)`` of the footprint ``ring`` (CCW, centred) on its line along ``axis``.

    ``alpha`` is the larger angle between an end's outward edge normal and the pinch axis; the two
    fingertips press along the axis, so a non-zero ``alpha`` turns part of the squeeze into a push
    that slides the object out of the grip along the wedge.
    """
    perp = np.array([-axis[1], axis[0]])
    nxt = np.roll(ring, -1, axis=0)
    a, b = ring @ perp, nxt @ perp
    hit = np.flatnonzero((a * b <= 0) & (a != b))
    if len(hit) < 2:
        return None
    w = a[hit] / (a[hit] - b[hit])
    u = ((1 - w)[:, None] * ring[hit] + w[:, None] * nxt[hit]) @ axis
    lo, hi = hit[np.argmin(u)], hit[np.argmax(u)]
    tans = []
    for e, sign in ((lo, -1.0), (hi, 1.0)):
        d = nxt[e] - ring[e]
        normal = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-12)  # outward for a CCW ring
        cos = float(np.clip(sign * normal @ axis, 1e-6, 1.0))
        tans.append(np.sqrt(max(1.0 - cos * cos, 0.0)) / cos)
    return float(u.max() - u.min()), max(tans)


def antipodal_axis_xy(
    vertices: np.ndarray,
    axis_xy: np.ndarray,
    max_turn_deg: float = 45.0,
    max_width_m: float | None = None,
    step_deg: float = 2.0,
    turn_cost: float = 0.1,
) -> tuple[np.ndarray, float, float]:
    """Horizontal pinch direction through the footprint centre whose two ends are most parallel.

    A tapered or rounded footprint pinched across its narrowest line is gripped on two faces that
    lean towards each other (clip 114: normals 115-143 deg apart instead of 180), and the squeeze
    pushes the object out of the fingers. Directions within ``max_turn_deg`` of ``axis_xy`` are
    scored by the wedge ``tan(alpha)`` of their end faces plus ``turn_cost`` per radian of turn,
    among those no wider than ``max_width_m`` (the open hand). Returns (unit axis, tan(alpha), width).
    """
    from scipy.spatial import ConvexHull

    xy = np.asarray(vertices, float)[:, :2]
    hull = ConvexHull(xy)
    c = (xy.max(0) + xy.min(0)) / 2
    ring = xy[hull.vertices] - c  # scipy returns 2-D hull vertices counter-clockwise
    base = np.arctan2(axis_xy[1], axis_xy[0])
    best: tuple[float, np.ndarray, float, float] | None = None
    for turn in np.radians(np.arange(-max_turn_deg, max_turn_deg + 1e-9, step_deg)):
        axis = np.array([np.cos(base + turn), np.sin(base + turn)])
        got = _chord_normals(ring, axis)
        if got is None or (max_width_m is not None and got[0] > max_width_m and turn != 0.0):
            continue
        cost = got[1] + turn_cost * abs(turn)
        if best is None or cost < best[0]:
            best = (cost, axis, got[1], got[0])
    if best is None:
        a = np.asarray(axis_xy, float)
        return a / np.linalg.norm(a), 0.0, float(np.ptp(xy @ a) / np.linalg.norm(a))
    return best[1], best[2], best[3]
