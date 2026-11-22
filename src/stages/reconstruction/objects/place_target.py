"""Resolve the object a task places things *onto*, and where the held object should end up on it.

``select_held_object`` answers "which object moves with the hand". A pick-and-place needs the
complementary question: "which object does the hand have to be accurate about at release". For
``pick up the white block ... and place it on the blue bowl`` the robot's pinch point has to meet the
block at closure and the bowl at release, so the hand-eye correction switches reference frames
mid-trajectory (see ``docs/notes/target_frame_switching.md``).

The receptacle is resolved from the reconstructed scene when one is available -- ``objects.json``
already carries named, metric, sim-frame meshes -- because that needs no extra perception and is the
most accurate source. Clips without a reconstruction fall back to a box built from the detections.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from stages.manipulation.mjcf_scene import BoxObject

# Reconstruction names the colour it sees, the task description names the colour a person would say.
_COLOUR_SYNONYMS = {
    "blue": {"blue", "teal", "cyan", "navy", "turquoise", "aqua"},
    "red": {"red", "crimson", "maroon", "scarlet"},
    "green": {"green", "olive", "lime", "mint"},
    "grey": {"grey", "gray", "silver"},
    "beige": {"beige", "tan", "cream", "ivory", "off"},
    "orange": {"orange", "amber"},
    "purple": {"purple", "violet", "magenta"},
    "yellow": {"yellow", "gold"},
    "white": {"white", "ivory"},
    "black": {"black", "dark"},
}
_SYNONYM_OF = {w: head for head, words in _COLOUR_SYNONYMS.items() for w in words}
MIN_SHELL_M = 0.002  # thinnest floor or wall a synthesised tub may have
WALL_SEGMENTS = 16  # wedges in the tub wall
TUB_FLOOR_FRAC = 0.6  # inner radius of the tub at its floor, as a fraction of the half size
TUB_RIM_FRAC = 0.82  # ... and at its rim (both only without a mesh to measure them on)
TUB_BANDS = 3  # height bands of the tub wall, each following the mesh's inner surface
TUB_MAX_FRAC = 0.95  # the rim never thinner than this
CLEARANCE_M = 0.01  # the object is released just above its resting place, not interpenetrating it
# The tub's wall is pulled out by this much (trajectory.receptacle_opening_margin_m, set by the planner
# script): a hand-held block that lags its command catches the measured rim (clip 104).
OPENING_MARGIN_M = 0.0


def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().replace("_", " ").replace("-", " ").split() if t]


_NOUN_SYNONYM = {"mug": "cup"}  # detectors say "cup", people say "mug"


def _canonical(token: str) -> str:
    return _NOUN_SYNONYM.get(token, _SYNONYM_OF.get(token, token))


def match_object_name(names: list[str], phrase: str) -> str | None:
    """Name whose tokens best match ``phrase``; the head noun ("bowl") must match, colours loosely.

    Requiring the head noun keeps "blue bowl" from matching "blue sponge" when no bowl was
    reconstructed, which would silently place the object on the wrong receptacle.
    """
    want = [_canonical(t) for t in _tokens(phrase)]
    if not want:
        return None
    head = want[-1]
    best: tuple[int, str] | None = None
    for name in names:
        have = {_canonical(t) for t in _tokens(name)}
        if head not in have:
            continue
        score = sum(t in have for t in want)
        if best is None or score > best[0]:
            best = (score, name)
    return best[1] if best is not None else None


def _mesh_world(obj: BoxObject) -> np.ndarray | None:
    if obj.mesh_vertices is None:
        return None
    return np.asarray(obj.mesh_vertices, float) + np.asarray(obj.pos_m, float)


def container_profile(obj: BoxObject) -> tuple[bool, float]:
    """``(is_container, inner_floor_z)`` in world z, from the mesh alone.

    Take the topmost slab of vertices and look at the middle of its footprint. A bowl's rim is an
    annulus, so the centre of the top slab is empty and the object has to be released on the inner
    floor; a plate or a box has material there and the object rests on its lid. One rule therefore
    covers "on the plate", "in the bowl" and "in the drawer" without a per-object-class table.
    """
    v = _mesh_world(obj)
    pos = np.asarray(obj.pos_m, float)
    half = np.asarray(obj.half_size_m, float)
    top_z = float(pos[2] + half[2]) if v is None else float(v[:, 2].max())
    if v is None or len(v) < 32:
        return False, top_z
    height = float(v[:, 2].max() - v[:, 2].min())
    slab = v[v[:, 2] > v[:, 2].max() - 0.15 * max(height, 1e-6)]
    centre_xy = pos[:2]
    inner_mask = (np.abs(v[:, :2] - centre_xy) <= 0.35 * half[:2]).all(axis=1)
    slab_inner = (np.abs(slab[:, :2] - centre_xy) <= 0.35 * half[:2]).all(axis=1)
    if slab_inner.sum() < 0.02 * len(slab) and inner_mask.any():
        return True, float(np.percentile(v[inner_mask, 2], 75))
    return False, top_z


def drop_height(obj: BoxObject, held_half_z_m: float, clearance_m: float = CLEARANCE_M) -> float:
    """z at which the held object's *centre* comes to rest on (or in) ``obj``."""
    _, surface_z = container_profile(obj)
    return surface_z + held_half_z_m + clearance_m


def _box_piece(centre: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    verts = centre + signs * half
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 7, 5],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _hull_faces(verts: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(verts)
    faces = np.asarray(hull.simplices, dtype=np.int64)
    centre = verts.mean(axis=0)
    normal = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    inward = np.einsum("ij,ij->i", normal, verts[faces[:, 0]] - centre) < 0
    faces[inward] = faces[inward][:, ::-1]  # outward winding, as the mesh compiler expects
    return faces


def tub_profile(obj: BoxObject, floor_z_world: float) -> tuple[np.ndarray, np.ndarray]:
    """``(z, frac)``: world heights of the tub's wall rings and their inner radius per half size.

    The inner surface is measured on the mesh: in each band above the floor, the narrowest
    vertices (a low percentile of the radius normalised by the half size) are the inner wall. A
    fixed rim at 0.82 of the half size left a 1.6 cm flat collision rim inside a bowl whose visual
    rim is 0.91 (clip 116), and an object let go near the edge came to rest on it, 7 cm up.
    """
    pos = np.asarray(obj.pos_m, float)
    half = np.asarray(obj.half_size_m, float)
    v = _mesh_world(obj)
    bottom = float(pos[2] - half[2]) if v is None else float(v[:, 2].min())
    top = float(pos[2] + half[2]) if v is None else float(v[:, 2].max())
    floor_z = float(np.clip(floor_z_world, bottom, top))
    top = max(top, floor_z + MIN_SHELL_M)
    z = np.linspace(floor_z, top, TUB_BANDS + 1)
    frac = np.interp(z, [floor_z, top], [TUB_FLOOR_FRAC, TUB_RIM_FRAC])
    if v is not None and len(v) >= 32:
        r = np.sqrt((((v[:, :2] - pos[:2]) / np.maximum(half[:2], 1e-6)) ** 2).sum(1))
        band = (top - floor_z) / TUB_BANDS
        wall = v[:, 2] > floor_z + 0.25 * band  # the floor's own vertices are not the wall
        for k in range(len(z)):
            lo, hi = (z[k], z[k] + band) if k < TUB_BANDS else (top - 0.5 * band, top)
            m = wall & (v[:, 2] >= lo) & (v[:, 2] <= hi)
            if m.sum() >= 8:
                # Only ever wider than the default cone: a closed mesh (a scoop whose handle fills
                # the middle, forced hollow by the task, clip 10) has vertices all the way in.
                frac[k] = max(frac[k], float(np.percentile(r[m], 5)))
        frac = np.maximum.accumulate(frac)  # a bowl only widens upwards
    thinnest = 1.0 - MIN_SHELL_M / max(float(half[:2].min()), 1e-6)
    margin = OPENING_MARGIN_M / max(float(half[:2].min()), 1e-6)
    return z, np.clip(frac + margin, 0.2, min(TUB_MAX_FRAC, thinnest) + margin)


def container_pieces(obj: BoxObject, floor_z_world: float) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """A tub, in body coordinates, standing in for a hollow receptacle's collision.

    A reconstructed bowl is an open shell, so it is not watertight and CoACD degenerates to its
    convex hull -- a solid dome that bounces the carried object onto the floor instead of catching
    it (measured on clip 104: the block ended 0.82 m below the table). The visual mesh stays exactly
    as reconstructed; only the collision is replaced by a floor plus a sloped wall ring that follows
    the mesh's inner surface (``tub_profile``), which is what "the object ends up inside it" needs.
    """
    pos = np.asarray(obj.pos_m, float)
    half = np.asarray(obj.half_size_m, float)
    v = _mesh_world(obj)
    bottom = float(pos[2] - half[2]) if v is None else float(v[:, 2].min())
    z, frac = tub_profile(obj, floor_z_world)
    # The tub's floor and walls meet exactly at the inner floor, with a minimum thickness taken
    # out of the object rather than left as a gap a thin object could fall through.
    floor_z = float(z[0])
    base_bottom = min(bottom, floor_z - MIN_SHELL_M)
    base_half = np.array([half[0], half[1], (floor_z - base_bottom) / 2])
    base_c = np.array([0.0, 0.0, (floor_z + base_bottom) / 2 - pos[2]])
    pieces = [_box_piece(base_c, base_half)]
    # A real bowl narrows towards its floor, so each band of the wall is a ring of convex wedges
    # whose inner surface slopes between the measured radii of its two rings. Straight walls at the
    # rim radius left a pad-sized gap the fingers could not enter.
    outer = half[:2] + OPENING_MARGIN_M
    angles = np.linspace(0.0, 2 * np.pi, WALL_SEGMENTS + 1)
    for k in range(len(z) - 1):
        for a0, a1 in zip(angles[:-1], angles[1:], strict=True):
            pts = []
            for zz, f in ((z[k], frac[k]), (z[k + 1], frac[k + 1])):
                for ang in (a0, a1):
                    d = np.array([np.cos(ang), np.sin(ang)])
                    pts.append([*(f * half[:2] * d), zz - pos[2]])
                    pts.append([*(outer * d), zz - pos[2]])
            verts = np.asarray(pts, float)
            pieces.append((verts, _hull_faces(verts)))
    return tuple(pieces)


@dataclass
class PlaceTarget:
    """Where a carried object has to be let go, and the body that receives it."""

    name: str
    drop_point_m: np.ndarray  # sim-frame centre of the held object once placed
    half_size_m: np.ndarray
    obj: BoxObject | None  # the receptacle body to add to the scene, static


def resolve_from_scene(
    scene_objects: tuple[BoxObject, ...], phrase: str | None, held_half_z_m: float
) -> PlaceTarget | None:
    """Pick the receptacle out of a reconstructed scene by name and compute its drop point."""
    if not phrase or not scene_objects:
        return None
    by_name = {o.name: o for o in scene_objects}
    name = match_object_name(list(by_name), phrase)
    if name is None:
        return None
    return place_target_for(by_name[name], held_half_z_m)


def place_target_for(obj: BoxObject, held_half_z_m: float, floor_z: float | None = None) -> PlaceTarget:
    """``obj`` as a welded receptacle (tub collision if hollow) with its drop point.

    ``floor_z`` is a known inner floor (world z) for a body whose open top is not in its mesh.
    """
    pos = np.asarray(obj.pos_m, float)
    hollow, surface_z = (True, floor_z) if floor_z is not None else container_profile(obj)
    drop = np.array([pos[0], pos[1], surface_z + held_half_z_m + CLEARANCE_M])
    pieces = obj.collision_pieces
    if hollow and (floor_z is None or not pieces):
        pieces = container_pieces(obj, surface_z)
    static = replace(obj, collision_pieces=pieces, static=True)
    return PlaceTarget(obj.name, drop, np.asarray(obj.half_size_m, float), static)


def resolve_from_detections(
    cand_sim: np.ndarray,  # (N, K, 3) candidate centroids in the sim frame
    extent_m: np.ndarray,  # (N, K, 3)
    usable: np.ndarray,  # (N,) bool
    release_pinch_m: np.ndarray,  # (3,) human pinch point at release
    held_half_z_m: float,
    exclude_xy_m: np.ndarray | None = None,  # resting xy of the grasped object
    max_dist_m: float = 0.25,
    max_motion_m: float = 0.03,
) -> PlaceTarget | None:
    """Fallback without a reconstruction: the stationary candidate nearest the release pinch.

    ``select_held_object`` keeps the candidate that *moves with* the pinch; the receptacle is
    selected by the opposite test, so the two can never resolve to the same slot.
    """
    idx = np.where(usable)[0]
    if len(idx) < 3:
        return None
    best: tuple[float, int] | None = None
    for k in range(cand_sim.shape[1]):
        track = cand_sim[idx, k]
        if not np.isfinite(track).all():
            continue
        if float(np.linalg.norm(track - track.mean(0), axis=1).max()) > max_motion_m:
            continue  # it moved: that is the carried object, not the receptacle
        centre = track.mean(0)
        if exclude_xy_m is not None and float(np.linalg.norm(centre[:2] - exclude_xy_m)) < 0.05:
            continue
        d = float(np.linalg.norm(centre[:2] - np.asarray(release_pinch_m, float)[:2]))
        if d < max_dist_m and (best is None or d < best[0]):
            best = (d, k)
    if best is None:
        return None
    k = best[1]
    centre = cand_sim[idx, k].mean(0)
    half = np.clip(np.median(extent_m[idx, k], axis=0) / 2, 0.02, 0.12)
    drop = np.array([centre[0], centre[1], centre[2] + half[2] + held_half_z_m + CLEARANCE_M])
    return PlaceTarget("place_target", drop, half, None)


def rebase(target: PlaceTarget, centre_xy_m: np.ndarray, table_top_z_m: float) -> PlaceTarget:
    """Move a receptacle into the retargeting frame without changing its shape.

    A reconstructed scene and the retargeted trajectory are expressed in frames that only agree up
    to the scene alignment: on clip 104 the bowl's reconstructed centre lands 0.3 m from where the
    human actually releases, i.e. off the synthetic table entirely. The mesh is still the best
    source of *shape* (and therefore of the drop height), but the trajectory is the only reliable
    source of *where* -- the human demonstrably releases the object over the receptacle. So keep the
    geometry and re-anchor it to the release point, standing on the table.
    """
    pos = np.asarray(target.obj.pos_m if target.obj is not None else target.drop_point_m, float)
    half = np.asarray(target.half_size_m, float)
    v = _mesh_world(target.obj) if target.obj is not None else None
    bottom = float(pos[2] - half[2]) if v is None else float(v[:, 2].min())
    delta = np.array([float(centre_xy_m[0]) - pos[0], float(centre_xy_m[1]) - pos[1], table_top_z_m - bottom])
    obj = target.obj
    if obj is not None:
        obj = replace(obj, pos_m=tuple(np.asarray(obj.pos_m, float) + delta))
    return PlaceTarget(target.name, target.drop_point_m + delta, half, obj)
