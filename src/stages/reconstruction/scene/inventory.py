"""Every object on the table, as simulation bodies registered to the retargeting frame.

``scripts/perceive_scene_objects.py`` lists the tabletop of a clip from one reference frame (OWLv2 ->
SAM -> MoGe-2 points, camera frame, unscaled) and ``generate_object_mesh.py --scene`` meshes each
crop. This module turns that inventory into ``BoxObject``s in the sim frame of the retargeted motion:

* **Scale.** The pipeline's depth scale is measured against the MoGe depth of ``objects_sam3.npz``;
  the inventory frame was run through MoGe separately, so the scale is carried over by the median
  ratio of the two depth maps.
* **Registration.** Monocular depth is good to a few centimetres, and the held object's resting
  point is already known in the sim frame (where the robot grips it), so the whole layout is
  translated to put the inventory's copy of the held object there. The relative layout -- which is
  what "the same scene" means -- is preserved.
* **Shape.** The generated mesh is rested on its stable face, then scaled per axis to the footprint
  (minimum-area rectangle of the object's points on the table) and height (top of the points above
  the table plane) and yawed to the footprint's long side.
* **Roles.** The held object is replaced by the pipeline's own (it is shaped for the grasp). The
  receptacle is handed to the place-target resolver like a reconstructed scene object, and whatever
  sits in it moves with it when it is re-anchored. A source container ("from the blue bowl") is
  welded, sunk so that its inner floor is the table the held object rests on, and exempt from robot
  contact (the fingers reach past its rim). Everything else is a free body with mass and friction
  from ``heuristic_physics``.
* **Colour.** The task's colour word overrides a detector category that contradicts it (an OWL
  "scoop" that is yellow when the task names a purple scoop).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from stages.manipulation.mjcf_scene import BoxObject
from stages.reconstruction.objects.place_target import (
    CLEARANCE_M,
    _hull_faces,
    container_pieces,
    container_profile,
    tub_profile,
)
from stages.reconstruction.scene.roles import (
    PhraseMatcher,
    clamp_inside,
    colour_name,
    find_held,
    inside,
    is_container,
    pick_receptacle,
    pick_source,
    source_shift,
)

HELD_MATCH_M = 0.15  # inventory copy of the held object must be this close to where the robot grips
MAX_REGISTRATION_M = 0.15  # larger offsets mean the match is wrong; keep the unregistered layout
WORKSPACE_RADIUS_M = 0.55  # objects further from the task (wall sockets, the far shelf) are dropped
MIN_SIZE_M = 0.012  # smallest footprint side of a fitted mesh
MIN_HEIGHT_M = 0.008
REST_CLEARANCE_M = 0.002
NEAR_EDGE_MARGIN_M = 0.02
DISTURBED_M = 0.03
HELD_KEEPOUT_M = 0.01  # default clearance of the scenery from the held object's box
SCENERY_MIN_SCORE = 0.2  # weaker detections (wall sockets, half-visible clutter) are not simulated
HOLLOW_MIN_RIM_M = 0.015  # shallower containers (a plate) are treated as a solid slab
HOLLOW_FLOOR_FRAC = 0.5  # interior points below this fraction of the rim height mean an open top
TRAY_WALL_FRAC = 0.12  # wall thickness of a synthesised tray, as a fraction of the short side
ROUND_WORDS = ("bowl", "scoop", "scooper", "cup", "plate", "pot", "basket")  # tub collision; else a tray
RECEPTACLE_FLOOR_FRAC = 0.25  # inner floor of a named container receptacle the geometry calls solid
if os.environ.get(
    "E2H_HELD_MATCH_M"
):  # walking clips: visual-odometry drift makes the grip land far from the held object
    HELD_MATCH_M = MAX_REGISTRATION_M = float(os.environ["E2H_HELD_MATCH_M"])
TUB_WORDS = ("bowl", "cup", "plate", "pot", "basket")  # always the smooth tub (a heightmap steps its slope)
SQUARE_ASPECT = 1.35  # meshes and footprints this close to square may also be turned by 90 degrees
HMAP_CELL_M = 0.015  # cell of a container's surface heightmap
HMAP_MAX_CELLS = 16  # per side
HMAP_MERGE_M = 0.005  # neighbouring columns this close in height are one collision box
HMAP_OPEN_FRAC = 0.25  # a mesh with this fraction of its cells below half its height is open-topped
DROP_SURFACE_TOL_M = 0.01  # drop spots this close to the lowest surface are equally good


@dataclass(frozen=True)
class InventoryItem:
    id: int
    name: str
    label: str
    points_cam: np.ndarray  # (N, 3) camera-frame MoGe points, unscaled
    rgb: tuple[int, int, int]
    crop: Path
    mesh: Path | None
    task_phrase: bool
    score: float = 1.0


@dataclass(frozen=True)
class Inventory:
    clip: Path
    frame: int
    items: tuple[InventoryItem, ...]
    table_normal_cam: np.ndarray
    table_offset_cam: float
    depth_small: np.ndarray


def load_inventory(clip: Path) -> Inventory | None:
    d = Path(clip) / "scene"
    if not (d / "inventory.json").exists():
        return None
    inv = json.loads((d / "inventory.json").read_text())
    arr = np.load(d / "points.npz")
    items = tuple(
        InventoryItem(
            id=int(o["id"]),
            name=o["name"],
            label=o["label"],
            points_cam=np.asarray(arr[o["points"]], float),
            rgb=tuple(o["rgb_median"]),
            crop=d / o["crop"],
            mesh=d / o["mesh"] if o.get("mesh") and (d / o["mesh"]).exists() else None,
            task_phrase=bool(o.get("task_phrase", False)),
            score=float(o.get("score", 1.0)),
        )
        for o in inv["objects"]
    )
    return Inventory(
        Path(clip),
        int(inv["frame"]),
        items,
        np.asarray(inv["table_normal_cam"], float),
        float(inv["table_offset_cam"]),
        np.asarray(arr["depth_small"], float),
    )


def depth_ratio(ref_depth: np.ndarray, inv_depth: np.ndarray) -> float:
    """Median of ``ref / inv`` over pixels valid in both (MoGe scale differs between runs)."""
    if ref_depth.shape != inv_depth.shape:
        return 1.0
    ok = np.isfinite(ref_depth) & np.isfinite(inv_depth) & (ref_depth > 0) & (inv_depth > 0)
    if ok.sum() < 100:
        return 1.0
    return float(np.median(ref_depth[ok] / inv_depth[ok]))


@dataclass(frozen=True)
class Footprint:
    """An object's extent in the sim frame: rectangle on the table plus heights above it."""

    center_xy: np.ndarray
    axis_xy: np.ndarray  # unit vector of the long side
    half_xy: np.ndarray  # (long, short) half sizes
    bottom_m: float  # lowest point above the table plane (0 for an object standing on it)
    top_m: float
    inner_m: float | None = None  # median height over the middle of the footprint (None: no points)

    @property
    def hollow_floor_m(self) -> float | None:
        """Inner floor height of an open-topped object seen from above, from the points alone."""
        if self.inner_m is None or self.top_m < HOLLOW_MIN_RIM_M:
            return None
        return self.inner_m if self.inner_m < HOLLOW_FLOOR_FRAC * self.top_m else None


def footprint(xy: np.ndarray, height: np.ndarray) -> Footprint:
    import cv2

    c = np.median(xy, axis=0)
    r = np.linalg.norm(xy - c, axis=1)
    core = xy[r <= np.percentile(r, 97)]  # stray mask pixels on the table or the next object
    (cx, cy), (w, h), ang = cv2.minAreaRect(core.astype(np.float32))
    a = np.radians(ang)
    axis = np.array([np.cos(a), np.sin(a)])
    if h > w:
        w, h = h, w
        axis = np.array([-axis[1], axis[0]])
    half = np.maximum(np.array([w, h]) / 2, MIN_SIZE_M / 2)
    d = xy - np.array([cx, cy])
    u = np.column_stack([d @ axis, d @ np.array([-axis[1], axis[0]])])
    mid = (np.abs(u) <= 0.5 * half).all(axis=1)
    return Footprint(
        np.array([cx, cy]),
        axis,
        half,
        float(max(np.percentile(height, 3), 0.0)),
        float(np.percentile(height, 97)),
        float(np.median(height[mid])) if mid.sum() >= 10 else None,
    )


def item_points(
    item: InventoryItem, scale: float, T_sim_camera: np.ndarray, normal_cam: np.ndarray, offset_cam: float
) -> tuple[np.ndarray, np.ndarray]:
    """(sim-frame xy, height above the table plane) of the item's points."""
    p = item.points_cam
    height = (p @ normal_cam - offset_cam) * scale
    sim = (p * scale) @ T_sim_camera[:3, :3].T + T_sim_camera[:3, 3]
    return sim[:, :2], height


def _long_side_yaw(xy: np.ndarray) -> float:
    """Yaw of the long side of the minimum-area rectangle around ``xy``."""
    import cv2

    _, (w, h), ang = cv2.minAreaRect(np.asarray(xy, np.float32))
    a = np.radians(ang)
    return float(a if w >= h else a + np.pi / 2)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_vertices(rest_vertices: np.ndarray, fp: Footprint, height_m: float, quarter: int = 0) -> np.ndarray:
    """Body-frame vertices (bbox centred) of a table-resting mesh stretched onto ``fp`` and ``height_m``.

    ``quarter`` turns the mesh by that many 90 degree steps on the footprint (see ``orient_quarter``).
    """
    v = np.asarray(rest_vertices, float)
    # The same minimum-area rectangle as ``footprint``, so the mesh's sides land on the footprint's
    # (and on the tray walls built from it). The principal axis of a near-square box is off its
    # sides (clip 0: 12 deg), which left the collision walls cutting through the visual lid.
    v = v @ _rot_z(-_long_side_yaw(v[:, :2]) + quarter * np.pi / 2).T  # long side along +x
    lo, hi = v.min(0), v.max(0)
    v = (v - 0.5 * (lo + hi)) / np.maximum(hi - lo, 1e-9)
    v = v * np.array([2 * fp.half_xy[0], 2 * fp.half_xy[1], height_m])
    return v @ _rot_z(np.arctan2(fp.axis_xy[1], fp.axis_xy[0])).T


def _surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int = 12000) -> np.ndarray:
    """Vertices plus area-uniform surface points (fixed seed: the scene is reproducible)."""
    import trimesh

    v = np.asarray(vertices, float)
    mesh = trimesh.Trimesh(v, np.asarray(faces), process=False)
    return np.concatenate([v, trimesh.sample.sample_surface(mesh, count, seed=0)[0]])


def orient_quarter(
    rest_vertices: np.ndarray, faces: np.ndarray, fp: Footprint, height_m: float, obs: np.ndarray
) -> int:
    """The 90 degree turn of the fitted mesh that best explains the observed points.

    The footprint fixes the long side only up to a half turn (and a near-square one not even that):
    clip 101's open case had its upright lid at the front, over the base the tape is put into.
    ``obs`` (N, 3) are the object's points relative to the footprint centre, z above its support;
    the turn with the smallest mean point-to-surface distance wins.
    """
    from scipy.spatial import cKDTree

    if len(obs) < 20:
        return 0
    v = np.asarray(rest_vertices, float)
    ext = np.ptp(v @ _rot_z(-_long_side_yaw(v[:, :2])).T, axis=0)
    square = ext[0] <= SQUARE_ASPECT * max(ext[1], 1e-9) and fp.half_xy[0] <= SQUARE_ASPECT * fp.half_xy[1]
    obs = np.asarray(obs, float)[:: max(1, len(obs) // 3000)]
    best = (np.inf, 0)
    for q in (0, 1, 2, 3) if square else (0, 2):
        s = _surface_samples(fit_vertices(v, fp, height_m, q), faces) + [0.0, 0.0, 0.5 * height_m]
        d = float(cKDTree(s).query(obs)[0].mean())
        if d < best[0] - 1e-6:
            best = (d, q)
    return best[1]


@dataclass(frozen=True)
class HeightMap:
    """Top surface of a body over a grid aligned with its footprint (world frame, NaN where empty)."""

    center_xy: np.ndarray
    axis_xy: np.ndarray
    cell_xy: np.ndarray  # cell size along (axis, side)
    top_z: np.ndarray  # (nx, ny) world z of the surface
    bottom_z: float

    def _frame(self) -> np.ndarray:
        return np.stack([self.axis_xy, [-self.axis_xy[1], self.axis_xy[0]]])

    @property
    def half_xy(self) -> np.ndarray:
        return 0.5 * self.cell_xy * np.asarray(self.top_z.shape)

    def surface_under(self, xy: np.ndarray, half_xy: np.ndarray) -> np.ndarray:
        """(M,) highest surface z under an axis-aligned ``half_xy`` box at each of ``xy`` (M, 2)."""
        R = self._frame()
        u = (np.atleast_2d(xy) - self.center_xy) @ R.T + self.half_xy
        ext = np.abs(R) @ np.asarray(half_xy, float)[:2]
        n = np.asarray(self.top_z.shape)
        lo = np.clip(np.floor((u - ext) / self.cell_xy).astype(int), 0, n - 1)
        hi = np.clip(np.floor((u + ext) / self.cell_xy).astype(int), 0, n - 1)
        top = np.where(np.isnan(self.top_z), self.bottom_z, self.top_z)
        return np.array([top[a[0] : b[0] + 1, a[1] : b[1] + 1].max() for a, b in zip(lo, hi, strict=True)])

    def free_spot(
        self, xy: np.ndarray, half_xy: np.ndarray, floor_z: float, max_shift_m: float = HMAP_CELL_M
    ) -> np.ndarray:
        """The spot within ``max_shift_m`` of ``xy`` where a ``half_xy`` box stands on the floor.

        Contents are laid out from their perceived footprint, which can overlap a wall of the
        container's actual mesh (clip 0: the cube inside the half-turned box spawned in its wall and
        was thrown out). Moved further, a content takes the room the human put the held object into
        (clip 0: the cube 4.7 cm in, the drop pushed out of reach); ``xy`` itself then, resting on
        whatever is under it.
        """
        R = self._frame()
        n = int(np.ceil(max_shift_m / HMAP_CELL_M * 3))
        g = np.arange(-n, n + 1) * HMAP_CELL_M / 3
        d = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        r = np.linalg.norm(d, axis=1)
        d = d[np.argsort(r, kind="stable")][: int((r <= max_shift_m + 1e-9).sum())]
        cand = np.asarray(xy, float)[:2] + d
        u = np.abs((cand - self.center_xy) @ R.T) + np.abs(R) @ np.asarray(half_xy, float)[:2]
        cand = cand[np.all(u <= self.half_xy, axis=1)]
        if not len(cand):
            return np.asarray(xy, float)[:2]
        ok = np.flatnonzero(self.surface_under(cand, half_xy) <= floor_z + HMAP_MERGE_M)
        return cand[ok[0]] if len(ok) else np.asarray(xy, float)[:2]

    def shifted(self, delta: np.ndarray) -> HeightMap:
        return replace(
            self,
            center_xy=self.center_xy + delta[:2],
            top_z=self.top_z + delta[2],
            bottom_z=self.bottom_z + delta[2],
        )


def body_heightmap(vertices: np.ndarray, faces: np.ndarray, fp: Footprint, pos: np.ndarray) -> HeightMap:
    """Heightmap of a body-frame mesh (``fit_vertices`` output) placed at ``pos``."""
    R = np.stack([fp.axis_xy, [-fp.axis_xy[1], fp.axis_xy[0]]])
    s = _surface_samples(vertices, faces)
    u = s[:, :2] @ R.T
    half = np.maximum(np.abs(u).max(0), 1e-4)
    n = np.clip(np.ceil(2 * half / HMAP_CELL_M), 2, HMAP_MAX_CELLS).astype(int)
    cell = 2 * half / n
    ij = np.clip(np.floor((u + half) / cell).astype(int), 0, n - 1)
    top = np.full(tuple(n), np.nan)
    np.fmax.at(top, (ij[:, 0], ij[:, 1]), s[:, 2])
    return HeightMap(
        np.asarray(pos[:2], float),
        np.asarray(fp.axis_xy, float),
        cell,
        _merge_rows(top) + float(pos[2]),
        float(pos[2] + s[:, 2].min()),
    )


def _merge_rows(top: np.ndarray) -> np.ndarray:
    """Each run of columns along a grid row within ``HMAP_MERGE_M`` of each other raised to its highest.

    The collision has one box per run, so the heightmap the floor, sinking and drop height are read
    from is the collision itself: from the unmerged cells clip 114's tray floor was 5 mm under its
    boxes and the block that rested in it spawned inside them and was thrown 5 cm up.
    """
    out = top.copy()
    nx, ny = top.shape
    for j in range(ny):
        i = 0
        while i < nx:
            if np.isnan(top[i, j]):
                i += 1
                continue
            k, hi = i, top[i, j]
            while k + 1 < nx and not np.isnan(top[k + 1, j]) and abs(top[k + 1, j] - hi) < HMAP_MERGE_M:
                k += 1
                hi = max(hi, top[k, j])
            out[i : k + 1, j] = hi
            i = k + 1
    return out


def heightmap_open(hm: HeightMap) -> bool:
    """Open-topped: a good part of the body lies well below its highest point."""
    occ = ~np.isnan(hm.top_z)
    if occ.sum() < 4:
        return False
    rel = (hm.top_z[occ] - hm.bottom_z) / max(float(np.nanmax(hm.top_z)) - hm.bottom_z, 1e-9)
    return bool((rel < 0.5).mean() >= HMAP_OPEN_FRAC)


def heightmap_floor_z(hm: HeightMap) -> float:
    """Inner floor (world z): the median of the cells in the lower half of the body."""
    t = hm.top_z[~np.isnan(hm.top_z)]
    low = t[t - hm.bottom_z < 0.5 * (t.max() - hm.bottom_z)]
    return float(np.median(low if len(low) else t))


def _heightmap_pieces(hm: HeightMap, pos: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """One box per run of similar columns along each grid row, in body coordinates.

    A tray's four full-height walls misrepresent anything that is not a tray: clip 101's open case
    has a 4 cm base in front of a 14 cm lid, clip 117's scoop a flat handle beside its pan.
    """
    R = hm._frame()
    nx, ny = hm.top_z.shape
    z_lo = hm.bottom_z - float(pos[2])
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    pieces = []
    for j in range(ny):
        i = 0
        while i < nx:
            if np.isnan(hm.top_z[i, j]):
                i += 1
                continue
            k, top = i, hm.top_z[i, j]
            # ``body_heightmap`` has already raised each run to one height: one box per run.
            while k + 1 < nx and hm.top_z[k + 1, j] == top:
                k += 1
            z_hi = max(float(top) - float(pos[2]), z_lo + 0.002)
            u = (np.array([i, j]) + np.array([k + 1, j + 1])) / 2 * hm.cell_xy - hm.half_xy
            h = np.array([(k + 1 - i) * hm.cell_xy[0] / 2, hm.cell_xy[1] / 2])
            c = np.array([*(u @ R), (z_lo + z_hi) / 2])
            hv = np.array([*h, (z_hi - z_lo) / 2])
            local = signs * hv
            v = np.column_stack([local[:, :2] @ R, local[:, 2]]) + c
            pieces.append((v, _hull_faces(v)))
            i = k + 1
    return tuple(pieces)


def _box_mesh(fp: Footprint, height_m: float) -> tuple[np.ndarray, np.ndarray]:
    from stages.reconstruction.objects.mesh_asset import prism_hull

    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-0.5, 0.5)], float)
    v = corners * np.array([fp.half_xy[0], fp.half_xy[1], height_m])
    v = v @ _rot_z(np.arctan2(fp.axis_xy[1], fp.axis_xy[0])).T
    return prism_hull(v)


def _tray_mesh(fp: Footprint, height_m: float, floor_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Open-topped box (outer walls, rim, inner walls, floor) for a hollow container without a mesh."""
    t = TRAY_WALL_FRAC * fp.half_xy[1]
    rings = []
    for (hx, hy), z in (
        (fp.half_xy, -0.5),
        (fp.half_xy, 0.5),
        (fp.half_xy - t, 0.5),
        (fp.half_xy - t, floor_m / max(height_m, 1e-9) - 0.5),
    ):
        rings.append([[sx * hx, sy * hy, z * height_m] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    v = np.asarray(rings, float).reshape(-1, 3) @ _rot_z(np.arctan2(fp.axis_xy[1], fp.axis_xy[0])).T
    faces = []
    for r in range(3):  # side bands between consecutive rings
        for k in range(4):
            a, b = 4 * r + k, 4 * r + (k + 1) % 4
            faces += [[a, b, b + 4], [a, b + 4, a + 4]]
    faces += [[0, 2, 1], [0, 3, 2], [12, 13, 14], [12, 14, 15]]  # bottom, inner floor
    return v, np.asarray(faces)


def _tray_pieces(
    fp: Footprint, obj: BoxObject, floor_z_world: float
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Floor slab plus four straight walls along the footprint, in body coordinates.

    The generic tub (``container_pieces``) is an axis-aligned elliptical ring; inside a yawed
    rectangular lid its walls cut through the corners, where the contents start (clip 0: the cube in
    the corner of the box lid was flung 0.5 m on the first step).
    """
    pos = np.asarray(obj.pos_m, float)
    half_z = float(obj.half_size_m[2])
    z_lo, z_hi = -half_z, half_z
    z_floor = float(np.clip(floor_z_world - pos[2], z_lo + 0.002, z_hi - 0.002))
    t = max(TRAY_WALL_FRAC * fp.half_xy[1], 0.003)
    hx, hy = fp.half_xy
    R = _rot_z(np.arctan2(fp.axis_xy[1], fp.axis_xy[0]))
    boxes = [
        ((0.0, 0.0, (z_lo + z_floor) / 2), (hx, hy, (z_floor - z_lo) / 2)),
        ((hx - t / 2, 0.0, 0.0), (t / 2, hy, half_z)),
        ((-hx + t / 2, 0.0, 0.0), (t / 2, hy, half_z)),
        ((0.0, hy - t / 2, 0.0), (hx, t / 2, half_z)),
        ((0.0, -hy + t / 2, 0.0), (hx, t / 2, half_z)),
    ]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    pieces = []
    for c, h in boxes:
        v = (np.asarray(c) + signs * np.asarray(h)) @ R.T
        pieces.append((v, _hull_faces(v)))
    return tuple(pieces)


def _mesh_asset(item: InventoryItem) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """(table-resting vertices, faces, uv, texture) of the generated mesh, unit scale."""
    import cv2
    import trimesh

    from stages.reconstruction.objects.mesh_asset import rest_transform, transform_points
    from stages.reconstruction.objects.mesh_texture import bake_texture

    raw = trimesh.load(item.mesh, force="mesh")
    T = rest_transform(raw.vertices, raw.faces, 1.0, min_height_m=0.0)
    crop = cv2.imread(str(item.crop), cv2.IMREAD_UNCHANGED)
    if crop is None or crop.ndim != 3 or crop.shape[-1] != 4:
        return transform_points(T, raw.vertices), np.asarray(raw.faces), None, None
    tex = bake_texture(
        np.asarray(raw.vertices), np.asarray(raw.faces), cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)
    )
    return transform_points(T, tex.vertices), tex.faces, tex.uv, tex.image


@dataclass
class SceneInventory:
    """The scene bodies plus what the pipeline needs to wire the task roles."""

    objects: list[BoxObject] = field(default_factory=list)  # free and welded scenery
    receptacle: BoxObject | None = None  # for ``resolve_from_scene``
    receptacle_contents: list[str] = field(default_factory=list)
    receptacle_floor_z: float | None = None  # inner floor (world z) of a hollow receptacle
    receptacle_fp: Footprint | None = None
    receptacle_round: bool = False  # its collision is the elliptical tub, not the rectangular tray
    receptacle_hmap: HeightMap | None = None  # its collision is this heightmap (an open mesh)
    report: dict = field(default_factory=dict)

    def move_receptacle(self, receptacle: BoxObject) -> None:
        """Carry the receptacle's contents along when the pipeline re-anchors it."""
        if self.receptacle is None:
            return
        delta = np.asarray(receptacle.pos_m, float) - np.asarray(self.receptacle.pos_m, float)
        self.objects = [
            replace(o, pos_m=tuple(np.asarray(o.pos_m, float) + delta))
            if o.name in self.receptacle_contents
            else o
            for o in self.objects
        ]
        prev = np.asarray(self.report.get("receptacle_shift_m", np.zeros(3)), float)
        self.report["receptacle_shift_m"] = [round(float(x), 4) for x in prev + delta]
        self.receptacle = receptacle  # a second move (pulled into reach) is measured from here
        if self.receptacle_floor_z is not None:
            self.receptacle_floor_z += float(delta[2])
        if self.receptacle_fp is not None:
            self.receptacle_fp = replace(
                self.receptacle_fp, center_xy=self.receptacle_fp.center_xy + delta[:2]
            )
        if self.receptacle_hmap is not None:
            self.receptacle_hmap = self.receptacle_hmap.shifted(delta)

    def drop_xy(self, want_xy: np.ndarray, held_half: np.ndarray, step_m: float = 0.005) -> np.ndarray:
        """Free spot inside the receptacle nearest ``want_xy`` for a ``held_half`` (xy or xyz) box.

        The human lets go where there was room in the *real* lid; in sim the contents are a few
        centimetres off, and dropping onto them leaves the object tilted on the cube (clip 0).
        """
        fp = self.receptacle_fp
        if fp is None:
            return np.asarray(want_xy, float)
        held_half = np.asarray(held_half, float)
        held_half_xy = held_half[:2]
        if self.receptacle_round and self.receptacle is not None and self.receptacle_floor_z is not None:
            # The tub narrows towards its floor, much smaller than the footprint rectangle: a spot
            # in the rectangle's corner is over the rim, and the object stays on it (clip 116: 6.6 cm
            # off-centre in a bowl with a 5 cm floor). The held box, released just above the floor,
            # has to clear the inner wall at its mid-height; from there it slides down the slope.
            z, frac = tub_profile(self.receptacle, self.receptacle_floor_z)
            mid_z = z[0] + CLEARANCE_M + (held_half[2] if len(held_half) > 2 else 0.0)
            c = np.asarray(self.receptacle.pos_m[:2], float)
            radii = float(np.interp(mid_z, z, frac)) * np.asarray(self.receptacle.half_size_m[:2], float)
            gu, gv = (np.arange(-r, r + 1e-9, step_m) for r in radii)
            cand = c + np.stack(np.meshgrid(gu, gv), -1).reshape(-1, 2)
            corners = np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * (held_half_xy + 0.003)
            rel = (cand - c)[:, None, :] + corners[None]
            excess = np.maximum(np.sqrt(((rel / radii) ** 2).sum(-1)).max(axis=1) - 1.0, 0.0)
        elif self.receptacle_hmap is not None:
            # Anywhere on the mesh, preferring its lowest surface: the base of clip 101's open case,
            # never the top of its lid.
            hm = self.receptacle_hmap
            R = hm._frame()
            room = np.maximum(hm.half_xy - np.abs(R) @ held_half_xy - 0.003, 0.0)
            gu, gv = (np.arange(-r, r + 1e-9, step_m) if r > 0 else np.zeros(1) for r in room)
            cand = hm.center_xy + np.stack(np.meshgrid(gu, gv), -1).reshape(-1, 2) @ R
            z = hm.surface_under(cand, held_half_xy + 0.003)
            excess = np.maximum(z - z.min() - DROP_SURFACE_TOL_M, 0.0)
        else:
            side = np.array([-fp.axis_xy[1], fp.axis_xy[0]])
            R = np.stack([fp.axis_xy, side])
            wall = max(TRAY_WALL_FRAC * fp.half_xy[1], 0.003)
            room = np.maximum(fp.half_xy - wall - np.abs(R) @ held_half_xy - 0.003, 0.0)
            gu, gv = (np.arange(-r, r + 1e-9, step_m) if r > 0 else np.zeros(1) for r in room)
            cand = fp.center_xy + np.stack(np.meshgrid(gu, gv), -1).reshape(-1, 2) @ R
            excess = np.zeros(len(cand))
        overlap = np.zeros(len(cand))
        for o in self.objects:
            if o.name not in self.receptacle_contents:
                continue
            d = np.maximum(
                np.asarray(held_half_xy)
                + np.asarray(o.half_size_m[:2])
                + 0.005
                - np.abs(cand - np.asarray(o.pos_m[:2])),
                0.0,
            )
            overlap += d.prod(axis=1)
        dist = np.linalg.norm(cand - want_xy, axis=1)
        return cand[np.lexsort((dist, np.round(overlap, 7), np.round(excess, 4)))[0]]

    def drop_z(
        self, xy: np.ndarray, held_half: np.ndarray, default: float, clearance_m: float = CLEARANCE_M
    ) -> float:
        """Centre z of the held ``held_half`` box released at ``xy`` on the receptacle's heightmap."""
        if self.receptacle_hmap is None:
            return default
        held_half = np.asarray(held_half, float)
        surface = self.receptacle_hmap.surface_under(np.asarray(xy, float)[:2], held_half[:2])[0]
        return float(surface + held_half[2] + clearance_m)


def _penetration(a: Footprint, b: Footprint) -> np.ndarray | None:
    """Minimum translation moving ``b`` out of ``a`` (separating-axis test on two rectangles)."""
    d = b.center_xy - a.center_xy
    best = None
    for fp in (a, b):
        for n in (fp.axis_xy, np.array([-fp.axis_xy[1], fp.axis_xy[0]])):
            ra = sum(h * abs(ax @ n) for h, ax in zip(a.half_xy, _axes(a), strict=True))
            rb = sum(h * abs(ax @ n) for h, ax in zip(b.half_xy, _axes(b), strict=True))
            depth = ra + rb - abs(d @ n)
            if depth <= 0:
                return None
            if best is None or depth < best[0]:
                best = (depth, n if d @ n >= 0 else -n)
    return best[1] * best[0]


def _axes(fp: Footprint) -> tuple[np.ndarray, np.ndarray]:
    return fp.axis_xy, np.array([-fp.axis_xy[1], fp.axis_xy[0]])


def separate(
    fps: dict[int, Footprint],
    movable: set[int],
    iters: int = 40,
    never_stacked: frozenset[frozenset[int]] = frozenset(),
) -> dict[int, np.ndarray]:
    """Push movable footprints apart (and off fixed ones); returns the shift applied to each.

    Perceived footprints are good to a centimetre or two, so neighbours on a cluttered table start
    interpenetrating and the solver flings them apart on the first step. One footprint lying wholly
    over another's centre is a deliberate stack (contents of a lid) and is left alone, unless the
    pair is in ``never_stacked``.
    """
    cur = dict(fps)
    for _ in range(iters):
        moved = False
        ids = list(cur)
        for k, i in enumerate(ids):
            for j in ids[k + 1 :]:
                if i not in movable and j not in movable:
                    continue
                stacked = inside(cur[i], cur[j].center_xy) or inside(cur[j], cur[i].center_xy)
                if stacked and frozenset((i, j)) not in never_stacked:
                    continue
                mtv = _penetration(cur[i], cur[j])
                if mtv is None:
                    continue
                mtv = mtv + 0.002 * mtv / max(np.linalg.norm(mtv), 1e-9)
                wi, wj = (i in movable), (j in movable)
                share = 0.5 if wi and wj else 1.0
                if wi:
                    cur[i] = replace(cur[i], center_xy=cur[i].center_xy - share * mtv)
                if wj:
                    cur[j] = replace(cur[j], center_xy=cur[j].center_xy + share * mtv)
                moved = True
        if not moved:
            break
    return {i: cur[i].center_xy - fps[i].center_xy for i in fps}


def _clamp_within_walls(holder: Footprint, xy: np.ndarray, half_xy: np.ndarray) -> np.ndarray:
    """Keep an axis-aligned ``half_xy`` box at ``xy`` clear of a tray's walls (it starts inside one)."""
    side = np.array([-holder.axis_xy[1], holder.axis_xy[0]])
    R = np.stack([holder.axis_xy, side])  # world -> holder frame
    ext = np.abs(R) @ half_xy
    room = np.maximum(holder.half_xy - max(TRAY_WALL_FRAC * holder.half_xy[1], 0.003) - ext - 0.002, 0.0)
    u = np.clip(R @ (xy - holder.center_xy), -room, room)
    return holder.center_xy + R.T @ u


def build_scene_inventory(
    inv: Inventory,
    scale: float,
    T_sim_camera: np.ndarray,
    table_z: float,
    held_xy: np.ndarray,
    held_phrase: str,
    place_phrase: str | None,
    source_phrase: str | None,
    table_near_x: float,
    place_hint_xy: np.ndarray | None = None,
    held_half_xy: np.ndarray | None = None,
    held_keepout_m: float = HELD_KEEPOUT_M,
    carry_xy: np.ndarray | None = None,
    source_from: str = "grip",
    max_registration_m: float | None = None,
) -> SceneInventory:
    """Sim bodies of every inventoried object except the held one (see the module docstring)."""
    from stages.reconstruction.objects.physics_params import heuristic_physics

    pts = {
        it.id: item_points(it, scale, T_sim_camera, inv.table_normal_cam, inv.table_offset_cam)
        for it in inv.items
    }
    fps = {i: footprint(*p) for i, p in pts.items()}
    # The points around their own footprint's centre, to turn each mesh the way it was seen.
    obs = {i: np.column_stack([p[0] - fps[i].center_xy, p[1]]) for i, p in pts.items()}
    names = {it.id: it.name for it in inv.items}
    labels = {it.id: it.label for it in inv.items}
    colours = {it.id: colour_name(it.rgb) for it in inv.items}
    rgbs = {it.id: it.rgb for it in inv.items}
    match = PhraseMatcher(names, labels, colours, rgbs, fps)
    report: dict = {"scale": round(float(scale), 4), "objects": []}
    cap = MAX_REGISTRATION_M if max_registration_m is None else max_registration_m
    held_id = find_held(match, held_xy, held_phrase, place_phrase, source_phrase, held_half_xy, HELD_MATCH_M)
    offset = np.zeros(2)
    registered, attempted = False, held_id is not None
    if held_id is not None:
        # The robot grips somewhere on the object, not necessarily at its centre (a long stapler):
        # register the grip to the nearest point of the inventory footprint.
        offset = held_xy - clamp_inside(fps[held_id], held_xy)
        report["grip_gap_m"] = float(np.linalg.norm(offset))
        registered = bool(np.linalg.norm(offset) <= cap)
        if not registered:
            offset = np.zeros(2)
    report["held_match"] = names.get(held_id)
    report["relabelled"] = match.relabelled  # filled in as the place and source phrases are matched
    report["registration_xy_m"] = [round(float(x), 4) for x in offset]
    fps = {i: replace(fp, center_xy=fp.center_xy + offset) for i, fp in fps.items()}
    match.fps = fps
    src_id = pick_source(match, fps, held_id, held_xy, held_half_xy, source_phrase, source_from, report)
    if src_id is not None:
        report["support"] = {"name": names[src_id], "top_m": round(float(fps[src_id].top_m), 4)}
        attempted = True
        # The held object is picked from the container's floor, never from inside its wall.
        shift = source_shift(fps[src_id], held_xy, held_half_xy)
        report["grip_gap_m"] = float(np.linalg.norm(offset + shift))
        if np.linalg.norm(offset + shift) <= cap:
            offset = offset + shift
            fps = {i: replace(fp, center_xy=fp.center_xy + shift) for i, fp in fps.items()}
            match.fps = fps
            report["registration_xy_m"] = [round(float(x), 4) for x in offset]
            registered = True
    registered = registered or not attempted
    if not registered:
        # The held object or its source was found, but too far from the grip to trust the layout
        # (clip 108: the pen's stand is 22 cm from the grip). Its items would stand where the human
        # hand goes (the robot gripped the pen inside the tables), so none is simulated and the
        # pipeline keeps its own held object and place target.
        src_id = None
        report["unregistered"] = True
        report["warning"] = (
            f"inventory dropped: grip {report['grip_gap_m']:.3f} m from its source/held object (cap {cap} m)"
        )
    rec_id = (
        pick_receptacle(match, held_id, src_id, held_xy, place_phrase, place_hint_xy, carry_xy, report)
        if registered
        else None
    )
    # Keep the scenery clear of the held object (as the pipeline shapes it) and of each other.
    keep = dict(fps)
    keepout_id = -1
    if held_half_xy is not None:
        keep[keepout_id] = Footprint(
            np.asarray(held_xy, float),
            np.array([1.0, 0.0]),
            np.asarray(held_half_xy, float) + held_keepout_m,
            0,
            0,
        )
    keep.pop(held_id, None)
    # The receptacle may be pushed too (it is perceived, not measured); the source holds the held
    # object by definition. Whatever stands in the receptacle moves with it.
    movable = {i for i in keep if i not in (keepout_id, src_id)}
    # The held object never starts in its own receptacle: a monocular layout that puts it there
    # (clip 108, depth stretched 20 cm) would trap it under the receptacle's walls.
    apart = frozenset({frozenset((keepout_id, rec_id))}) if rec_id is not None else frozenset()
    shift = separate(keep, movable, never_stacked=apart)
    if rec_id is not None and np.linalg.norm(shift[rec_id]) > 0:
        for i in movable - {rec_id}:
            if inside(fps[rec_id], fps[i].center_xy, 0.9):
                shift[i] = shift[i] + shift[rec_id]
    fps = {i: replace(fp, center_xy=fp.center_xy + shift.get(i, 0.0)) for i, fp in fps.items()}
    report["separated_m"] = {
        names[i]: round(float(np.linalg.norm(v)), 4)
        for i, v in shift.items()
        if i in names and np.linalg.norm(v) > 1e-4
    }
    task_xy = [held_xy] + ([fps[rec_id].center_xy] if rec_id is not None else [])
    out = SceneInventory(report=report)
    floors: dict[int, float] = {}  # world inner-floor z of welded hollow containers
    hmaps: dict[int, HeightMap] = {}  # their surface, for those whose collision is a heightmap
    order = sorted(fps, key=lambda i: (i not in (rec_id, src_id), not is_container(labels[i])))
    for i in order:
        it = next(x for x in inv.items if x.id == i)
        fp = fps[i]
        entry = {"name": it.name, "label": labels[i], "id": i}
        if i == held_id:
            report["objects"].append({**entry, "role": "held", "kept": False})
            continue
        if not registered:
            report["objects"].append({**entry, "role": "scenery", "kept": False, "why": "unregistered"})
            continue
        if i not in (rec_id, src_id) and it.score < SCENERY_MIN_SCORE:
            report["objects"].append({**entry, "role": "scenery", "kept": False, "why": "low score"})
            continue
        dist = min(float(np.linalg.norm(fp.center_xy - p)) for p in task_xy)
        if i not in (rec_id, src_id) and dist > WORKSPACE_RADIUS_M:
            report["objects"].append({**entry, "role": "scenery", "kept": False, "why": "outside workspace"})
            continue
        corners_x = fp.center_xy[0] - abs(fp.axis_xy[0]) * fp.half_xy[0] - abs(fp.axis_xy[1]) * fp.half_xy[1]
        if i not in (rec_id, src_id) and corners_x < table_near_x + NEAR_EDGE_MARGIN_M:
            report["objects"].append({**entry, "role": "scenery", "kept": False, "why": "off the near edge"})
            continue
        # What it stands on: the table, or the floor of a welded container under it.
        support_z, holder = table_z, None
        for sid, floor_z in floors.items():
            if inside(fps[sid], fp.center_xy, 0.8) and floor_z > support_z:
                support_z, holder = floor_z, sid
        height = max(fp.top_m - (support_z - table_z), MIN_HEIGHT_M)
        # Only a container by name can be open-topped: a table or stand in a task role receives on its
        # top, and the floor seen between its legs reads as a dip (clip 108: the pen went "into" it).
        named_container = is_container(labels[i])
        container = named_container or i in (rec_id, src_id)
        pts_floor = fp.hollow_floor_m if named_container else None
        if it.mesh is not None:
            v0, faces, uv, tex = _mesh_asset(it)
            seen = obs[i] - [0.0, 0.0, support_z - table_z]
            quarter = orient_quarter(v0, faces, fp, height, seen)
            verts = fit_vertices(v0, fp, height, quarter)
            if quarter:
                entry["quarter_turns"] = quarter
        elif pts_floor is not None:
            verts, faces = _tray_mesh(fp, height, pts_floor)
            uv, tex = None, None
        else:
            verts, faces = _box_mesh(fp, height)
            uv, tex = None, None
        half = 0.5 * (verts.max(0) - verts.min(0))
        pos = np.array([*fp.center_xy, support_z + half[2] + REST_CLEARANCE_M])
        if holder is not None and holder in hmaps:
            pos[:2] = hmaps[holder].free_spot(pos[:2], half[:2], floors[holder])
            pos[2] = hmaps[holder].surface_under(pos[:2], half[:2])[0] + half[2] + REST_CLEARANCE_M
        elif holder is not None:
            pos[:2] = _clamp_within_walls(fps[holder], pos[:2], half[:2])
        static = i in (rec_id, src_id)
        volume = float(8 * half.prod() * 0.5)
        phys = heuristic_physics(labels[i], volume)
        rgba = (1.0, 1.0, 1.0, 1.0) if tex is not None else (*(np.asarray(it.rgb) / 255.0), 1.0)
        obj = BoxObject(
            name=f"scene_{it.name}",
            half_size_m=tuple(half),
            pos_m=tuple(pos),
            mass_kg=float(phys["mass"]),
            friction=(float(phys["friction"]), 0.01, 0.001),
            rgba=tuple(rgba),
            mesh_vertices=verts,
            mesh_faces=np.asarray(faces),
            mesh_uv=uv,
            mesh_texture=tex,
            static=static,
        )
        role = "receptacle" if i == rec_id else "source" if i == src_id else "scenery"
        round_ = any(w in labels[i].lower() for w in ROUND_WORDS)
        tub = any(w in labels[i].lower() for w in TUB_WORDS)
        hmap = None
        if container and it.mesh is not None and not tub:
            hm = body_heightmap(verts, faces, fp, np.asarray(obj.pos_m, float))
            hmap = hm if heightmap_open(hm) else None
        if hmap is not None:
            # An open mesh is its own collision: the floor, walls, lid and handle where they are.
            obj = replace(obj, collision_pieces=_heightmap_pieces(hmap, np.asarray(obj.pos_m, float)))
            floor_z = heightmap_floor_z(hmap)
            entry["hollow"], entry["collision"] = True, "heightmap"
            if i == src_id:
                # The held object rests on the floor under its own footprint, which can be a run
                # higher than the median floor; a wall under it does not count.
                rest_z = floor_z
                if held_half_xy is not None:
                    under = hmap.surface_under(np.asarray(held_xy, float)[:2], np.asarray(held_half_xy))[0]
                    rest_z = float(np.clip(under, floor_z, floor_z + HMAP_MERGE_M))
                sink = rest_z - table_z
                obj = replace(
                    obj, pos_m=(obj.pos_m[0], obj.pos_m[1], obj.pos_m[2] - sink), robot_collision=False
                )
                hmap = hmap.shifted(np.array([0.0, 0.0, -sink]))
                entry["sunk_m"] = round(sink, 4)
        elif container:
            hollow, floor_z = container_profile(obj) if named_container else (False, None)
            if pts_floor is not None:
                # The points see the open top even when the generated mesh is closed (a lid, a tray).
                hollow, floor_z = True, float(support_z + pts_floor)
            if not hollow and i == rec_id and named_container:
                # The task puts the object *in* it: a scoop whose handle fills the footprint's middle
                # and whose mesh came out closed still has a pan (clip 10: dropped on its top).
                hollow, floor_z = True, float(support_z + RECEPTACLE_FLOOR_FRAC * height)
                entry["hollow_from"] = "task"
            if hollow:
                pieces = container_pieces(obj, floor_z) if round_ else _tray_pieces(fp, obj, floor_z)
                obj = replace(obj, collision_pieces=pieces)
            if i == src_id:
                # The held object rests on the table in the retargeting frame; the container it is
                # picked from is sunk so that its inner floor (or its top, for a stand) is that table.
                sink = (floor_z if hollow else float(obj.pos_m[2] + half[2])) - table_z
                obj = replace(
                    obj, pos_m=(obj.pos_m[0], obj.pos_m[1], obj.pos_m[2] - sink), robot_collision=False
                )
                entry["sunk_m"] = round(sink, 4)
            entry["hollow"] = bool(hollow)
        if static and entry.get("hollow"):
            floors[i] = floor_z if i != src_id else floor_z - entry["sunk_m"]
            if hmap is not None:
                hmaps[i] = hmap
        if i == rec_id:
            out.receptacle = obj
            out.receptacle_floor_z = floor_z if entry.get("hollow") else None
            out.receptacle_fp = fp
            out.receptacle_round = bool(entry.get("hollow") and round_ and hmap is None)
            out.receptacle_hmap = hmap
        else:
            out.objects.append(obj)
        report["objects"].append(
            {
                **entry,
                "role": role,
                "kept": True,
                "static": static,
                "pos_m": [round(float(x), 4) for x in obj.pos_m],
                "size_m": [round(float(x), 4) for x in 2 * half],
                "mass_kg": round(float(obj.mass_kg), 3),
                "mesh": it.mesh is not None,
            }
        )
    if rec_id is not None:
        rfp = fps[rec_id]
        out.receptacle_contents = [
            o.name
            for o in out.objects
            if not o.static
            and inside(rfp, np.asarray(o.pos_m[:2]), 0.9)
            and o.pos_m[2] - o.half_size_m[2] > table_z
        ]
    report["receptacle"] = names.get(rec_id)
    report["source"] = names.get(src_id)
    report["receptacle_contents"] = list(out.receptacle_contents)
    return out


def scene_extent_xy(objects: list[BoxObject]) -> tuple[np.ndarray, np.ndarray] | None:
    """(lo, hi) xy corners covering every object, for growing the table under them."""
    if not objects:
        return None
    lo = np.min([np.asarray(o.pos_m[:2]) - np.asarray(o.half_size_m[:2]) for o in objects], axis=0)
    hi = np.max([np.asarray(o.pos_m[:2]) + np.asarray(o.half_size_m[:2]) for o in objects], axis=0)
    return lo, hi


def disturbance(object_T_sim: np.ndarray, names: list[str], skip: set[str]) -> dict:
    """How far each free scene object was moved from its start (pushed, knocked over, carried)."""
    out = {}
    for k, name in enumerate(names):
        if name in skip:
            continue
        p = object_T_sim[:, k, :3, 3]
        out[name] = {
            "final_xy_m": round(float(np.linalg.norm(p[-1, :2] - p[0, :2])), 4),
            "max_m": round(float(np.linalg.norm(p - p[0], axis=1).max()), 4),
        }
    disturbed = sorted(n for n, d in out.items() if d["final_xy_m"] > DISTURBED_M)
    return {"per_object": out, "disturbed": disturbed}
