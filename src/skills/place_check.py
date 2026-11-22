"""Is the held object hidden inside the receptacle after the place? (Track RSI)

The collision proxy of a receptacle (convex pieces, or an axis box) can leave the object resting inside
its visual mesh, so the physics is quiet yet the object disappears in the render. A held-mesh vertex is
*covered* when a receptacle face lies above it (ray along the world +z, farther than
``EPS_M`` so an object resting on the surface does not count). No watertight mesh is needed. A bowl
that holds the object is open at the top and covers nothing.
"""

import numpy as np

EPS_M = 0.003  # a face this close above a vertex is the surface it rests on
BURIED_FRAC = 0.3  # covered share of the held vertices that fires PLACE_BURIED
MAX_POINTS = 300
_BOX_FACES = np.array(
    [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
    ]
    + [[1, 5, 7], [1, 7, 3]]
)


def box_mesh(half_size_m) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(half_size_m, float)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return corners * h, _BOX_FACES


def _transform(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v @ T[:3, :3].T + T[:3, 3]


def _faces_above(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per point: is there a triangle (Moller-Trumbore, ray +z) more than EPS_M above it."""
    p0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    # ray direction d = +z: h = d x e2, a = e1 . h
    h = np.stack([-e2[:, 1], e2[:, 0], np.zeros(len(e2))], axis=1)
    a = np.einsum("fi,fi->f", e1, h)
    ok = np.abs(a) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    covered = np.zeros(len(points), bool)
    for i, p in enumerate(points):
        s = p - p0
        u = inv * np.einsum("fi,fi->f", s, h)
        q = np.cross(s, e1)
        v = inv * q[:, 2]
        t = inv * np.einsum("fi,fi->f", e2, q)
        covered[i] = bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > EPS_M)).any())
    return covered


def buried_fraction(
    held_vertices: np.ndarray,
    T_held: np.ndarray,
    receptacle_vertices: np.ndarray,
    receptacle_faces: np.ndarray,
    T_receptacle: np.ndarray,
) -> float:
    """Share of the held vertices (body frame) covered by the receptacle mesh from above."""
    v = np.asarray(held_vertices, float)
    if len(v) > MAX_POINTS:
        v = v[np.linspace(0, len(v) - 1, MAX_POINTS).astype(int)]
    world = _transform(np.asarray(T_held, float), v)
    tris = _transform(np.asarray(T_receptacle, float), np.asarray(receptacle_vertices, float))
    return float(_faces_above(world, tris[np.asarray(receptacle_faces, int)]).mean())


def object_mesh(obj) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame visual mesh of a scene object (its half-size box when it has no mesh)."""
    if getattr(obj, "mesh_vertices", None) is not None and getattr(obj, "mesh_faces", None) is not None:
        return np.asarray(obj.mesh_vertices, float), np.asarray(obj.mesh_faces, int)
    return box_mesh(obj.half_size_m)


def buried_fraction_of(held, receptacle, T_held: np.ndarray, T_receptacle: np.ndarray) -> float:
    hv, _ = object_mesh(held)
    rv, rf = object_mesh(receptacle)
    return buried_fraction(hv, T_held, rv, rf, T_receptacle)
