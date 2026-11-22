"""Twin asset package -> simulator object (Track BT, BT.8).

The package mesh frame has its origin at the bbox bottom centre, so ``pos_m`` is the position of that
point (the foot of the object). Visual mesh carries mass/inertia; CoACD pieces collide.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import trimesh

from stages.manipulation.mjcf_scene import BoxObject
from stages.reconstruction.objects.twin_schema import AssetSpec, load_asset_spec

TWIN_MIN_MASS_KG = 0.005


def asset_mass_kg(spec: AssetSpec, mesh: trimesh.Trimesh) -> float:
    """``physics.mass_kg`` as given, else density x solid volume (hull volume for open meshes)."""
    if spec.physics.mass_kg is not None:
        return float(spec.physics.mass_kg)
    volume = abs(mesh.volume) if mesh.is_watertight else mesh.convex_hull.volume
    return max(float(spec.physics.density_kg_m3 * volume), TWIN_MIN_MASS_KG)


def _texture(mesh: trimesh.Trimesh) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mesh.visual.kind != "texture" or mesh.visual.uv is None:
        return None, None
    image = getattr(mesh.visual.material, "baseColorTexture", None)
    if image is None:
        return None, None
    uv = np.asarray(mesh.visual.uv, np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # trimesh counts v from the image bottom; BoxObject (and glTF) from the top
    return uv, np.asarray(image.convert("RGB"), np.uint8)


def load_twin_asset(
    asset_dir: str | Path,
    name: str,
    pos_m: tuple[float, float, float],
    *,
    friction: float | None = None,
    mass_kg: float | None = None,
    static: bool = False,
    robot_collision: bool = True,
) -> BoxObject:
    """Build the simulator object of an asset; ``friction``/``mass_kg`` override the package prior (RSI)."""
    asset_dir = Path(asset_dir)
    spec = load_asset_spec(asset_dir / "asset.yaml")
    visual = trimesh.load(asset_dir / "visual.glb", force="mesh", process=False)
    pieces = []
    for f in sorted((asset_dir / "collision").glob("*.obj")):
        m = trimesh.load(f, force="mesh", process=False)
        pieces.append((np.asarray(m.vertices), np.asarray(m.faces)))
    uv, tex = _texture(visual)
    h, s, v = spec.colour_hsv[0]
    rgba = (*colorsys.hsv_to_rgb(h / 360.0, s, v), 1.0)
    mu = spec.physics.friction if friction is None else friction
    return BoxObject(
        name=name,
        half_size_m=tuple(x / 2 for x in spec.size_m),
        pos_m=pos_m,
        mass_kg=asset_mass_kg(spec, visual) if mass_kg is None else mass_kg,
        friction=(mu, 0.01, 0.001),
        rgba=rgba,
        mesh_vertices=np.asarray(visual.vertices),
        mesh_faces=np.asarray(visual.faces),
        mesh_uv=uv,
        mesh_texture=tex,
        collision_pieces=tuple(pieces),
        static=static,
        robot_collision=robot_collision,
    )
