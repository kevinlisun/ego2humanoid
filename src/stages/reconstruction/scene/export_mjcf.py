"""Turn a reconstructed multi-object scene (per-object GLBs + poses) into sim objects.

Input is ``<scene_dir>/objects.json``::

    {"objects": [{"name": "cup", "glb": "cup.glb", "T_sim_object": [[4x4]],
                  "mass_kg": 0.2, "friction": [1.0, 0.01, 0.001]}]}

``T_sim_object`` places the GLB (metres) in the sim frame (z up, table top at ``table_top_z_m``).
The SimFoundry runner writes this manifest from stage-12 output; this module is the single
consumer, so the sim never depends on SimFoundry's own state format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from stages.manipulation.mjcf_scene import BoxObject


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene" if path.suffix.lower() in {".glb", ".gltf"} else None)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"no triangle mesh in {path}")
    return mesh


def _glb_image(path: Path) -> np.ndarray | None:
    """First embedded image of a GLB; trimesh drops it when KHR_materials_specular is present."""
    import io
    import struct

    from PIL import Image

    d = path.read_bytes()
    jlen = struct.unpack("<I", d[12:16])[0]
    meta = json.loads(d[20 : 20 + jlen])
    if not meta.get("images") or "bufferView" not in meta["images"][0]:
        return None
    view = meta["bufferViews"][meta["images"][0]["bufferView"]]
    start = 20 + jlen + 8 + view.get("byteOffset", 0)
    return np.asarray(Image.open(io.BytesIO(d[start : start + view["byteLength"]])).convert("RGB"), np.uint8)


def _mtl_image(path: Path) -> np.ndarray | None:
    """``map_Kd`` image of the OBJ's .mtl, for when trimesh could not resolve the relative path."""
    from PIL import Image

    mtl = path.with_suffix(".mtl")
    if not mtl.exists():
        return None
    for line in mtl.read_text().splitlines():
        if line.strip().startswith("map_Kd"):
            tex = (mtl.parent / line.split(maxsplit=1)[1].strip()).resolve()
            if tex.exists():
                return np.asarray(Image.open(tex).convert("RGB"), np.uint8)
    return None


def _texture(mesh: trimesh.Trimesh, path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-vertex uv (v from the image top, as MuJoCo expects) and the base-colour image, if present."""
    vis = mesh.visual
    uv = getattr(vis, "uv", None)
    mat = getattr(vis, "material", None)
    image = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    if image is not None:
        image = np.asarray(image.convert("RGB"), np.uint8)
    elif path.suffix.lower() == ".glb":
        image = _glb_image(path)
    elif path.suffix.lower() == ".obj":
        image = _mtl_image(path)
    if uv is None or image is None or len(uv) != len(mesh.vertices):
        return None, None
    uv = np.asarray(uv, float).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv, image


def load_scene_objects(scene_dir: str | Path, decompose: bool = False) -> tuple[BoxObject, ...]:
    """Body-frame meshes (origin at the world-frame bounding-box centre) with sim-frame positions.

    ``decompose`` adds CoACD convex collision pieces (slower, faithful for concave objects).
    Only the rotation/translation of ``T_sim_object`` is applied; the mesh is expected in metres.
    """
    root = Path(scene_dir)
    manifest = json.loads((root / "objects.json").read_text())
    out = []
    for entry in manifest["objects"]:
        mesh = _load_mesh(root / entry["glb"])
        T = np.asarray(entry["T_sim_object"], float)
        if T.shape != (4, 4):
            raise ValueError(f"{entry['name']}: T_sim_object must be 4x4")
        uv, image = _texture(mesh, root / entry["glb"])
        mesh.apply_transform(T)
        centre = (mesh.bounds[0] + mesh.bounds[1]) / 2
        verts = np.asarray(mesh.vertices) - centre
        half = tuple(float(x) for x in (mesh.bounds[1] - mesh.bounds[0]) / 2)
        pieces = ()
        if decompose:
            from stages.reconstruction.scene.collision import convex_pieces

            pieces = convex_pieces(verts, np.asarray(mesh.faces))
        out.append(
            BoxObject(
                name=entry["name"],
                half_size_m=half,
                pos_m=tuple(float(x) for x in centre),
                mass_kg=float(entry.get("mass_kg", 0.1)),
                friction=tuple(entry.get("friction", (1.0, 0.01, 0.001))),
                mesh_vertices=verts,
                mesh_faces=np.asarray(mesh.faces),
                collision_pieces=pieces,
                mesh_uv=uv,
                mesh_texture=image,
                rgba=(1.0, 1.0, 1.0, 1.0) if image is not None else (0.75, 0.75, 0.75, 1.0),
            )
        )
    names = [o.name for o in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate object names in objects.json")
    return tuple(out)
