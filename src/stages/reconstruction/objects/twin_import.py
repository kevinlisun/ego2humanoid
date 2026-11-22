"""Import an authored mesh (Blender export) into a twin asset package (Track BT, BT.3).

``.glb/.gltf`` files are Y-up by convention (Blender's exporter converts), ``.obj/.stl/.ply`` keep the
author's axes (Blender: Z-up). The package stores Z-up metres with the origin at the bbox bottom centre.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from stages.reconstruction.objects.twin_schema import (
    MAX_COLLISION_PIECES,
    AssetSpec,
    RestPose,
    sha256_file,
    validate_asset_dir,
)

# Rotation taking Y-up coordinates to Z-up (x stays, y -> z, z -> -y).
_Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
MIN_WALL_M = 0.0005


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load one mesh, merging scene nodes; keeps UVs/texture when the source has a single material."""
    path = Path(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path} holds no triangle mesh")
    if path.suffix.lower() in (".glb", ".gltf"):
        mesh.apply_transform(_Y_UP_TO_Z_UP)
    return mesh


def normalise_mesh(
    mesh: trimesh.Trimesh, size_m: tuple[float, float, float] | None = None
) -> trimesh.Trimesh:
    """Z-up metres, origin at the bbox bottom centre; non-uniformly scaled to ``size_m`` when given."""
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    if size_m is not None:
        extent = np.maximum(hi - lo, 1e-9)
        mesh.apply_scale(np.asarray(size_m) / extent)
        lo, hi = mesh.bounds
    mesh.apply_translation(-np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]))
    return mesh


def mesh_report(mesh: trimesh.Trimesh) -> dict:
    """Quality checks that decide how the asset is simulated (watertight -> CoACD, else hull)."""
    extent = mesh.extents
    report = {
        "size_m": [float(x) for x in extent],
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "volume_m3": float(abs(mesh.volume)) if mesh.is_watertight else float(mesh.convex_hull.volume),
        "hull_volume_m3": float(mesh.convex_hull.volume),
        "has_texture": bool(getattr(mesh.visual, "uv", None) is not None and mesh.visual.kind == "texture"),
        "warnings": [],
    }
    if not report["watertight"]:
        report["warnings"].append("not watertight: collision falls back to the convex hull")
    if min(extent) < MIN_WALL_M:
        report["warnings"].append(f"thinner than {MIN_WALL_M * 1000:.1f} mm along one axis")
    if report["watertight"] and report["volume_m3"] < 0.02 * report["hull_volume_m3"]:
        report["warnings"].append("volume under 2% of the hull: thin shell, check the mass prior")
    return report


def collision_pieces(
    mesh: trimesh.Trimesh, threshold: float = 0.05, max_pieces: int = 16
) -> list[trimesh.Trimesh]:
    from stages.reconstruction.scene.collision import convex_pieces

    pieces = convex_pieces(mesh.vertices, mesh.faces, threshold=threshold, max_pieces=max_pieces)
    return [trimesh.Trimesh(v, f, process=False) for v, f in pieces][:MAX_COLLISION_PIECES]


def import_twin(
    src: str | Path,
    out_dir: str | Path,
    *,
    asset_id: str,
    category: str,
    size_m: tuple[float, float, float] | None,
    colour_hsv: list[tuple[float, float, float]],
    physics: dict,
    licence: str,
    author: str,
    synonyms: list[str] | None = None,
    hollow_inner_floor_z_m: float | None = None,
    rest_poses: list[dict] | None = None,
    max_pieces: int = 16,
) -> dict:
    """Write ``out_dir`` (visual.glb, collision/*.obj, asset.yaml, import_report.json); return the report."""
    out_dir = Path(out_dir)
    mesh = normalise_mesh(load_mesh(src), size_m)
    report = mesh_report(mesh)
    (out_dir / "collision").mkdir(parents=True, exist_ok=True)
    mesh.export(out_dir / "visual.glb")
    for old in (out_dir / "collision").glob("*.obj"):
        old.unlink()
    pieces = collision_pieces(mesh, max_pieces=max_pieces)
    for i, p in enumerate(pieces):
        p.export(out_dir / "collision" / f"piece_{i:02d}.obj")
    report["collision_pieces"] = len(pieces)

    rest = [RestPose(**r) for r in rest_poses] if rest_poses else [RestPose(down_face="-z", probability=1.0)]
    files = ["visual.glb", *(f"collision/{p.name}" for p in sorted((out_dir / "collision").glob("*.obj")))]
    spec = AssetSpec(
        id=asset_id,
        category=category,
        synonyms=synonyms or [],
        colour_hsv=colour_hsv,
        size_m=tuple(float(x) for x in mesh.extents),
        rest_poses=rest,
        physics=physics,
        hollow={"inner_floor_z_m": hollow_inner_floor_z_m} if hollow_inner_floor_z_m else None,
        licence=licence,
        author=author,
        sha256={f: sha256_file(out_dir / f) for f in files},
    )
    (out_dir / "asset.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    report["validation"] = validate_asset_dir(out_dir)
    (out_dir / "import_report.json").write_text(json.dumps(report, indent=2))
    return report
