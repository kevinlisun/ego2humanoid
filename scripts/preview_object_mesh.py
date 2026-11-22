"""Preview the sim object asset: the SAM 3 crop next to four orbit views of the textured mesh.

Usage: MUJOCO_GL=egl preview_object_mesh.py <clip_dir>   (needs object_crop.png, object_mesh_raw.glb, report)
Writes ``<clip_dir>/object_preview.png``.
"""

import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import trimesh

from stages.manipulation.mjcf_scene import BoxObject, add_object
from stages.reconstruction.objects.mesh_asset import build_mesh_asset

clip = Path(sys.argv[1])
report = json.loads((clip / "retarget_rgb.json").read_text())["perceived_object"]
extent = 2 * float(max(report["half_size_m"]))
raw = trimesh.load(clip / "object_mesh_raw.glb", force="mesh")
crop = cv2.cvtColor(cv2.imread(str(clip / "object_crop.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
asset = build_mesh_asset(raw.vertices, raw.faces, extent, np.array([1.0, 0.0]), crop)

spec = mujoco.MjSpec()
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[0.75, 0.7, 0.65, 1])
spec.worldbody.add_light(pos=[0.3, -0.3, 1.0], dir=[-0.3, 0.3, -1.0])
half_z = float(asset.vertices[:, 2].max() - asset.vertices[:, 2].min()) / 2
add_object(
    spec,
    BoxObject(
        "obj",
        (0.01, 0.01, half_z),
        (0.0, 0.0, half_z),
        rgba=(1, 1, 1, 1) if asset.texture is not None else (0.75, 0.75, 0.75, 1),
        mesh_vertices=asset.vertices,
        mesh_faces=asset.faces,
        mesh_uv=asset.uv,
        mesh_texture=asset.texture,
    ),
)
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 360, 360)
tiles = [cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB), (360, 360))]
for az in (0, 90, 180, 270):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, half_z]
    cam.distance, cam.azimuth, cam.elevation = 2.6 * max(extent, 0.05), az, -30
    renderer.update_scene(data, cam)
    tiles.append(renderer.render())
cv2.imwrite(str(clip / "object_preview.png"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
print(f"extent_m={extent:.3f} texture_iou={asset.texture_iou}")
