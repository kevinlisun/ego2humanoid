"""Blender replay of a MuJoCo rollout from the robot's head camera (Track BT, BT.17); runs in the bpy venv.

    /tmp/bpy_venv/bin/python scripts/render_twin_sim.py <clip_dir> [--tag _bt] --frames 0 300 500     # PNGs
    /tmp/bpy_venv/bin/python scripts/render_twin_sim.py <clip_dir> [--tag _bt] --fps 10 --device gpu   # video
    /tmp/bpy_venv/bin/python scripts/render_twin_sim.py <clip_dir> --view human --out a.png   # real camera

``--appearance <json>`` (default ``review/twin_appearance.json`` if present) applies a ``TwinAppearance``
(shape, colour, light, table, walls; BT.23/BT.24); without one the look is the original. ``--view human``
renders the scene at record 0 from the real video camera of ``twin_scene/scene.json`` with the robot hidden.

Inputs: ``review/sim_state_sharpa<tag>.npz`` and ``review/sim_model/`` (``record_sim_state.py``), plus
``review/twin_scene/`` (``export_twin_scene.py``) for the textured scene objects. MuJoCo owns the physics;
this script only sets transforms (geom pose = body pose @ geom local pose) and renders with Cycles.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from render_twin_cam import D, mat  # noqa: E402
from twin_appearance_blender import (  # noqa: E402
    add_box,
    add_room,
    appearance_object,
    first_frame,
    projected_colors,
    visual_geometry,
)

from contracts.twin_appearance import TwinAppearance

D435_FOVY_DEG = 42.5
ROBOT_RGBA = (0.62, 0.64, 0.68, 1.0)


def enable_gpu(sc) -> str:
    """Cycles on the best GPU backend (OPTIX, then CUDA); falls back to CPU."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except (TypeError, RuntimeError):
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = "GPU"
            return f"{kind}: {', '.join(d.name for d in gpus)}"
    sc.cycles.device = "CPU"
    return "CPU (no GPU backend found)"


def mesh_object(name, verts, faces, material, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, float).tolist(), [], np.asarray(faces).tolist())
    if uv is not None:
        layer = me.uv_layers.new(name="uv")
        for poly in me.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices, strict=True):
                layer.data[li].uv = (float(uv[vi, 0]), 1.0 - float(uv[vi, 1]))
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


def primitive_object(g, material):
    kind, s = g["type"], g["size"]
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1)
        scale = (2 * s[0], 2 * s[1], 2 * s[2])
    elif kind in ("cylinder", "capsule"):
        bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=1)
        scale = (s[0], s[0], 2 * s[1] + (2 * s[0] if kind == "capsule" else 0))
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
        scale = (s[0],) * 3
    elif kind == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
        scale = tuple(s[:3])
    else:
        return None
    ob = bpy.context.object
    ob.name = g["name"]
    ob.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    ob.data.materials.append(material)
    return ob


def local_matrix(g) -> Matrix:
    return Matrix.Translation(g["pos"]) @ Quaternion(g["quat"]).to_matrix().to_4x4()


def object_geometry(clip, so, oa, local, pose0, twin):
    """Blender object of a scene object under appearance ``oa``; projected colours use the clip's frame 0."""
    review = clip / "review"
    d = np.load(review / "twin_scene" / so["mesh"])
    v, f = visual_geometry(oa, d["vertices"], d["faces"])
    if oa.shape == "mesh" and oa.texture == "atlas":
        tex = d["texture"] if "texture" in d.files else None
        uv = d["uv"] if "uv" in d.files else None
        return mesh_object(so["body"], v, f, mat(so["body"], so["rgba"], oa.roughness, tex), uv)
    corners = None
    if oa.texture == "projected":
        world = (pose0 @ local) @ np.c_[v, np.ones(len(v))].T
        corners = projected_colors(
            world[:3].T,
            f,
            np.array(twin["camera_T_sim"]),  # T_sim_camera
            twin["focal_px"],
            twin["image_wh"],
            first_frame(clip / "video.mp4"),
            oa.color_rgb,
        )
    return appearance_object(so["body"], v, f, oa, corners)


def build(clip: Path, tag: str, app: TwinAppearance, human: bool):
    review = clip / "review"
    model = json.loads((review / "sim_model" / "sim_model.json").read_text())
    meshes = np.load(review / "sim_model" / "sim_model_meshes.npz")
    state = np.load(review / f"sim_state_sharpa{tag}.npz")
    twin = json.loads((review / "twin_scene" / "scene.json").read_text())
    scene_obj = {o["body"]: o for o in twin["objects"]}
    body_idx = {str(n): i for i, n in enumerate(state["body_names"])}
    robot_mat = mat("robot", ROBOT_RGBA, 0.45)
    items = []
    for g in model["geoms"]:
        if g["rgba"][3] < 0.05 or g["body"] == "world":
            continue
        body, so = g["body"], scene_obj.get(g["body"])
        if human and so is None:
            continue
        own_mesh = so is not None and g["name"] == f"{body}_geom"
        # The twin_scene meshes are in the body frame; the compiled MuJoCo geom pose only re-aligns MuJoCo's own
        # (inertia-aligned) copy of the mesh, so applying it to these vertices would rotate the object wrongly.
        local = Matrix.Identity(4) if own_mesh else local_matrix(g)
        if own_mesh and so["name"] in app.objects:
            q = Quaternion(state["xquat"][0, body_idx[body]].tolist()).to_matrix().to_4x4()
            pose0 = np.array(Matrix.Translation(state["xpos"][0, body_idx[body]].tolist()) @ q)
            ob = object_geometry(clip, so, app.objects[so["name"]], np.array(local), pose0, twin)
        elif own_mesh:
            d = np.load(review / "twin_scene" / so["mesh"])
            tex = d["texture"] if "texture" in d.files else None
            m = mat(body, so["rgba"], 0.6, tex)
            ob = mesh_object(body, d["vertices"], d["faces"], m, d["uv"] if "uv" in d.files else None)
        elif g["type"] == "mesh":
            m = robot_mat if body not in scene_obj else mat(g["name"], g["rgba"], 0.5)
            ob = mesh_object(g["name"], meshes[f"{g['mesh']}_v"], meshes[f"{g['mesh']}_f"], m)
        else:
            ob = primitive_object(g, mat(g["name"], g["rgba"], 0.6))
        if ob is not None:
            items.append((ob, body_idx[body], local))
    return state, items, twin


def pose_frame(state, items, i):
    for ob, b, local in items:
        p, q = state["xpos"][i, b], state["xquat"][i, b]
        ob.matrix_world = Matrix.Translation(p.tolist()) @ Quaternion(q.tolist()).to_matrix().to_4x4() @ local


def camera_matrix(state, i, name="d435") -> Matrix:
    c = [str(n) for n in state["camera_names"]].index(name)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = state["cam_mat"][i, c], state["cam_pos"][i, c]
    return Matrix(M.tolist())  # MuJoCo and Blender cameras both look along -z with +y up


def human_camera(cam, twin, width):
    """Real camera; ``camera_T_sim`` in scene.json is T_sim_camera (OpenCV axes), hence the axis flip."""
    w, h = twin["image_wh"]
    cam.data.angle = 2 * math.atan(h / 2 / twin["focal_px"])
    cam.matrix_world = Matrix(twin["camera_T_sim"]) @ D
    return width, round(width * h / w)


def setup_world(sc, samples, width, app, twin, show_table):
    sc.render.engine, sc.cycles.samples = "CYCLES", samples
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 3, 2, 1
    sc.view_settings.view_transform = "Standard"
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = width, width * 9 // 16
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    sc.world = world
    bg = world.node_tree.nodes["Background"]
    light = app.light
    bg.inputs[0].default_value, bg.inputs[1].default_value = (*light.world_rgb, 1), light.world_strength
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = light.sun_energy
    sun.rotation_euler = (math.radians(90 - light.sun_elevation_deg), math.radians(light.sun_azimuth_deg), 0)
    bpy.context.collection.objects.link(sun)
    bpy.ops.mesh.primitive_plane_add(size=20)  # floor at z = 0 (the robot stands on it)
    bpy.context.object.data.materials.append(mat("floor", (0.55, 0.5, 0.45, 1), 0.8))
    t = twin["table"]
    if show_table:
        top, c, h = t["table_top_z_m"], t["table_center_xy_m"], t["table_half_size_xy_m"]
        add_box(
            "table", (*c, top - 0.02), (2 * h[0], 2 * h[1], 0.04), app.table.color_rgb, app.table.roughness
        )
    if app.room.walls:
        add_room(t["table_center_xy_m"], 2.5, 2.6, app.room.wall_rgb)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit, cam.data.angle, cam.data.clip_start = "VERTICAL", math.radians(D435_FOVY_DEG), 0.02
    bpy.context.collection.objects.link(cam)
    sc.camera = cam
    return cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--tag", default="_bt")
    ap.add_argument("--frames", type=int, nargs="*", help="record indices (PNGs to /tmp/twinsim_<i>.png)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument(
        "--out", default="twin_sim_bt.mp4", help="video in review/, or the PNG path for --view human"
    )
    ap.add_argument("--view", choices=("robot", "human"), default="robot")
    ap.add_argument(
        "--appearance", type=Path, help="TwinAppearance JSON (default: review/twin_appearance.json)"
    )
    a = ap.parse_args()
    appearance_path = a.appearance or a.clip / "review" / "twin_appearance.json"
    loaded = appearance_path.exists()
    app = TwinAppearance.model_validate_json(appearance_path.read_text()) if loaded else TwinAppearance()
    human = a.view == "human"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    state, items, twin = build(a.clip, a.tag, app, human)
    sc = bpy.context.scene
    cam = setup_world(sc, a.samples, a.width, app, twin, human or loaded)
    if human:
        sc.render.resolution_x, sc.render.resolution_y = human_camera(cam, twin, a.width)
        print("device:", enable_gpu(sc) if a.device == "gpu" else "CPU", flush=True)
        pose_frame(state, items, 0)
        sc.render.filepath = str(a.out)
        bpy.ops.render.render(write_still=True)
        print("wrote", a.out)
        return
    print("device:", enable_gpu(sc) if a.device == "gpu" else "CPU", flush=True)
    t = state["t"]
    n_total = len(t)
    if a.frames:
        picks = a.frames
    else:
        picks = sorted({int(np.searchsorted(t, x)) for x in np.arange(t[0], t[-1], 1.0 / a.fps)})
        picks = [min(p, n_total - 1) for p in picks]
    box = st = None
    if not a.frames:
        import av
        from PIL import Image

        box = av.open(str(a.clip / "review" / a.out), "w")
        st = box.add_stream("libx264", rate=a.fps)
        st.width, st.height, st.pix_fmt = sc.render.resolution_x, sc.render.resolution_y, "yuv420p"
    for k, i in enumerate(picks):
        pose_frame(state, items, i)
        cam.matrix_world = camera_matrix(state, i)
        path = Path("/tmp") / f"twinsim_{i}.png"
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        if box:
            for pkt in st.encode(av.VideoFrame.from_image(Image.open(path).convert("RGB"))):
                box.mux(pkt)
        print(f"frame {k + 1}/{len(picks)} (record {i}, t={t[i]:.2f}s)", flush=True)
    if box:
        for pkt in st.encode():
            box.mux(pkt)
        box.close()
        print("wrote", a.clip / "review" / a.out)


if __name__ == "__main__":
    main()
