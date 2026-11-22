"""Blender (bpy) renderer of an exported clip scene (Track BT, BT.10). Runs in a separate bpy venv.

    /tmp/bpy_venv/bin/python scripts/render_twin_blender.py <clip>/review/twin_scene --view ego|overview \\
        --out <clip>/review/twin_render_ego_bt.mp4 [--stride 3] [--samples 16] [--width 960]
"""

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _material(name: str, rgba, tex: np.ndarray | None = None, roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    if tex is None:
        bsdf.inputs["Base Color"].default_value = rgba
        return mat
    h, w = tex.shape[:2]
    img = bpy.data.images.new(f"{name}_tex", w, h)
    rgba_f = np.concatenate([tex[::-1] / 255.0, np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba_f.ravel())
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _mesh_object(name: str, data: dict, rgba):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(data["vertices"].tolist(), [], data["faces"].tolist())
    mesh.update()
    tex = data["texture"] if "texture" in data else None
    if "uv" in data:
        layer = mesh.uv_layers.new(name="uv")
        uv = data["uv"]
        for poly in mesh.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices, strict=True):
                u, v = float(uv[vi, 0]), float(uv[vi, 1])
                layer.data[li].uv = (u, 1.0 - v)  # BoxObject v counts from the top
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_material(name, rgba, tex))
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def _box(name: str, centre, size, rgba):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
    obj = bpy.context.active_object
    obj.name, obj.scale = name, size
    obj.data.materials.append(_material(name, rgba))
    return obj


def _sphere(name: str, radius: float, rgba):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=16, ring_count=8)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(_material(name, rgba, roughness=0.5))
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def build(scene_dir: Path):
    meta = json.loads((scene_dir / "scene.json").read_text())
    traj = np.load(scene_dir / "traj.npz")
    t = meta["table"]
    cx, cy = t["table_center_xy_m"]
    hx, hy = t["table_half_size_xy_m"]
    top = t["table_top_z_m"]
    _box("table", (cx, cy, top - 0.02), (2 * hx, 2 * hy, 0.04), (0.55, 0.42, 0.30, 1))
    _box("floor", (cx, cy, top - 0.80), (6, 6, 0.02), (0.35, 0.36, 0.38, 1))
    held = None
    for o in meta["objects"]:
        data = dict(np.load(scene_dir / o["mesh"]))
        obj = _mesh_object(o["name"], data, o["rgba"])
        obj.location = o["pos_m"]
        if o["kind"] == "held":
            held = obj
    hands = [_sphere(f"wrist{i}", 0.03, c) for i, c in enumerate([(0.9, 0.3, 0.2, 1), (0.2, 0.5, 0.9, 1)])]
    tips = [[_sphere(f"tip{h}{f}", 0.009, (0.95, 0.75, 0.6, 1)) for f in range(5)] for h in range(2)]

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(35), math.radians(10), math.radians(25))
    bpy.context.collection.objects.link(sun)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.75, 0.78, 0.85, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    bpy.context.scene.world = world
    return meta, traj, held, hands, tips, (cx, cy, top)


def make_camera(view: str, meta: dict, centre, width: int):
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    w, h = meta["image_wh"]
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = meta["focal_px"] * 36.0 / w
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = width, int(width * h / w)
    if view == "ego":
        # OpenCV camera (x right, y down, z forward) -> Blender camera (x right, y up, z backward)
        cam.matrix_world = Matrix(meta["camera_T_sim"]) @ Matrix.Diagonal((1, -1, -1, 1))
    else:
        cx, cy, top = centre
        cam.location = (cx + 0.15, cy - 0.95, top + 0.75)
        target = Vector((cx, cy, top + 0.03))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = 28.0
    return cam


def render(args) -> None:
    import av

    _clear()
    scene_dir = Path(args.scene)
    meta, traj, held, hands, tips, centre = build(scene_dir)
    make_camera(args.view, meta, centre, args.width)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.render.image_settings.file_format = "PNG"
    n = len(traj["object_pos_m"])
    frames = list(range(0, n, args.stride))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp") / f"twin_{out.stem}.png"
    box = av.open(str(out), "w")
    st = box.add_stream("libx264", rate=args.fps)
    st.width, st.height, st.pix_fmt = sc.render.resolution_x, sc.render.resolution_y, "yuv420p"
    from PIL import Image

    for k, i in enumerate(frames):
        R = np.eye(4)
        R[:3, :3], R[:3, 3] = traj["object_rot"][i], traj["object_pos_m"][i]
        held.matrix_world = Matrix(R.tolist())
        for h in range(2):
            hands[h].location = traj["wrist_m"][i, h]
            for f in range(5):
                tips[h][f].location = traj["fingertip_m"][i, h, f]
        sc.render.filepath = str(tmp)
        bpy.ops.render.render(write_still=True)
        frame = av.VideoFrame.from_image(Image.open(tmp).convert("RGB"))
        for pkt in st.encode(frame):
            box.mux(pkt)
        print(f"frame {k + 1}/{len(frames)}", flush=True)
    for pkt in st.encode():
        box.mux(pkt)
    box.close()
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--view", choices=["ego", "overview"], default="ego")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--width", type=int, default=960)
    render(ap.parse_args())
