"""Blender-side realisation of ``TwinAppearance`` for ``render_twin_sim.py`` (BT.24); runs in the bpy venv.

Object geometry is built in the *mesh frame* of ``twin_scene/<obj>.npz`` (the frame of ``half_size_m`` in
``scene.json``); the visual yaw/offset are baked into the vertices, so the MuJoCo pose is untouched.
"""

import math
from pathlib import Path

import bpy
import numpy as np
from render_twin_cam import mat

CYL_SEGMENTS = 48
# Corner order is x + 2y + 4z over (-1, +1); every quad winds counter-clockwise seen from outside.
BOX_QUADS = ((0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6))
BOX_CORNERS = np.array(
    [[(i & 1) * 2 - 1, (i >> 1 & 1) * 2 - 1, (i >> 2 & 1) * 2 - 1] for i in range(8)], float
)


def box_geometry(centre, size):
    v = np.asarray(centre, float) + 0.5 * np.asarray(size, float) * BOX_CORNERS
    return v, [list(q) for q in BOX_QUADS]


def cylinder_geometry(centre, size):
    """Elliptic cylinder along z whose bounding box is ``size``."""
    a = np.linspace(0, 2 * math.pi, CYL_SEGMENTS, endpoint=False)
    ring = np.stack([0.5 * size[0] * np.cos(a), 0.5 * size[1] * np.sin(a)], 1)
    lo, hi = -0.5 * size[2], 0.5 * size[2]
    v = np.concatenate([np.c_[ring, np.full(len(a), lo)], np.c_[ring, np.full(len(a), hi)]]) + centre
    n = CYL_SEGMENTS
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return v, [*faces, list(range(n - 1, -1, -1)), list(range(n, 2 * n))]


def tray_geometry(centre, size, wall):
    """Open box: a floor slab of thickness ``wall`` plus four walls of thickness ``wall``."""
    c, s = np.asarray(centre, float), np.asarray(size, float)
    w = min(wall, 0.5 * s[0], 0.5 * s[1], s[2])
    parts = [(c + [0, 0, w / 2 - s[2] / 2], [s[0], s[1], w])]
    wall_z = c[2] + w / 2
    wall_h = s[2] - w
    for sx in (-1, 1):
        parts.append(((c[0] + sx * (s[0] - w) / 2, c[1], wall_z), (w, s[1], wall_h)))
        parts.append(((c[0], c[1] + sx * (s[1] - w) / 2, wall_z), (s[0] - 2 * w, w, wall_h)))
    verts, faces = [], []
    for pc, ps in parts:
        v, f = box_geometry(pc, ps)
        faces += [[i + len(verts) for i in q] for q in f]
        verts += list(v)
    return np.array(verts), faces


def visual_geometry(app, verts, faces):
    """Vertices/faces of the object in its mesh frame after applying the appearance."""
    lo, hi = verts.min(0), verts.max(0)
    centre, bbox = (lo + hi) / 2, hi - lo
    size = np.asarray(app.size_m if app.size_m is not None else bbox, float)
    if app.shape == "mesh":
        v, f = centre + (verts - centre) * app.scale, [list(p) for p in faces]
    elif app.shape == "box":
        v, f = box_geometry(centre, size)
    elif app.shape == "cylinder":
        v, f = cylinder_geometry(centre, size)
    else:
        v, f = tray_geometry(centre, size, app.wall_m)
    c, s = math.cos(math.radians(app.yaw_deg)), math.sin(math.radians(app.yaw_deg))
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return centre + np.asarray(app.offset_m) + (v - centre) @ rz.T, f


def first_frame(video: Path) -> np.ndarray:
    import av

    with av.open(str(video)) as box:
        return next(box.decode(video=0)).to_ndarray(format="rgb24")


def projected_colors(world_verts, faces, T_sim_camera, focal_px, image_wh, rgb, fallback):
    """Per-corner sRGB (n_corners, 3) sampled from ``rgb`` (frame 0); ``fallback`` where not visible.

    A corner is visible when its vertex projects inside the image and its face looks at the camera.
    Hand occlusion is not modelled.
    """
    R, eye = T_sim_camera[:3, :3], T_sim_camera[:3, 3]
    cam = (world_verts - eye) @ R
    z = np.maximum(cam[:, 2], 1e-6)
    scale = rgb.shape[1] / image_wh[0]
    u = (focal_px * cam[:, 0] / z + image_wh[0] / 2) * scale
    v = (focal_px * cam[:, 1] / z + image_wh[1] / 2) * scale
    inside = (cam[:, 2] > 0) & (u >= 0) & (u < rgb.shape[1] - 1) & (v >= 0) & (v < rgb.shape[0] - 1)
    cols = rgb[
        np.clip(v.round().astype(int), 0, rgb.shape[0] - 1),
        np.clip(u.round().astype(int), 0, rgb.shape[1] - 1),
    ]
    cols = cols / 255.0
    out = []
    for f in faces:
        p = world_verts[f]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        facing = float(n @ (eye - p.mean(0))) > 0
        out += [cols[i] if facing and inside[i] else np.asarray(fallback, float) for i in f]
    return np.array(out)


def appearance_object(name, verts, faces, app, corner_srgb=None):
    """Blender object from visual geometry; ``corner_srgb`` (one colour per face corner) overrides colour."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts).tolist(), [], faces)
    me.shade_flat()
    material = mat(name, (*app.color_rgb, 1.0), app.roughness)
    if corner_srgb is not None:
        layer = me.color_attributes.new("proj", "FLOAT_COLOR", "CORNER")
        rgba = np.c_[np.asarray(corner_srgb) ** 2.2, np.ones(len(corner_srgb))]
        layer.data.foreach_set("color", rgba.astype(np.float32).ravel())
        nodes = material.node_tree.nodes
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "proj"
        material.node_tree.links.new(attr.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


def add_box(name, centre, size, rgb, roughness):
    v, f = box_geometry(centre, size)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat(name, (*rgb, 1.0), roughness))
    return ob


def add_room(table_xy, half_m, height_m, rgb):
    """Four thin walls enclosing the table area."""
    t = 0.05
    for k, (dx, dy, sx, sy) in enumerate(
        [(1, 0, t, 2 * half_m), (-1, 0, t, 2 * half_m), (0, 1, 2 * half_m, t), (0, -1, 2 * half_m, t)]
    ):
        c = (table_xy[0] + dx * half_m, table_xy[1] + dy * half_m, height_m / 2)
        add_box(f"wall{k}", c, (sx, sy, height_m), rgb, 0.9)
