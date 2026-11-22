"""Blender renderer of the camera-frame twin scene (Track BT, BT.10); runs in the bpy venv.

    /tmp/bpy_venv/bin/python scripts/render_twin_cam.py <clip_dir> --video-frames 0 190 287          # PNGs
    /tmp/bpy_venv/bin/python scripts/render_twin_cam.py <clip_dir> --stride 3 --out twin_cam_bt.mp4  # video

Requires ``build_twin_scene_cam.py`` and ``export_twin_scene.py`` outputs (small-object meshes).
Assets that Blender authors itself (stands, pen) are procedural placeholders for the authored library.
"""

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

D = Matrix.Diagonal((1, -1, -1, 1))  # OpenCV camera frame -> Blender camera frame
MANO_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
SKIN = (0.86, 0.62, 0.50, 1)


def lin(rgba):
    """sRGB (sampled from the video) -> linear for Blender."""
    return tuple(float(c) ** 2.2 for c in rgba[:3]) + (1.0,)


def mat(name, rgba, rough=0.55, tex=None):
    rgba = lin(rgba)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    if tex is None:
        b.inputs["Base Color"].default_value = rgba
        return m
    h, w = tex.shape[:2]
    img = bpy.data.images.new(f"{name}_t", w, h)
    px = np.concatenate([tex[::-1] / 255.0, np.ones((h, w, 1))], 2).astype(np.float32)
    img.pixels.foreach_set(px.ravel())
    n = m.node_tree.nodes.new("ShaderNodeTexImage")
    n.image = img
    m.node_tree.links.new(n.outputs["Color"], b.inputs["Base Color"])
    return m


def cam_matrix(R: np.ndarray, p: np.ndarray) -> Matrix:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return D @ Matrix(T.tolist())


def add(prim, name, material, **kw):
    getattr(bpy.ops.mesh, prim)(**kw)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(material)
    for poly in o.data.polygons:
        poly.use_smooth = prim != "primitive_cube_add"
    return o


def box_local(name, material, centre, size):
    o = add("primitive_cube_add", name, material, size=1.0)
    o.location, o.scale = centre, size
    return o


class Table:
    """Table-aligned frame: x = a e1 + b e2 + (h + plane_off) up."""

    def __init__(self, meta):
        self.B = np.array(meta["basis"])
        self.off = meta["plane_off"]

    def cam(self, abh):
        a, b, h = abh
        return self.B.T @ np.array([a, b, h + self.off])

    def frame(self, abh, yaw=0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return cam_matrix(self.B.T @ Rz, self.cam(abh))


def parent_all(parent, objs):
    for o in objs:
        o.parent = parent


def square_stand(o, T):
    root = bpy.data.objects.new(o["name"], None)
    bpy.context.collection.objects.link(root)
    root.matrix_world = T.frame((*o["centre_ab"], 0.0), o["yaw"])
    m = mat("white_stand", (0.92, 0.92, 0.90, 1), 0.45)
    L, W, top = o["size_ab"][0], o["size_ab"][1], o["top_h"]
    slab, leg = 0.032, 0.036
    parts = [box_local("slab", m, (0, 0, top - slab / 2), (L, W, slab))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * (L / 2 - leg / 2 - 0.004), sy * (W / 2 - leg / 2 - 0.004), (top - slab) / 2)
            parts.append(box_local("leg", m, c, (leg, leg, top - slab)))
    parent_all(root, parts)
    for p in parts:
        p.matrix_parent_inverse = Matrix.Identity(4)
    return root


def round_stand(o, T):
    root = bpy.data.objects.new(o["name"], None)
    bpy.context.collection.objects.link(root)
    root.matrix_world = T.frame((*o["centre_ab"], 0.0), o["yaw"])
    m = mat("white_round", (0.93, 0.93, 0.91, 1), 0.45)
    r, top, slab = max(o["size_ab"]) / 2, o["top_h"], 0.014
    parts = [
        add("primitive_cylinder_add", "disc", m, radius=r, depth=slab, location=(0, 0, top - slab / 2)),
        add(
            "primitive_cylinder_add",
            "stem",
            m,
            radius=0.022,
            depth=top - slab - 0.01,
            location=(0, 0, 0.01 + (top - slab - 0.01) / 2),
        ),
        add("primitive_cylinder_add", "foot", m, radius=0.055, depth=0.012, location=(0, 0, 0.006)),
    ]
    parent_all(root, parts)
    for p in parts:
        p.matrix_parent_inverse = Matrix.Identity(4)
    return root


def recon_object(o, mesh_npz, T):
    d = np.load(mesh_npz)
    v = d["vertices"].astype(float)
    ext = v.max(0) - v.min(0)
    k = max(o["size_ab"]) / max(ext[0], ext[1])
    v = (
        v
        - np.array([(v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2, v[:, 2].min()])
    ) * k
    me = bpy.data.meshes.new(o["name"])
    me.from_pydata(v.tolist(), [], d["faces"].tolist())
    if "uv" in d.files:
        layer = me.uv_layers.new(name="uv")
        for poly in me.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices, strict=True):
                layer.data[li].uv = (float(d["uv"][vi, 0]), 1.0 - float(d["uv"][vi, 1]))
    ob = bpy.data.objects.new(o["name"], me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(
        mat(o["name"], (0.6, 0.5, 0.4, 1), 0.6, d["texture"] if "texture" in d.files else None)
    )
    ob.matrix_world = T.frame((*o["centre_ab"], 0.0), o["yaw"])
    return ob


class Hands:
    def __init__(self, hands):
        self.J, self.valid, self.cam_T = hands["joints3d"], hands["valid"], hands["cam_T"]
        sk = mat("skin", SKIN, 0.6)
        self.joints = [
            [
                add(
                    "primitive_uv_sphere_add",
                    f"j{h}_{i}",
                    sk,
                    radius=0.0075 if i else 0.012,
                    segments=10,
                    ring_count=6,
                )
                for i in range(21)
            ]
            for h in range(2)
        ]
        self.bones = [
            [
                add("primitive_cylinder_add", f"b{h}_{k}", sk, radius=0.006, depth=1.0, vertices=10)
                for k in range(len(MANO_BONES))
            ]
            for h in range(2)
        ]

    def world(self, i):
        """Joints of frame i in the keyframe camera frame (the static world)."""
        T = self.cam_T[i]
        return self.J[i] @ T[:3, :3].T + T[:3, 3]

    def pose(self, i):
        Jw = self.world(i)
        for h in range(2):
            vis = bool(self.valid[i, h])
            P = Jw[h]
            for j, o in enumerate(self.joints[h]):
                o.hide_render = not vis
                o.matrix_world = D @ Matrix.Translation(Vector(P[j]))
            for (a, b), o in zip(MANO_BONES, self.bones[h], strict=True):
                o.hide_render = not vis
                d = Vector(P[b] - P[a])
                L = max(d.length, 1e-4)
                q = d.to_track_quat("Z", "Y")
                M = (
                    Matrix.Translation(Vector((P[a] + P[b]) / 2))
                    @ q.to_matrix().to_4x4()
                    @ Matrix.Diagonal((1, 1, L, 1))
                )
                o.matrix_world = D @ M


class Pen:
    def __init__(self, meta, T):
        p = meta["pen"]
        self.T, self.meta = T, p
        self.start = np.array(p["start_abh"])
        L, r = p["length_m"], p["radius_m"]
        self.obj = add(
            "primitive_cylinder_add", "pen", mat("pen_blue", (0.05, 0.08, 0.5, 1), 0.35), radius=r, depth=L
        )
        tip = add(
            "primitive_cone_add", "pen_tip", mat("pen_tip", (0.02, 0.02, 0.02, 1), 0.4), radius1=r, depth=0.02
        )
        tip.parent = self.obj
        tip.location = (0, 0, -L / 2 - 0.01)
        tip.rotation_euler = (0, 0, 0)
        self.hold = p["hold_s"]
        self.last = self.start

    def pose(self, t, Jw, valid, i):
        h0, h1 = self.hold
        grip = (Jw[0, 4] + Jw[0, 8]) / 2
        gt = np.einsum("ij,j->i", self.T.B, grip) - np.array([0, 0, self.T.off])  # table coords
        if h0 <= t <= h1 and valid[i, 0]:
            self.last = gt
            pos, w = gt, 1.0
        elif t > h1:
            pos, w = self.last, 1.0
        else:
            pos, w = self.start, 0.0
        yaw = -0.3 + 0.8 * w
        c, s = math.cos(yaw), math.sin(yaw)
        # lying along the table-frame direction (cos yaw, sin yaw, 0)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # cylinder z axis -> table y axis
        self.obj.matrix_world = cam_matrix(self.T.B.T @ Rz @ Rx, self.T.cam(pos))


def build(clip: Path):
    sdir = clip / "review" / "twin_scene_cam"
    meta = json.loads((sdir / "scene_cam.json").read_text())
    T = Table(meta)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    up = T.B[2]

    def slab(name, centre_cam, normal_cam, rgb, size=6.0):
        o = add("primitive_plane_add", name, mat(name, (*rgb, 1), 0.7), size=size)
        z = Vector(normal_cam)
        R = z.to_track_quat("Z", "Y").to_matrix().to_4x4()
        o.matrix_world = D @ Matrix.Translation(Vector(centre_cam)) @ R
        return o

    slab("table", up * T.off, up, meta["table_rgb"])
    for k, w in enumerate(meta["walls"]):
        slab(f"wall{k}", w["centre"], w["normal"], w["rgb"], size=8.0)
    for o in meta["objects"]:
        if o["name"].startswith("white_square"):
            square_stand(o, T)
        elif o["name"].startswith("white_round"):
            round_stand(o, T)
        else:
            recon_object(o, clip / "review" / "twin_scene" / f"scene_{o['name']}.npz", T)
    hands = Hands(np.load(sdir / "hands.npz"))
    pen = Pen(meta, T)

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 1.2
    sun.matrix_world = Matrix.Rotation(math.radians(35), 4, "X")
    bpy.context.collection.objects.link(sun)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 0.97, 0.92, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    bpy.context.scene.world = world

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    W, H = meta["image_wh"]
    cam.data.sensor_fit, cam.data.sensor_width = "HORIZONTAL", 36.0
    cam.data.lens = meta["focal_px"] * 36.0 / W
    return meta, T, hands, pen, cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--video-frames", type=int, nargs="*")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default="twin_cam_bt.mp4")
    a = ap.parse_args()
    meta, T, hands, pen, cam = build(a.clip)
    sc = bpy.context.scene
    sc.render.engine, sc.cycles.device, sc.cycles.samples = "CYCLES", "CPU", a.samples
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 3, 2, 1
    sc.view_settings.view_transform = "Standard"
    sc.cycles.use_denoising = True
    W, H = meta["image_wh"]
    sc.render.resolution_x, sc.render.resolution_y = a.width, int(a.width * H / W)
    n = len(hands.J)
    frames = a.video_frames if a.video_frames else list(range(0, n, a.stride))
    review = a.clip / "review"
    from PIL import Image

    box = st = None
    if not a.video_frames:
        import av

        box = av.open(str(review / a.out), "w")
        st = box.add_stream("libx264", rate=int(round(meta["fps"] / a.stride)))
        st.width, st.height, st.pix_fmt = sc.render.resolution_x, sc.render.resolution_y, "yuv420p"
    for k, i in enumerate(frames):
        hands.pose(i)
        pen.pose(i / meta["fps"], hands.world(i), hands.valid, i)
        cam.matrix_world = D @ Matrix(hands.cam_T[i].tolist()) @ D
        path = Path("/tmp") / f"twincam_{i}.png"
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        if box:
            import av

            for pkt in st.encode(av.VideoFrame.from_image(Image.open(path).convert("RGB"))):
                box.mux(pkt)
        print(f"frame {k + 1}/{len(frames)} (video frame {i})", flush=True)
    if box:
        for pkt in st.encode():
            box.mux(pkt)
        box.close()
        print("wrote", review / a.out)


if __name__ == "__main__":
    main()
