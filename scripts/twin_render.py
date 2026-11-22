# ruff: noqa: E501
"""All Blender views of a clip in one bpy session (Track BT, BT.29); runs in the bpy venv.

    /tmp/bpy_venv/bin/python scripts/twin_render.py <clip_dir> --tag _bt --views ego third --fps 10 \
        --samples 16 --width 960 --device gpu

Views: ``ego`` (D435 head camera from ``sim_state``), ``third`` (fixed camera equal to the old MuJoCo panel:
lookat (0.25, -0.05, 0.9), 1.5 m, azimuth 150, elevation -25, 16:9, same as the ego view) and ``human`` (real camera of
``twin_scene/scene.json``, robot hidden, record 0 or ``--frame``). Outputs ``review/<view>_blender<tag>.mp4``
(``human``: ``.png``). Every k-th 50 Hz record is rendered, k = round(50 / fps). Frames are cached as PNGs under
``review/.twin_frames/<tag>/<view>/`` and existing PNGs are skipped (resumable).

The scene is built once by ``render_twin_sim.build``; the table slab (``table_top`` geom of ``sim_model``, else
``scene.json``) is drawn in every view. If ``sim_state`` has ``pad_force_N`` (T, P) and ``pad_geom_names`` (P,),
each pad is coloured grey -> red by its force (0..PAD_FORCE_MAX_N) per frame.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from render_twin_cam import lin, mat  # noqa: E402
from render_twin_sim import (  # noqa: E402
    build,
    camera_matrix,
    enable_gpu,
    human_camera,
    pose_frame,
    setup_world,
)
from twin_appearance_blender import add_box  # noqa: E402

from contracts.twin_appearance import TwinAppearance

PAD_FORCE_MAX_N = 8.2
PAD_IDLE_RGB = (0.62, 0.62, 0.64)
PAD_HOT_RGB = (1.0, 0.1, 0.05)
SIM_HZ = 50
THIRD_LOOKAT, THIRD_DIST_M, THIRD_AZ_DEG, THIRD_EL_DEG, THIRD_FOVY_DEG = (
    (0.25, -0.05, 0.9),
    1.5,
    150,
    -25,
    45.0,
)  # noqa: E501
EGO_FOVY_DEG = 42.5
TABLE_THICKNESS_M = 0.04


def heat_rgb(force_n: float) -> tuple[float, float, float]:
    """Robot grey (no force) to red (PAD_FORCE_MAX_N); untouched pads keep the robot's own look."""
    x = min(max(force_n / PAD_FORCE_MAX_N, 0.0), 1.0)
    return tuple(g + x * (r - g) for g, r in zip(PAD_IDLE_RGB, PAD_HOT_RGB, strict=True))


def add_table(model, twin, app, loaded):
    """Table slab from the ``table_top`` box geom (world body); scene.json is the fallback. Returns the object."""
    g = next((g for g in model["geoms"] if g["name"] == "table_top" and g["type"] == "box"), None)
    if g is None:
        t = twin["table"]
        top, c, h = t["table_top_z_m"], t["table_center_xy_m"], t["table_half_size_xy_m"]
        return add_box(
            "table", (*c, top - TABLE_THICKNESS_M / 2), (2 * h[0], 2 * h[1], TABLE_THICKNESS_M),
            app.table.color_rgb, app.table.roughness,
        )  # fmt: skip
    if g["body"] != "world":
        return None  # drawn with its body by build()
    rgb, rough = (app.table.color_rgb, app.table.roughness) if loaded else (g["rgba"][:3], 0.7)
    ob = add_box("table", (0, 0, 0), [2 * s for s in g["size"]], rgb, rough)
    ob.matrix_world = Matrix.Translation(g["pos"]) @ Quaternion(g["quat"]).to_matrix().to_4x4()
    return ob


def pad_materials(state, items, model):
    """{body name: (Principled BSDF, pad column indices)}; one private material per pad body.

    Pads are collision proxies (``sim_model['pad_geoms']``, same order as ``pad_geom_names``), so the heat colours
    the visual meshes of the pad's body. Empty for v1 states or models without ``pad_geoms``.
    """
    pad_geoms = model.get("pad_geoms")
    if not pad_geoms or "pad_force_N" not in state.files or len(pad_geoms) != state["pad_force_N"].shape[1]:
        return {}
    names = [str(x) for x in state["body_names"]]
    cols: dict[str, list[int]] = {}
    for j, g in enumerate(pad_geoms):
        cols.setdefault(g["body"], []).append(j)
    out = {}
    for body, js in cols.items():
        obs = [ob for ob, b, _ in items if names[b] == body]
        if not obs:
            continue
        m = mat(f"pad_{body}", (*PAD_IDLE_RGB, 1), 0.5)
        for ob in obs:
            ob.data.materials.clear()
            ob.data.materials.append(m)
        out[body] = (m.node_tree.nodes["Principled BSDF"], js)
    return out


def color_pads(state, pads, i):
    for node, js in pads.values():
        f = float(np.max(state["pad_force_N"][i, js]))
        node.inputs["Base Color"].default_value = lin((*heat_rgb(f), 1))


def third_matrix() -> Matrix:
    az, el = math.radians(THIRD_AZ_DEG), math.radians(THIRD_EL_DEG)
    fwd = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    eye = Vector(THIRD_LOOKAT) - THIRD_DIST_M * fwd
    rot = fwd.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(eye) @ rot


def write_video(pngs, path, fps, size):
    import av
    from PIL import Image

    box = av.open(str(path), "w")
    st = box.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
    for p in pngs:
        for pkt in st.encode(av.VideoFrame.from_image(Image.open(p).convert("RGB"))):
            box.mux(pkt)
    for pkt in st.encode():
        box.mux(pkt)
    box.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--tag", default="_bt")
    ap.add_argument("--state-tag", help="tag of sim_state_sharpa<tag>.npz (default: --tag)")
    ap.add_argument("--views", nargs="+", choices=("ego", "third", "human"), default=["ego", "third"])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--frame", type=int, default=0, help="record index of the human view")
    ap.add_argument(
        "--appearance", type=Path, help="TwinAppearance JSON (default: review/twin_appearance.json)"
    )  # noqa: E501
    ap.add_argument("--frames-dir", type=Path, help="PNG cache root (default: review/.twin_frames)")
    a = ap.parse_args()

    review = a.clip / "review"
    app_path = a.appearance or review / "twin_appearance.json"
    loaded = app_path.exists()
    app = TwinAppearance.model_validate_json(app_path.read_text()) if loaded else TwinAppearance()
    model = json.loads((review / "sim_model" / "sim_model.json").read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # The robot-less human view needs its own build (scene objects only); other views share one build.
    robot_views = [v for v in a.views if v != "human"]
    sc = bpy.context.scene
    t0 = time.time()
    state, items, twin = build(a.clip, a.state_tag or a.tag, app, human=not robot_views)
    scene_bodies = {o["body"] for o in twin["objects"]}
    names = [str(x) for x in state["body_names"]]
    cam = setup_world(sc, a.samples, a.width, app, twin, show_table=False)
    add_table(model, twin, app, loaded)
    print("device:", enable_gpu(sc) if a.device == "gpu" else "CPU", flush=True)
    pads = pad_materials(state, items, model) if robot_views else {}
    print(f"scene built in {time.time() - t0:.1f}s; pads coloured: {len(pads)}", flush=True)

    k = max(1, round(SIM_HZ / a.fps))
    n = len(state["t"])
    picks = list(range(0, n, k))
    cache = (a.frames_dir or review / ".twin_frames") / a.tag.lstrip("_")
    for view in a.views:
        t1 = time.time()
        vdir = cache / view
        vdir.mkdir(parents=True, exist_ok=True)
        if view == "human":
            sc.render.resolution_x, sc.render.resolution_y = human_camera(cam, twin, a.width)
            for ob, b, _ in items:
                ob.hide_render = names[b] not in scene_bodies
            pose_frame(state, items, a.frame)
            out = review / f"human_blender{a.tag}.png"
            sc.render.filepath = str(out)
            bpy.ops.render.render(write_still=True)
            print("wrote", out, flush=True)
            continue
        for ob, _, _ in items:
            ob.hide_render = False
        if view == "ego":
            cam.data.angle = math.radians(EGO_FOVY_DEG)
            sc.render.resolution_x, sc.render.resolution_y = a.width, a.width * 9 // 16
        else:
            cam.data.angle = math.radians(THIRD_FOVY_DEG)
            cam.matrix_world = third_matrix()
            sc.render.resolution_x, sc.render.resolution_y = a.width, a.width * 9 // 16
        pngs, done = [], 0
        for j, i in enumerate(picks):
            png = vdir / f"f{i:05d}.png"
            pngs.append(png)
            if png.exists():
                continue
            pose_frame(state, items, i)
            if pads:
                color_pads(state, pads, i)
            if view == "ego":
                cam.matrix_world = camera_matrix(state, i)
            sc.render.filepath = str(png)
            bpy.ops.render.render(write_still=True)
            done += 1
            print(f"{view} {j + 1}/{len(picks)} (record {i})", flush=True)
        out = review / f"{view}_blender{a.tag}.mp4"
        write_video(pngs, out, a.fps, (sc.render.resolution_x, sc.render.resolution_y))
        print(f"wrote {out} ({done} new frames, {time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
