"""Drop every stage-11 sim-ready object on a table in MuJoCo and render a settle video.

Collision uses the per-object convex pieces, mass/inertia/friction come from the URDF
and scene_objects_info.json, and the visual is the textured mesh.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from synthetic.video import write_mp4

TABLE_Z = 0.0


def _object_xml(
    idx: int, urdf: Path, friction: float, pos: np.ndarray, quat: str = "1 0 0 0"
) -> tuple[str, str]:
    root = ET.parse(urdf).getroot()
    link = root.find("link")
    mass = float(link.find("inertial/mass").get("value"))
    inertia = link.find("inertial/inertia").attrib
    com = [float(v) for v in link.find("inertial/origin").get("xyz").split()]
    assets, geoms = [], []
    base = urdf.parent
    vis = link.find("visual/geometry/mesh").get("filename")
    assets.append(f'<mesh name="o{idx}_vis" file="{(base / vis).resolve()}"/>')
    geoms.append(
        f'<geom type="mesh" mesh="o{idx}_vis" contype="0" conaffinity="0" group="1" '
        f'material="m{idx}" mass="0"/>'
    )
    for j, col in enumerate(link.findall("collision")):
        m = col.find("geometry/mesh")
        sc = m.get("scale").split()
        name = f"o{idx}_c{j}"
        assets.append(
            f'<mesh name="{name}" file="{(base / m.get("filename")).resolve()}" scale="{" ".join(sc)}"/>'
        )
        geoms.append(
            f'<geom type="mesh" mesh="{name}" group="3" rgba="1 0 0 0.2" mass="0" '
            f'friction="{friction} 0.005 0.0001" solref="0.004 1" condim="3"/>'
        )
    tex = sorted((base / "material").glob("*_Kd.png"))
    mat = ""
    if tex:
        assets.append(f'<texture name="t{idx}" type="2d" file="{tex[0].resolve()}"/>')
        mat = f'<material name="m{idx}" texture="t{idx}"/>'
    assets.append(mat or f'<material name="m{idx}" rgba="0.7 0.7 0.7 1"/>')
    inert = (
        f'<inertial pos="{" ".join(map(str, com))}" mass="{mass}" '
        f'diaginertia="{inertia["ixx"]} {inertia["iyy"]} {inertia["izz"]}"/>'
    )
    body = (
        f'<body name="obj{idx}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat}"><freejoint/>{inert}'
        + "".join(geoms)
        + "</body>"
    )
    return "".join(assets), body


def build_xml(scene: Path) -> tuple[str, list[str]]:
    info = json.loads((scene / "s11_sim/scene_objects_info.json").read_text())
    assets, bodies, names = [], [], []
    n = len(info)
    poses_path = scene / "s12_physics/pb_scene_poses.json"
    poses = json.loads(poses_path.read_text()) if poses_path.exists() else {}
    for k, (i, o) in enumerate(sorted(info.items(), key=lambda kv: int(kv[0]))):
        urdf = scene / "s11_sim/objects" / o["category"] / o["model"] / "urdf" / f"{o['model']}.urdf"
        x = (k - (n - 1) / 2) * 0.2
        pose = poses.get(o["name"])
        if pose:
            # Stage-12 poses are PyBullet (xyzw); MuJoCo wants wxyz. Lift slightly to test the drop.
            qx, qy, qz, qw = pose[1]
            p = np.array(pose[0]) + [0, 0, 0.01]
            a, b = _object_xml(int(i), urdf, o["friction"], p, f"{qw} {qx} {qy} {qz}")
            assets.append(a)
            bodies.append(b)
            names.append(o["category"])
            continue
        a, b = _object_xml(int(i), urdf, o["friction"], np.array([x, 0.0, TABLE_Z + 0.12 + 0.06 * k]))
        assets.append(a)
        bodies.append(b)
        names.append(o["category"])
    xml = f"""<mujoco><option timestep="0.002"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><texture name="grid" type="2d" builtin="checker" width="256" height="256"
 rgb1=".75 .75 .75" rgb2=".6 .6 .6"/>
<material name="floor" texture="grid" texrepeat="6 6" reflectance="0.05"/>{"".join(assets)}</asset>
<worldbody><light pos="0 -0.5 1.5" dir="0 0.3 -1" diffuse="0.9 0.9 0.9"/>
<geom type="plane" size="1 1 0.01" material="floor" friction="0.8 0.005 0.0001"/>
{"".join(bodies)}</worldbody></mujoco>"""
    return xml, names


def main(scene_dir: str) -> None:
    scene = Path(scene_dir)
    xml, names = build_xml(scene)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.04]
    cam.distance, cam.elevation = 0.95, -28
    frames, report = [], {}
    fps, dur = 30, 4.0
    for f in range(int(fps * dur)):
        for _ in range(int(1 / fps / model.opt.timestep)):
            mujoco.mj_step(model, data)
        cam.azimuth = 90 + 40 * np.sin(2 * np.pi * f / (fps * dur))
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
    for i, n in enumerate(names):
        b = model.body(f"obj{i}")
        report[n] = {
            "mass_kg": float(model.body_mass[b.id]),
            "final_pos": data.xpos[b.id].round(4).tolist(),
            "final_speed": float(np.linalg.norm(data.cvel[b.id])),
        }
    out = scene / "review/stage12_sim_scene.mp4"
    write_mp4(out, np.stack(frames), fps=fps)
    (scene / "review/stage12_sim_report.json").write_text(json.dumps(report, indent=1))
    print(out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/scenes/egodex_104")
