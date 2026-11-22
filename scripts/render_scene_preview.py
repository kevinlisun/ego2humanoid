"""Orbit video of a clip's real2sim locomanipulation scene (no planning): ``render_scene_preview.py <clip>``."""

import sys
from pathlib import Path

import av
import mujoco

from stages.locomotion.clip_task import DERIVED, load_clip

W, H, FPS, N = 960, 540, 25, 150

clip = sys.argv[1] if len(sys.argv) > 1 else "loco_manip_allinone"
c = load_clip(clip)
m, d = c.scenario.build("sharpa")
d.qpos[:3] = [2.3, 0.0, 0.79]
mujoco.mj_forward(m, d)
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [2.9, -0.05, 0.8]
cam.distance, cam.elevation = 1.8, -30.0
out = Path(DERIVED) / clip / "review" / "loco_manip_scene.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
box = av.open(str(out), "w")
st = box.add_stream("libx264", rate=FPS)
st.width, st.height, st.pix_fmt = W, H, "yuv420p"
for i in range(N):
    cam.azimuth = 120.0 + 120.0 * i / N
    r.update_scene(d, cam)
    for p in st.encode(av.VideoFrame.from_ndarray(r.render(), format="rgb24")):
        box.mux(p)
for p in st.encode():
    box.mux(p)
box.close()
print(out)
