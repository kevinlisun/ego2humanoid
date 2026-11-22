"""Third-person H.264 recording of a locomotion + manipulation episode (PyAV, EGL offscreen)."""

from __future__ import annotations

from pathlib import Path

import av
import mujoco

FPS = 25
SIZE = (960, 540)  # width, height


class VideoRecorder:
    """Call ``attach`` once the model exists, then ``maybe_capture`` after every physics step."""

    def __init__(self, path: str | Path, fps: int = FPS):
        self.path, self.fps = Path(path), fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.path), "w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = SIZE[0], SIZE[1], "yuv420p"
        self.model = self.data = self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.every = 1
        self.n = 0

    def attach(self, model, data) -> None:
        self.model, self.data = model, data
        model.vis.global_.offwidth, model.vis.global_.offheight = (
            max(SIZE[0], model.vis.global_.offwidth),
            max(SIZE[1], model.vis.global_.offheight),
        )
        self.renderer = mujoco.Renderer(model, height=SIZE[1], width=SIZE[0])
        self.every = max(1, round(1.0 / (self.fps * model.opt.timestep)))
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.4, -18.0, 150.0

    def maybe_capture(self, robot=None) -> None:
        self.n += 1
        if self.n % self.every:
            return
        # Follow the pelvis, looking slightly ahead so the table and object stay in view.
        pelvis = self.data.qpos[:3]
        self.cam.lookat[:] = [pelvis[0] + 0.3, pelvis[1], 0.85]
        self.renderer.update_scene(self.data, self.cam)
        frame = av.VideoFrame.from_ndarray(self.renderer.render(), format="rgb24")
        for pkt in self.stream.encode(frame):
            self.container.mux(pkt)

    def close(self) -> None:
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()
        if self.renderer is not None:
            self.renderer.close()
