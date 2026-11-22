"""Headless MuJoCo rendering to MP4 (EGL by default, OSMesa fallback)."""

import os
from pathlib import Path

import mujoco
import numpy as np


class VideoRecorder:
    """Collects frames via ``on_record`` and encodes them to H.264 MP4 on ``close``."""

    def __init__(self, path: str | Path, fps: int = 50, size: tuple[int, int] = (480, 640)):
        self.path, self.fps, self.size = Path(path), fps, size
        self._renderer: mujoco.Renderer | None = None
        self._frames: list[np.ndarray] = []
        self._cam = mujoco.MjvCamera()
        self._cam.lookat[:] = (0.4, -0.05, 0.85)
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.1, 150.0, -25.0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, *self.size)
        self._renderer.update_scene(data, self._cam)
        self._frames.append(self._renderer.render().copy())

    @property
    def frames(self) -> list[np.ndarray]:
        """Recorded RGB frames, also usable as a panel source after ``close``."""
        return self._frames

    def close(self) -> Path:
        import av

        if self._renderer is not None:
            self._renderer.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(self.path), "w") as out:
            stream = out.add_stream("libx264", rate=self.fps)
            stream.width, stream.height, stream.pix_fmt = self.size[1], self.size[0], "yuv420p"
            for f in self._frames:
                for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    out.mux(pkt)
            for pkt in stream.encode():
                out.mux(pkt)
        return self.path


def select_gl_backend() -> str:
    """Return the configured GL backend (EGL by default; set MUJOCO_GL=osmesa before import to override)."""
    return os.environ.get("MUJOCO_GL", "egl")
