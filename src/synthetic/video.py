"""Synthetic moving-box video (H.264 MP4 with constant 30 fps PTS)."""

from pathlib import Path

import numpy as np


def moving_box_frames(n: int = 30, width: int = 160, height: int = 120) -> np.ndarray:
    """uint8 RGB frames (n, h, w, 3): a red box moving left to right over a grey background."""
    frames = np.full((n, height, width, 3), 90, np.uint8)
    size = 20
    for i in range(n):
        x = int((width - size) * i / max(n - 1, 1))
        frames[i, height // 2 - size // 2 : height // 2 + size // 2, x : x + size] = (220, 30, 30)
    return frames


def write_mp4(path: str | Path, frames: np.ndarray, fps: int = 30) -> Path:
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def read_frames(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode all frames; returns (frames uint8 (n, h, w, 3), pts_seconds (n,))."""
    import av

    frames, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            pts.append(float(frame.pts * stream.time_base))
    return np.stack(frames), np.asarray(pts)
