"""2x2 review grid: one video showing perception, simulation, scene rendering and task-space paths.

Panel order is fixed so successive runs stay comparable:

===============  =========================================================================
top-left         source ego RGB with the estimated hand skeletons and tactile forces
top-right        the retargeted robot executing the clip in MuJoCo (free 3/4 camera)
bottom-left      the robot composited into the reconstructed scene, ego viewpoint
bottom-right     task-space trajectories: end-effector, fingertips, object pose, camera pose
===============  =========================================================================

Every panel is on the simulation clock (50 Hz) and the lists must therefore have equal length;
a missing panel is drawn as a labelled placeholder so the layout never shifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import av
import cv2
import numpy as np

Panel = tuple[str, Sequence[np.ndarray] | None]
PLACEHOLDER_NOTE = "not available in this run"


def fit_cell(frame: np.ndarray, cell_hw: tuple[int, int]) -> np.ndarray:
    """Letterbox an RGB frame into ``cell_hw`` without distorting its aspect ratio."""
    ch, cw = cell_hw
    fh, fw = frame.shape[:2]
    s = min(cw / fw, ch / fh)
    small = cv2.resize(frame, (max(int(fw * s), 1), max(int(fh * s), 1)))
    out = np.zeros((ch, cw, 3), np.uint8)
    y, x = (ch - small.shape[0]) // 2, (cw - small.shape[1]) // 2
    out[y : y + small.shape[0], x : x + small.shape[1]] = small
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    scale = max(img.shape[0] / 480, 1.0) * 0.6
    cv2.rectangle(img, (0, 0), (img.shape[1], int(34 * scale)), (0, 0, 0), -1)
    cv2.putText(
        img, text, (12, int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def _placeholder(text: str, cell_hw: tuple[int, int], note: str) -> np.ndarray:
    img = np.full((*cell_hw, 3), 32, np.uint8)
    scale = max(cell_hw[0] / 480, 1.0) * 0.6
    cv2.putText(
        img,
        note,
        (12, cell_hw[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    return _label(img, text)


def compose_grid(
    panels: Sequence[Panel],
    out_path: Path,
    cell_hw: tuple[int, int] = (480, 640),
    fps: int = 50,
    notes: dict[str, str] | None = None,
) -> Path:
    """Write the 2x2 grid of ``panels`` (top-left, top-right, bottom-left, bottom-right)."""
    if len(panels) != 4:
        raise ValueError(f"expected 4 panels, got {len(panels)}")
    lengths = [len(f) for _, f in panels if f is not None]
    if not lengths:
        raise ValueError("every panel is empty")
    n = min(lengths)
    notes = notes or {}
    ch, cw = cell_hw
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 2 * cw, 2 * ch, "yuv420p"
        blanks = {
            name: _placeholder(name, cell_hw, notes.get(name, PLACEHOLDER_NOTE))
            for name, frames in panels
            if frames is None
        }
        for i in range(n):
            cells = [
                blanks[name].copy() if frames is None else _label(fit_cell(frames[i], cell_hw), name)
                for name, frames in panels
            ]
            grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
            for pkt in stream.encode(av.VideoFrame.from_ndarray(grid, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return out_path


def write_video(out_path: Path, frames: Sequence[np.ndarray], fps: int = 50) -> Path:
    """Write ``frames`` (RGB, uniform size) as a single-panel H.264 video."""
    if not len(frames):
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for frame in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return out_path


def read_video_rgb(path: Path) -> list[np.ndarray]:
    """Read every frame of ``path`` as RGB uint8."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames
