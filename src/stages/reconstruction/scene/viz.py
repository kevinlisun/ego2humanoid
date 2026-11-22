"""Captioned slideshow / side-by-side MP4s used to review scene-reconstruction sub-modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthetic.video import write_mp4


def _caption(image: np.ndarray, text: str, height: int = 36) -> np.ndarray:
    import cv2

    bar = np.zeros((height, image.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def _fit(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Letterbox ``image`` (uint8 RGB or gray) into ``size_hw`` keeping the aspect ratio."""
    import cv2

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    h, w = size_hw
    scale = min(h / image.shape[0], w / image.shape[1])
    nh, nw = max(int(image.shape[0] * scale), 1), max(int(image.shape[1] * scale), 1)
    resized = cv2.resize(image[..., :3], (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def slideshow_frames(
    slides: list[list[tuple[str, np.ndarray]]],
    panel_hw: tuple[int, int] = (480, 640),
    fps: int = 10,
    hold_s: float = 1.5,
) -> np.ndarray:
    """Each slide is a row of (caption, uint8 image) panels shown side by side for ``hold_s`` seconds."""
    rows = []
    for slide in slides:
        panels = [_caption(_fit(img, panel_hw), text) for text, img in slide]
        rows.append(np.hstack(panels))
    width = max(r.shape[1] for r in rows)
    height = max(r.shape[0] for r in rows)
    width, height = width + width % 2, height + height % 2
    out = []
    for row in rows:
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[: row.shape[0], : row.shape[1]] = row
        out.extend([canvas] * max(int(round(hold_s * fps)), 1))
    return np.stack(out)


def write_slideshow(
    path: str | Path,
    slides: list[list[tuple[str, np.ndarray]]],
    panel_hw: tuple[int, int] = (480, 640),
    fps: int = 10,
    hold_s: float = 1.5,
) -> Path:
    return write_mp4(path, slideshow_frames(slides, panel_hw, fps, hold_s), fps=fps)
