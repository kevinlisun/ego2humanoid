"""Compare a real RGB frame with its rendered digital-twin frame (Track BT, BT.25).

Score 0 = identical, 1 = maximally different. Hand regions can be excluded with ``hand_mask`` because the
twin does not render the real hand.
"""

from __future__ import annotations

import cv2
import numpy as np

W_L1, W_SSIM, W_HIST = 0.4, 0.3, 0.3
HIST_BINS = 32


def hand_mask(joints2d: np.ndarray, image_wh: tuple[int, int], dilate_px: int = 25) -> np.ndarray:
    """Union of dilated convex hulls of 2D hand joints (full-res pixels); NaN joints are skipped."""
    w, h = image_wh
    pts = np.asarray(joints2d, dtype=np.float64)
    pts = pts.reshape(-1, *pts.shape[-2:]) if pts.ndim >= 2 else pts.reshape(1, -1, 2)
    mask = np.zeros((h, w), np.uint8)
    for hand in pts:
        ok = hand[np.isfinite(hand).all(axis=1)]
        if len(ok) == 0:
            continue
        hull = cv2.convexHull(np.round(ok).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        if len(hull) < 3:
            cv2.polylines(mask, [hull], False, 1, 1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def _common(real: np.ndarray, twin: np.ndarray, ignore: np.ndarray | None):
    """Resize twin (and ignore) to the real frame size."""
    h, w = real.shape[:2]
    if twin.shape[:2] != (h, w):
        twin = cv2.resize(twin, (w, h), interpolation=cv2.INTER_AREA)
    if ignore is None:
        ignore = np.zeros((h, w), bool)
    elif ignore.shape[:2] != (h, w):
        ignore = cv2.resize(ignore.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return real, twin, ignore.astype(bool)


def _ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """SSIM map of two grayscale float images in [0, 1] with a gaussian window."""
    c1, c2 = 0.01**2, 0.03**2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def _score(real: np.ndarray, twin: np.ndarray, ignore: np.ndarray) -> dict:
    keep = ~ignore
    if not keep.any():
        return {"score": 0.0, "l1": 0.0, "ssim_loss": 0.0, "hist": 0.0}
    a = real.astype(np.float32) / 255.0
    b = twin.astype(np.float32) / 255.0
    l1 = float(np.abs(a - b)[keep].mean())
    ga = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gb = cv2.cvtColor(twin, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ssim_loss = float(np.clip(1.0 - _ssim_map(ga, gb)[keep].mean(), 0.0, 1.0))
    hist = 0.0
    for c in range(3):
        ha = np.histogram(real[..., c][keep], bins=HIST_BINS, range=(0, 256))[0].astype(np.float64)
        hb = np.histogram(twin[..., c][keep], bins=HIST_BINS, range=(0, 256))[0].astype(np.float64)
        hist += 0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum() / 3
    score = W_L1 * l1 + W_SSIM * ssim_loss + W_HIST * hist
    return {"score": float(np.clip(score, 0.0, 1.0)), "l1": l1, "ssim_loss": ssim_loss, "hist": float(hist)}


def compare(
    real: np.ndarray,
    twin: np.ndarray,
    ignore: np.ndarray | None = None,
    rois: dict[str, tuple[int, int, int, int]] | None = None,
) -> dict:
    """Score the twin against the real frame; ``rois`` boxes are (x0, y0, x1, y1) in real pixels."""
    real, twin, ignore = _common(real, twin, ignore)
    out = _score(real, twin, ignore)
    out["rois"] = {}
    h, w = real.shape[:2]
    for name, (x0, y0, x1, y1) in (rois or {}).items():
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            out["rois"][name] = 0.0
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        out["rois"][name] = _score(real[sl], twin[sl], ignore[sl])["score"]
    return out


def diff_image(real: np.ndarray, twin: np.ndarray, ignore: np.ndarray | None = None) -> np.ndarray:
    """RGB heat-map of the per-pixel absolute difference; ignored pixels are grey."""
    real, twin, ignore = _common(real, twin, ignore)
    d = np.abs(real.astype(np.int16) - twin.astype(np.int16)).mean(axis=2)
    d8 = np.clip(d * 2.0, 0, 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    heat[ignore] = 128
    return heat


def side_by_side(real: np.ndarray, twin: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Three labelled panels: real | twin | diff, all at the real frame size."""
    h, w = real.shape[:2]
    panels = []
    for img, label in ((real, "real"), (twin, "twin"), (diff, "diff")):
        p = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        p = np.ascontiguousarray(p)
        scale = max(0.5, h / 480)
        cv2.putText(p, label, (8, int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(p, label, (8, int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        panels.append(p)
    return np.concatenate(panels, axis=1)
