"""Temporal cleanup of per-frame hand landmark estimates: gap fill, outlier rejection, smoothing."""

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def _fill(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    t = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(t, t[ok], flat[ok, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape(x.shape)


def clean_landmarks(
    landmarks: np.ndarray,
    valid: np.ndarray,
    outlier_m: float = 0.05,
    median_window: int = 9,
    sigma_frames: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """``landmarks`` (N, 21, 3), ``valid`` (N,) -> (smoothed landmarks, kept mask).

    Frames whose wrist deviates from the running median of the wrist by more than ``outlier_m`` are dropped
    and re-interpolated before Gaussian smoothing. Raises if fewer than 3 frames survive.
    """
    ok = np.asarray(valid, bool).copy()
    if ok.sum() < 3:
        raise ValueError("fewer than 3 valid frames")
    filled = _fill(landmarks, ok)
    for _ in range(2):
        med = median_filter(filled[:, 0], size=(median_window, 1), mode="nearest")
        bad = np.linalg.norm(filled[:, 0] - med, axis=1) > outlier_m
        ok &= ~bad
        if ok.sum() < 3:
            raise ValueError("fewer than 3 frames after outlier rejection")
        filled = _fill(landmarks, ok)
    return gaussian_filter1d(filled, sigma_frames, axis=0, mode="nearest"), ok
