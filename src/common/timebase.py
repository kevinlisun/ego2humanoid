"""Timebase helpers: source-video PTS seconds and resampling."""

import numpy as np

from common.transforms import matrix_to_quat_wxyz, quat_wxyz_to_matrix


def frame_timestamps(n_frames: int, fps: float, t0_s: float = 0.0) -> np.ndarray:
    return t0_s + np.arange(n_frames, dtype=np.float64) / fps


def uniform_grid(t_start_s: float, t_end_s: float, rate_hz: float) -> np.ndarray:
    n = int(np.floor((t_end_s - t_start_s) * rate_hz + 1e-9)) + 1
    return t_start_s + np.arange(n, dtype=np.float64) / rate_hz


def resample_linear(t_src_s: np.ndarray, values: np.ndarray, t_dst_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``values`` (N, ...) sampled at ``t_src_s`` onto ``t_dst_s``.

    Not for rotations or boolean data; use ``resample_se3`` / ``resample_nearest``.
    """
    flat = values.reshape(len(t_src_s), -1)
    out = np.stack([np.interp(t_dst_s, t_src_s, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_dst_s),) + values.shape[1:])


def resample_nearest(t_src_s: np.ndarray, values: np.ndarray, t_dst_s: np.ndarray) -> np.ndarray:
    idx = np.abs(t_dst_s[:, None] - t_src_s[None, :]).argmin(axis=1)
    return values[idx]


def resample_se3(t_src_s: np.ndarray, T: np.ndarray, t_dst_s: np.ndarray) -> np.ndarray:
    """Interpolate (N, 4, 4) rigid transforms: lerp translation, slerp rotation."""
    q = matrix_to_quat_wxyz(T[:, :3, :3])
    for i in range(1, len(q)):  # keep a consistent hemisphere
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    t_dst = np.clip(t_dst_s, t_src_s[0], t_src_s[-1])
    i1 = np.clip(np.searchsorted(t_src_s, t_dst, side="right"), 1, len(t_src_s) - 1)
    i0 = i1 - 1
    a = ((t_dst - t_src_s[i0]) / (t_src_s[i1] - t_src_s[i0]))[:, None]
    q0, q1 = q[i0], q[i1]
    dot = np.clip((q0 * q1).sum(1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-8
    safe = np.where(small, 1.0, sin)
    w0 = np.where(small, 1 - a, np.sin((1 - a) * theta) / safe)
    w1 = np.where(small, a, np.sin(a * theta) / safe)
    out = np.tile(np.eye(4), (len(t_dst), 1, 1))
    out[:, :3, :3] = quat_wxyz_to_matrix(w0 * q0 + w1 * q1)
    out[:, :3, 3] = (1 - a) * T[i0, :3, 3] + a * T[i1, :3, 3]
    return out
