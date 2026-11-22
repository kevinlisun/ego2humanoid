"""Rigid (SE3) and similarity (Sim3) transforms and wxyz quaternion utilities.

Convention: ``T_A_B`` maps coordinates expressed in frame B into frame A.
Quaternions are scalar-first (w, x, y, z).
"""

import numpy as np


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def compose(*Ts: np.ndarray) -> np.ndarray:
    """compose(T_A_B, T_B_C) -> T_A_C (matmul chain, broadcasting over leading axes)."""
    out = np.asarray(Ts[0], dtype=np.float64)
    for T in Ts[1:]:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def apply_se3(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a (4,4) transform to points of shape (..., 3)."""
    return points @ T[:3, :3].T + T[:3, 3]


def make_sim3(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 similarity transform: x' = s R x + t."""
    T = np.eye(4)
    T[:3, :3] = scale * np.asarray(R)
    T[:3, 3] = t
    return T


def sim3_scale(T: np.ndarray) -> float:
    return float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion(s) (..., 4) in wxyz order to rotation matrices (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) to unit wxyz quaternions with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, m in enumerate(flat):
        tr = np.trace(m)
        if tr > 0:
            s = 2 * np.sqrt(tr + 1.0)
            q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = 2 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] >= m[2, 2]:
            s = 2 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
        q = np.array(q)
        out[i] = q if q[0] >= 0 else -q
    return out.reshape(R.shape[:-2] + (4,))


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Rotation angle in radians of matrices (..., 3, 3)."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues formula."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Least-squares Sim3 (or SE3) mapping ``src`` points onto ``dst`` points, both (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / xs.var(0).sum() if with_scale else 1.0
    return make_sim3(s, R, mu_d - s * R @ mu_s)
