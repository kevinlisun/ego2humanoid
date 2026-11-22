"""Pose/camera conventions used by scripts/twin_render.py, checked without bpy."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def _T(p, q_wxyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T[:3, 3] = p
    return T


def test_world_matrix_is_translation_times_rotation():
    p, q = np.array([0.3, -0.2, 0.9]), np.array([0.5, 0.5, -0.5, 0.5])
    T_geom = _T([0.01, 0.02, 0.03], [1, 0, 0, 0])
    W = _T(p, q) @ T_geom
    T = np.eye(4)
    T[:3, 3] = p
    Rm = np.eye(4)
    Rm[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]  # hand-derived from wxyz (0.5, 0.5, -0.5, 0.5)
    assert np.allclose(_T(p, q), T @ Rm)
    local = np.array([0.1, 0.0, 0.0, 1.0])
    assert np.allclose(W @ local, T @ Rm @ T_geom @ local)
    assert np.allclose(W[:3, 3], p + Rm[:3, :3] @ [0.01, 0.02, 0.03])


def _project(T_sim_camera, f, w, h, X_sim):
    Xc = np.linalg.inv(T_sim_camera) @ np.append(X_sim, 1.0)
    return np.array([f * Xc[0] / Xc[2] + w / 2, f * Xc[1] / Xc[2] + h / 2])


def test_pinhole_projection_known_pixel():
    f, w, h = 600.0, 960, 540
    T = _T([0.1, 0.2, 1.0], [np.cos(0.3), 0, 0, np.sin(0.3)])  # T_sim_camera, OpenCV axes
    Xc = np.array([0.05, -0.04, 2.0])
    X_sim = (T @ np.append(Xc, 1.0))[:3]
    uv = _project(T, f, w, h, X_sim)
    assert np.allclose(uv, [f * 0.05 / 2 + w / 2, f * -0.04 / 2 + h / 2], atol=1e-6)


def test_mujoco_camera_to_opencv_flip():
    # MuJoCo camera looks along -z with +y up; OpenCV looks along +z with +y down.
    cam_mat = R.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()  # columns: camera x,y,z in world
    cam_pos = np.array([0.4, 0.1, 1.2])
    R_cv = cam_mat @ np.diag([1, -1, -1])
    fwd_mj = -cam_mat[:, 2]
    assert np.allclose(R_cv[:, 2], fwd_mj)
    assert np.allclose(np.linalg.det(R_cv), 1.0)
    X = cam_pos + 2.0 * fwd_mj + 0.1 * cam_mat[:, 0] + 0.2 * cam_mat[:, 1]  # right 0.1, up 0.2
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_cv, cam_pos
    f, w, h = 500.0, 640, 480
    u, v = _project(T, f, w, h, X)
    assert np.isclose(u, f * 0.1 / 2 + w / 2, atol=1e-6)
    assert np.isclose(v, -f * 0.2 / 2 + h / 2, atol=1e-6)  # up in MuJoCo = smaller row
