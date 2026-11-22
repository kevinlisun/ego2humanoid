import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot.head_camera import MJ_TO_CV, HeadCamera
from stages.manipulation.mjcf_scene import SceneConfig, build_scene


def test_pose_matches_mujoco_camera_and_fov():
    cam = HeadCamera.load()
    assert abs(cam.fovx_deg - 69.4) < 0.5
    m = build_scene(SceneConfig())
    d = mujoco.MjData(m)
    d.qpos[m.joint("waist_pitch_joint").qposadr[0]] = 0.3
    mujoco.mj_forward(m, d)
    T = cam.T_world_cam(m, d)
    c = m.camera("d435")
    np.testing.assert_allclose(T[:3, 3], d.cam_xpos[c.id], atol=1e-9)
    np.testing.assert_allclose(T[:3, :3], d.cam_xmat[c.id].reshape(3, 3) @ MJ_TO_CV, atol=1e-9)


def test_project_point_on_optical_axis_hits_principal_point():
    cam = HeadCamera.load()
    T = np.eye(4)
    uv, z = cam.project(T, np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_allclose(uv[0], [cam.width / 2, cam.height / 2])
    assert z[0] == 2.0
