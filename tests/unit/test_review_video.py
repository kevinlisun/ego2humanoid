"""Review-video building blocks: the 2x2 grid composer and the task-space trajectory panel."""

import numpy as np
import pytest

from stages.qa.grid_video import compose_grid, fit_cell
from stages.qa.traj_panel import IsoProjection, draw_trajectory_panel


def _traj(n=8):
    t = np.linspace(0, 1, n)
    wrist = np.stack([np.stack([0.3 + t, 0.1 * np.ones(n), 0.9 + 0.1 * t], 1)] * 2, 1)
    return {
        "wrist_m": wrist,
        "fingertip_m": np.repeat(wrist[:, :, None], 5, axis=2),
        "object_pos_m": np.stack([0.4 + t, np.zeros(n), 0.8 + 0.2 * t], 1),
        "object_rot": np.tile(np.eye(3), (n, 1, 1)),
        "camera_T_sim": np.eye(4),
    }


def test_fit_cell_letterboxes_without_distortion():
    out = fit_cell(np.full((100, 400, 3), 255, np.uint8), (480, 640))
    assert out.shape == (480, 640, 3)
    assert out[0, 0].sum() == 0 and out[240, 320].sum() == 765  # padded above, image in the middle


def test_iso_projection_keeps_every_point_inside_the_panel():
    traj = _traj()
    pts = np.concatenate([traj["wrist_m"].reshape(-1, 3), traj["object_pos_m"]])
    proj = IsoProjection(pts, (480, 640))
    px = proj(pts)
    assert (px[:, 0] >= 0).all() and (px[:, 0] < 640).all()
    assert (px[:, 1] >= 0).all() and (px[:, 1] < 480).all()
    # The isometric map is injective in z: lifting the object moves it up in the panel.
    assert proj(np.array([0.4, 0.0, 1.0]))[1] < proj(np.array([0.4, 0.0, 0.8]))[1]


def test_draw_trajectory_panel_returns_a_drawn_rgb_frame():
    frame = draw_trajectory_panel(_traj(), 4, (240, 320))
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    assert frame.std() > 1.0  # something was actually drawn on the background


def test_compose_grid_writes_a_2x2_video_with_a_placeholder(tmp_path):
    import av

    frames = [np.full((120, 160, 3), 200, np.uint8)] * 5
    out = compose_grid(
        [("a", frames), ("b", frames), ("c", None), ("d", frames)],
        tmp_path / "grid.mp4",
        cell_hw=(120, 160),
    )
    stream = av.open(str(out)).streams.video[0]
    assert (stream.width, stream.height) == (320, 240)


def test_compose_grid_rejects_a_wrong_panel_count(tmp_path):
    with pytest.raises(ValueError):
        compose_grid([("a", [np.zeros((4, 4, 3), np.uint8)])], tmp_path / "x.mp4")


def test_scene_ego_camera_matches_the_estimated_pose():
    """The compiled "ego" camera reproduces T_sim_camera including roll (a free camera cannot)."""
    import mujoco

    from stages.manipulation.mjcf_scene import SceneConfig, build_scene

    rot = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # rolled ego view
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, [0.1, 0.2, 1.4]
    model = build_scene(SceneConfig(table_top_z_m=0.8, ego_camera_T_sim=T, ego_camera_fovy_deg=58.0))
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    assert cam >= 0 and abs(model.cam_fovy[cam] - 58.0) < 1e-6
    np.testing.assert_allclose(model.cam_pos[cam], T[:3, 3], atol=1e-6)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, model.cam_quat[cam])
    # MuJoCo cameras look along -z with +y up; OpenCV looks along +z with +y down.
    np.testing.assert_allclose(mat.reshape(3, 3) @ np.array([0, 0, -1.0]), rot[:, 2], atol=1e-6)
    np.testing.assert_allclose(mat.reshape(3, 3) @ np.array([0, -1.0, 0]), rot[:, 1], atol=1e-6)


def test_scene_d435_camera_matches_the_unitree_description():
    """The "d435" camera reproduces the head camera of the official G1 URDF and tracks the torso."""
    import mujoco

    from stages.manipulation.mjcf_scene import (
        D435_PARENT_BODY,
        D435_PITCH_RAD,
        D435_POS_IN_TORSO_M,
        SceneConfig,
        build_scene,
    )

    base = np.eye(4)
    base[2, 3] = 0.793
    model = build_scene(SceneConfig(T_sim_robot_base=base, table_top_z_m=0.8))
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435")
    assert cam >= 0
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, D435_PARENT_BODY)
    assert model.cam_bodyid[cam] == torso  # rigidly mounted, so it moves with the robot
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rot_torso = data.xmat[torso].reshape(3, 3)
    pos_in_torso = rot_torso.T @ (data.cam_xpos[cam] - data.xpos[torso])
    np.testing.assert_allclose(pos_in_torso, D435_POS_IN_TORSO_M, atol=1e-6)
    forward = -data.cam_xmat[cam].reshape(3, 3)[:, 2]  # MuJoCo cameras look along -z
    assert np.arcsin(-forward[2]) == pytest.approx(D435_PITCH_RAD, abs=1e-6)
    assert forward[0] > 0 and abs(forward[1]) < 1e-9
