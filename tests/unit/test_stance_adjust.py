import numpy as np

from stages.locomotion.stance_adjust import _start_pose, critical_window, reanchor


def test_reanchor_moves_the_robot_not_the_world():
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.2, 0.8]
    moved = reanchor(T, 0.1, -0.2, 0.0)
    assert np.allclose(moved[:3, 3], [0.4, 0.4, 0.8])
    turned = reanchor(T, 0.0, 0.0, np.pi / 2)  # a robot turned left sees the world turned right
    assert np.allclose(turned[:3, 3], [0.2, -0.5, 0.8], atol=1e-9)


def test_start_pose_is_the_original_robot_in_the_new_frame():
    dx, dy, yaw = 0.3, -0.1, 0.4
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    x, y, psi = _start_pose(dx, dy, yaw)
    # the new robot, seen from the original frame, is at (dx, dy); the original one is its inverse there
    assert np.allclose(R @ np.array([x, y]) + np.array([dx, dy]), 0.0, atol=1e-9)
    assert np.isclose(psi, -yaw)


def test_critical_window_covers_grasp_to_release():
    t = np.arange(0, 6, 0.5)
    w = critical_window(t, 2.0, 3.0)
    assert t[w].min() == 1.0 and t[w].max() == 3.5
