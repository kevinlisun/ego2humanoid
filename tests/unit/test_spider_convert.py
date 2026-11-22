import json

import numpy as np

from stages.retargeting.spider_convert import convert_clip, mat_to_pose7


def test_convert_clip_layout(tmp_path):
    n = 4
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.linspace(0, 0.3, n)
    T[:, 2, 3] = 0.85
    lm = np.random.default_rng(0).normal(size=(n, 21, 3))
    clip = convert_clip(
        tmp_path,
        "pick_cup",
        0,
        0.02,
        {"right": T, "left": T},
        {"right": lm, "left": lm},
        T,
        "right",
        object_mesh_dir="processed/ego2humanoid/assets/objects/cup",
        table_top_z_m=0.8,
    )
    z = np.load(clip / "trajectory_keypoints.npz")
    assert z["qpos_wrist_right"].shape == (n, 7) and z["qpos_finger_left"].shape == (n, 5, 7)
    assert np.allclose(z["qpos_obj_right"][:, 2], 0.05) and z["qpos_obj_left"][0, 0] > 1
    assert np.allclose(z["qpos_wrist_right"][:, 2], 0.05)
    assert np.allclose(z["qpos_finger_right"][:, 0, 2], lm[:, 4, 2] - 0.8)
    info = json.loads((clip.parent / "task_info.json").read_text())
    assert info["right_object_mesh_dir"] and info["left_object_mesh_dir"] is None


def test_quaternion_is_wxyz():
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # +90 deg about z
    q = mat_to_pose7(T)[3:]
    assert np.allclose(q, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
