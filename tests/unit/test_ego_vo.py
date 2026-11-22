import cv2
import numpy as np

from stages.reconstruction.align.ego_vo import estimate_poses

W, H, F = 320, 240, 300.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
PLANE_Z = 3.0
RES = 0.004  # metres per texture pixel


def _render(cam_xyz, tex):
    d = PLANE_Z - cam_xyz[2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - W / 2) * d / F + cam_xyz[0]
    y = (v - H / 2) * d / F + cam_xyz[1]
    mx = (x / RES + tex.shape[1] / 2).astype(np.float32)
    my = (y / RES + tex.shape[0] / 2).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR), np.full((H, W), d, np.float32)


def test_translating_camera_recovers_metric_path():
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((1500, 1500)) * 255).astype(np.uint8), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    path = [np.array([0.03 * i, 0.0, 0.02 * i]) for i in range(12)]
    out = [_render(p, tex) for p in path]
    frames = np.stack([np.repeat(g[..., None], 3, axis=2) for g, _ in out])
    depths = np.stack([d for _, d in out])
    res = estimate_poses(frames, K, depths)
    got = res.poses[:, :3, 3]
    assert res.ok.all()
    assert np.abs(got - np.array(path)).max() < 0.02


def test_points_to_world_and_smoothing():
    from stages.reconstruction.align.ego_vo import points_to_world, smooth_poses

    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = np.arange(5) * 0.1
    pts = np.zeros((5, 2, 3))
    pts[:, :, 2] = 1.0
    w = points_to_world(pts, poses)
    assert np.allclose(w[3, 0], [0.3, 0.0, 1.0])
    assert np.allclose(smooth_poses(poses, 1), poses)


def test_mug_matches_cup_label():
    from stages.reconstruction.objects.place_target import match_object_name

    assert match_object_name(["cup scene_cup", "tray scene_tray"], "white mug") == "cup scene_cup"
