"""Unit tests for the background Gaussian splat and the neural ego compositor."""

import numpy as np

from stages.reconstruction.scene.bg_splat import _quats_from_normals, fill_smooth, match_colors
from stages.reconstruction.scene.neural_ego import FAR_DEPTH_M, composite_neural_ego


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_gaussian_discs_are_oriented_along_their_surface_normal():
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]])
    for q, n in zip(_quats_from_normals(normals), normals, strict=True):
        third = _quat_to_matrix(q)[:, 2]
        assert np.allclose(third, n / np.linalg.norm(n), atol=1e-6)


def test_each_surface_is_recoloured_on_its_own():
    # Two surfaces repainted in opposite directions, as FLUX does to the table and the backdrop.
    labels = np.zeros((10, 10), int)
    labels[5:] = 1
    target = np.zeros((10, 10, 3), np.float32)
    target[:5] = 200.0
    target[5:] = 50.0
    source = np.full((10, 10, 3), 100.0, np.float32)
    out = match_colors(source, target, labels, np.ones((10, 10), bool))
    assert np.allclose(out, target, atol=1e-3)


def test_the_human_is_robust_to_covering_most_of_a_surface():
    # 60 % of the surface is hidden by something dark; the median would follow it, a high quantile
    # follows the surface instead.
    labels = np.zeros((10, 10), int)
    target = np.full((10, 10, 3), 200.0, np.float32)
    target[:6] = 20.0
    source = np.full((10, 10, 3), 100.0, np.float32)
    assert match_colors(source, target, labels, np.ones((10, 10), bool), quantile=75.0).max() > 150


def test_depth_is_diffused_into_the_masked_region():
    values = np.tile(np.linspace(1.0, 2.0, 16, dtype=np.float32), (16, 1))
    mask = np.zeros((16, 16), bool)
    mask[6:10, 6:10] = True
    filled = fill_smooth(values, mask, iters=600)
    assert np.allclose(filled[~mask], values[~mask])
    assert np.abs(filled[mask] - values[mask]).max() < 0.05


def test_the_background_is_shown_only_where_it_is_in_front(tmp_path):
    h, w, n = 4, 4, 2
    depth = np.full((n, h, w), FAR_DEPTH_M, np.float32)
    depth[:, :2] = 0.5  # the simulated robot covers the top half
    ego = tmp_path / "ego.npz"
    np.savez(
        ego,
        rgb=np.full((n, h, w, 3), 10, np.uint8),
        depth=depth,
        body_rgb=np.zeros((n, h, w, 3), np.uint8),
        body_mask=np.zeros((n, h, w), np.float32),
        cam_T_world=np.tile(np.eye(4), (n, 1, 1)),
        fovy_deg=np.float32(45.0),
        size=np.asarray([h, w], np.int32),
    )
    bg = tmp_path / "bg.npz"
    np.savez(
        bg,
        rgb=np.full((n, h, w, 3), 200, np.uint8),
        depth=np.full((n, h, w), 1.0, np.float32),  # always behind the robot, in front of nothing
        alpha=np.ones((n, h, w), np.float32),
    )
    out = np.stack(composite_neural_ego(ego, bg, ghost_alpha=0.0))
    assert (out[:, :2] == 10).all()  # robot wins where it is nearer
    assert (out[:, 2:] == 200).all()  # background fills the rest
