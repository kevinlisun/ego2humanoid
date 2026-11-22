import numpy as np

from skills.twin_compare import compare, diff_image, hand_mask, side_by_side


def _img(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None] * np.ones((48, 1, 3), np.uint8)
    return np.clip(base.astype(int) + rng.integers(-5, 5, base.shape), 0, 255).astype(np.uint8)


def test_identical_scores_zero():
    img = _img()
    assert compare(img, img.copy())["score"] < 1e-6


def test_difference_inside_mask_is_ignored():
    real = _img()
    twin = real.copy()
    twin[10:20, 10:20] = 255 - twin[10:20, 10:20]
    ignore = np.zeros(real.shape[:2], bool)
    ignore[2:28, 2:28] = True
    assert compare(real, twin)["score"] > 0.01
    assert compare(real, twin, ignore)["score"] < 1e-6


def test_colour_shift_scores_higher_than_slight_change():
    real = _img()
    slight = np.clip(real.astype(int) + 4, 0, 255).astype(np.uint8)
    shifted = real.copy()
    shifted[..., 0] = 255
    assert compare(real, shifted)["score"] > compare(real, slight)["score"]


def test_hand_mask_covers_joints_and_dilates():
    joints = np.array([[[20.0, 20.0], [40.0, 20.0], [30.0, 40.0], [np.nan, np.nan]]])
    tight = hand_mask(joints, (100, 80), dilate_px=0)
    wide = hand_mask(joints, (100, 80), dilate_px=10)
    assert tight.shape == (80, 100) and tight[25, 30] and not tight[70, 90]
    assert wide.sum() > tight.sum() and wide[20, 12]


def test_roi_isolates_changed_region_and_panels():
    real = _img()
    twin = real.copy()
    twin[:, :16] = 0
    res = compare(real, twin, rois={"left": (0, 0, 16, 48), "right": (32, 0, 64, 48)})
    assert res["rois"]["left"] > 0.1 and res["rois"]["right"] < 1e-6
    d = diff_image(real, twin)
    assert side_by_side(real, twin, d).shape == (48, 192, 3)
