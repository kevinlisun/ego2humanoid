import numpy as np

from stages.reconstruction.scene.removal_quality import passes, removal_metrics


def _imgs():
    before = np.full((100, 100, 3), 120, np.uint8)
    before[40:60, 40:60] = (0, 128, 255)
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    return before, mask


def test_clean_removal_passes():
    before, mask = _imgs()
    after = before.copy()
    after[mask] = 120
    assert passes(removal_metrics(before, after, mask))


def test_residual_object_fails():
    before, mask = _imgs()
    assert not passes(removal_metrics(before, before.copy(), mask))


def test_outside_change_fails():
    before, mask = _imgs()
    after = before.copy()
    after[mask] = 120
    after[:20] = 255
    assert not passes(removal_metrics(before, after, mask))
