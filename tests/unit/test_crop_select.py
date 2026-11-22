import numpy as np

from stages.reconstruction.objects.crop_select import is_fragmented, select_crop_mask


def _blob(*rects, shape=(100, 100)):
    m = np.zeros(shape, bool)
    for y0, y1, x0, x1 in rects:
        m[y0:y1, x0:x1] = True
    return m


def test_fragmented_mask_rejected():
    whole = _blob((10, 30, 10, 30))
    split = _blob((10, 30, 10, 20), (10, 30, 40, 50))
    assert not is_fragmented(whole)
    assert is_fragmented(split)
    # the split mask is larger but must lose to the whole one
    assert select_crop_mask([split, whole]) == 1


def test_all_rejected_returns_none():
    assert select_crop_mask([np.zeros((100, 100), bool), _blob((0, 100, 0, 100))]) is None


def test_inventory_area_gate():
    small, big = _blob((10, 15, 10, 15)), _blob((10, 30, 10, 30))
    assert select_crop_mask([small], inventory_area_px=big.sum()) is None
    assert select_crop_mask([small, big], inventory_area_px=big.sum()) == 1
