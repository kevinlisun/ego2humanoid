import numpy as np

from stages.reconstruction.objects.grasp_proxy import detect_grasp_proxy


def _lm(ap, z):
    lm = np.zeros((len(ap), 21, 3))
    lm[:, 4, 0] = ap / 2
    lm[:, 8, 0] = -ap / 2
    lm[:, [4, 8], 2] = z[:, None]
    return lm


def test_detects_closed_pinch_near_table():
    t = np.arange(60) / 30.0
    ap = np.full(60, 0.10)
    ap[20:45] = 0.03
    z = np.full(60, 0.85)
    p = detect_grasp_proxy(_lm(ap, z), t)
    assert p is not None and 20 <= p.grasp_idx <= 21 and p.release_idx == 44
    assert abs(p.center_m[2] - (0.80 + p.half_size_m[2])) < 1e-9


def test_no_grasp_when_always_open():
    t = np.arange(60) / 30.0
    assert detect_grasp_proxy(_lm(np.full(60, 0.1), np.full(60, 0.85)), t) is None
