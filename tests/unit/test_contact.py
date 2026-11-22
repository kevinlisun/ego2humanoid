import numpy as np

from stages.retargeting.contact import contact_flags, write_alignment_bundle


def test_pinch_flag_only_when_both_tips_touch(tmp_path):
    lm = np.zeros((3, 21, 3))
    lm[:, 4] = [[0.5, 0, 0], [0.01, 0, 0], [0.01, 0, 0]]
    lm[:, 8] = [[0.5, 0, 0], [0.5, 0, 0], [-0.01, 0, 0]]
    obj = np.array([[0.0, 0, 0]])
    f = contact_flags(lm, obj)
    assert f["thumb"].tolist() == [False, True, True]
    assert f["pinch"].tolist() == [False, False, True]
    root = write_alignment_bundle(tmp_path, np.eye(4), 0.8, 1.0, f)
    assert (root / "contact_flags.npz").exists() and (root / "alignment.json").exists()
