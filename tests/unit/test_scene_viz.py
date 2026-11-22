import numpy as np

from stages.reconstruction.scene.viz import slideshow_frames


def test_slideshow_frames_shape_and_even_size():
    a = np.zeros((30, 50, 3), np.uint8)
    b = np.full((20, 20), 200, np.uint8)
    frames = slideshow_frames([[("a", a), ("b", b)], [("a", a)]], panel_hw=(40, 60), fps=4, hold_s=1.0)
    assert frames.shape[0] == 8
    assert frames.shape[1] % 2 == 0 and frames.shape[2] % 2 == 0
    assert frames.shape[2] == 120  # two 60 px panels
