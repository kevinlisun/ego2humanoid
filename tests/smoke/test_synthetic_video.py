import numpy as np
import pytest

from synthetic.video import moving_box_frames, read_frames, write_mp4


@pytest.mark.smoke
def test_mp4_roundtrip_preserves_frame_count_and_pts(tmp_path):
    frames = moving_box_frames(30)
    path = write_mp4(tmp_path / "v.mp4", frames, fps=30)
    out, pts = read_frames(path)
    assert out.shape == frames.shape
    assert np.allclose(np.diff(pts), 1 / 30, atol=1e-6)
    # codec is lossy; the red box must still be where it started
    assert out[0, 60, 5, 0] > 150 and out[-1, 60, 150, 0] > 150
