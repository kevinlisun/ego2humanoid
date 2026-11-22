import numpy as np
import pytest

pytest.importorskip("mujoco")

from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.reconstruction.align.camera_fit import HeadCamera, best_waist_pitch, human_uv


def test_human_uv_centre_and_offset():
    uv = human_uv(np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]), 500.0, (1000, 800))
    assert np.allclose(uv[0], [0.5, 0.5])
    assert uv[1, 0] == pytest.approx(0.55)


def test_recovers_known_pitch():
    cam = HeadCamera(build_scene(SceneConfig()))
    target_pitch = 0.2
    pos, rot = cam.pose(target_pitch)
    # points 0.5 m in front of the tilted camera, slightly off axis
    pts = np.array([pos + rot @ np.array([dx, 0.02, -0.5]) for dx in (-0.05, 0.0, 0.05)])
    uv_ref, _ = cam.project(pts, target_pitch)
    got = best_waist_pitch(cam, pts[:, None, :], pts, uv_ref)
    assert got.waist_pitch_rad == pytest.approx(target_pitch, abs=0.03)
    assert got.coverage_fraction == 1.0
    assert got.layout_err < 0.02
