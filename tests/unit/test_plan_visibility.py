import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot.head_camera import HeadCamera
from stages.locomotion.plan.candidates import Stance, sample_candidates
from stages.locomotion.plan.visibility import box_surface_points, visible_fraction
from stages.locomotion.tabletop import Scenario


def _pts(sc):
    return box_surface_points(sc.object_start_xyz(), sc.cfg["object"]["half_size"])


def test_facing_with_waist_pitch_sees_object_and_facing_away_does_not():
    sc = Scenario.load()
    m, d = sc.build()
    cam = HeadCamera.load()
    obj = sc.object_start_xyz()
    best = max(
        visible_fraction(m, d, cam, Stance(obj[0] - 0.6, obj[1], 0.0, 0.74, wp), _pts(sc))
        for wp in (0.0, 0.2, 0.4, 0.5)
    )
    away = visible_fraction(m, d, cam, Stance(obj[0] - 0.6, obj[1], np.pi, 0.74, 0.0), _pts(sc))
    assert best > 0.5 and away == 0.0


def test_candidates_reject_blocked_footprint():
    sc = Scenario.load()
    ext, res = ((-1.0, 5.0), (-2.0, 2.0)), 0.05
    grid = sc.occupancy(ext, res, robot_margin=0.25)
    obj = sc.object_start_xyz()
    cands, rej = sample_candidates(obj[:2], grid, ext, res)
    assert cands and rej["footprint_blocked"] > 0
    assert all(not grid[int((c.y + 2.0) / res), int((c.x + 1.0) / res)] for c in cands)
