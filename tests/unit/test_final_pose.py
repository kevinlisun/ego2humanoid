import numpy as np

from skills.final_pose import final_pose


def _track(n=100, dt=0.05):
    t = np.arange(n) * dt
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 2, 3] = 0.8
    return t, T


def _tilt(T, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T[:, :3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_upright_resting_object_has_no_defects():
    t, T = _track()
    fp = final_pose(t, T, bare_table=True)
    assert fp.tilt_deg < 1e-6 and fp.drift_m == 0.0 and fp.height_m == 0.0
    assert fp.defects() == []


def test_tilted_drifting_and_sunken_object():
    t, T = _track()
    _tilt(T[60:], 40.0)
    T[-10:, 0, 3] = np.linspace(0.0, 0.02, 10)  # still sliding in the rest window
    T[60:, 2, 3] = 0.77  # rests 3 cm below its start height
    fp = final_pose(t, T, bare_table=True)
    assert abs(fp.tilt_deg - 40.0) < 1e-6
    assert abs(fp.drift_m - 0.02) < 1e-9 and abs(fp.height_m - 0.03) < 1e-9
    assert fp.defects() == ["FINAL_TILTED", "FINAL_NOT_AT_REST", "FINAL_HEIGHT"]


def test_height_is_not_judged_with_a_receptacle():
    t, T = _track()
    T[60:, 2, 3] = 0.9
    fp = final_pose(t, T, bare_table=False)
    assert fp.height_m is None and fp.defects() == []


def test_non_finite_track_is_skipped():
    t, T = _track(n=3)
    T[1:] = np.nan
    assert final_pose(t, T, bare_table=True) is None
