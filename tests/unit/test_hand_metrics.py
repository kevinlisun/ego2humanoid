import numpy as np
import pytest

from common.transforms import make_se3, rot_axis_angle
from evaluation.hand_metrics import evaluate_hands, mpjpe_m, wrist_pose_errors


def _gt(n=20, j=21, seed=0):
    return np.random.default_rng(seed).normal(0, 0.05, (n, j, 3))


def test_perfect_prediction_is_zero():
    gt = _gt()
    r = evaluate_hands(gt, gt)
    assert r["mpjpe_m"] == 0 and r["pa_mpjpe_m"] < 1e-9 and r["sim3_aligned_mpjpe_m"] < 1e-9


def test_sim3_removes_global_scale_and_pose_but_raw_mpjpe_does_not():
    gt = _gt()
    R = rot_axis_angle([0, 0, 1], 0.7)
    pred = 2.5 * gt @ R.T + np.array([0.3, -0.1, 1.0])
    r = evaluate_hands(pred, gt)
    assert r["mpjpe_m"] > 0.1
    assert r["sim3_aligned_mpjpe_m"] < 1e-9
    assert r["sim3_scale"] == pytest.approx(1 / 2.5, rel=1e-6)


def test_noise_is_reported_and_invalid_frames_are_ignored():
    gt = _gt()
    pred = gt.copy()
    pred[:10] += 0.01
    pred[10:] = np.nan
    valid = np.arange(20) < 10
    assert mpjpe_m(pred, gt, valid) == pytest.approx(0.01 * np.sqrt(3), rel=1e-6)


def test_empty_overlap_raises():
    gt = _gt()
    with pytest.raises(ValueError):
        evaluate_hands(gt, gt, valid=np.zeros(20, bool))


def test_wrist_pose_errors():
    T_gt = np.tile(np.eye(4), (4, 1, 1))
    T_pred = np.tile(make_se3(rot_axis_angle([0, 1, 0], np.radians(10)), [0.02, 0, 0]), (4, 1, 1))
    e = wrist_pose_errors(T_pred, T_gt)
    assert e["wrist_trans_mean_m"] == pytest.approx(0.02)
    assert e["wrist_rot_mean_deg"] == pytest.approx(10.0, abs=1e-6)
