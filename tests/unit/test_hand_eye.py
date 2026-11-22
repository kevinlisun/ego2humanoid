"""Hand-eye closure: the offset loop and the window that confines it to the grasp."""

import numpy as np
import pytest

from stages.retargeting.hand_eye import MAX_OFFSET_M, align_pinch, grasp_window, settle_approach


def test_grasp_window_is_one_on_the_hold_and_zero_away_from_it():
    w = grasp_window(200, np.arange(80, 120))
    assert np.allclose(w[80:120], 1.0)
    assert w[0] == 0.0 and w[-1] == 0.0
    assert 0.0 < w[70] < 1.0  # fades in rather than stepping, so the wrist is not jerked


def test_align_pinch_drives_the_robot_pinch_onto_the_target():
    # The robot's pinch point lags the commanded hand by a fixed bias and only follows half of it.
    bias = np.array([0.03, -0.02, 0.05])
    target = np.zeros((60, 3))
    weight = grasp_window(60, np.arange(20, 40))

    def retarget(offset):
        return offset

    def measure(offset):
        return bias + 0.5 * offset

    _, offset, rms = align_pinch(retarget, measure, target, weight, max_iters=12, tol_m=1e-4)
    assert rms < 1e-3
    hold = weight > 0.99
    assert np.allclose(offset[hold], -2 * bias, atol=1e-3)
    assert np.allclose(offset[weight == 0.0], 0.0)  # the rest of the clip is left untouched


def test_align_pinch_never_moves_the_hand_further_than_the_arm_can_follow():
    target = np.full((30, 3), 10.0)  # unreachable, so the loop would run away unclipped
    weight = np.ones(30)
    _, offset, _ = align_pinch(lambda o: o, lambda o: np.zeros((30, 3)), target, weight)
    assert np.abs(offset).max() <= MAX_OFFSET_M + 1e-9


def test_settle_approach_arrives_early_from_above():
    n, cs = 60, 40
    target = np.zeros((n, 3))
    target[:cs, 1] = np.linspace(-0.3, 0.0, cs)  # sideways sweep that reaches the grasp point at cs
    out, w = settle_approach(target, cs, approach_frames=30, settle_frames=9, height_m=0.03)
    assert np.allclose(out[cs:], target[cs:]) and np.allclose(out[: cs - 30], target[: cs - 30])
    assert np.allclose(out[cs - 9 : cs, :2], 0.0)  # over the object for the last 0.3 s
    assert np.all(np.diff(out[cs - 9 : cs + 1, 2]) < 0)  # coming straight down onto it
    assert abs(out[cs - 9, 2] - 0.03) < 1e-9
    assert w[cs - 30] < w[cs - 25] and np.all(w[cs - 20 : cs] == 1.0) and w[cs] == 0.0


def test_settle_approach_waits_at_the_lowest_close_point():
    n, cs = 60, 40
    target = np.zeros((n, 3))
    target[cs:, 2] = np.r_[np.linspace(0.0, -0.015, 12), np.linspace(-0.015, 0.1, n - cs - 12)]
    out, _ = settle_approach(target, cs, approach_frames=30, settle_frames=9, height_m=0.03, close_frames=12)
    assert np.allclose(out[cs : cs + 12, 2], -0.015)  # the fingers close at the bottom, not on the way down
    assert np.allclose(out[cs + 11 :], target[cs + 11 :])  # the lift is untouched
    assert abs(out[cs - 9, 2] - (-0.015 + 0.03)) < 1e-9


def test_settle_approach_hovers_over_the_object_through_a_long_human_dwell():
    # The human hand reaches the object at frame 10 and rests its fingers on it until the close at 70.
    n, cs = 90, 70
    target = np.zeros((n, 3))
    target[:10, 0] = np.linspace(0.3, 0.0, 10)
    out, w = settle_approach(target, cs, approach_frames=30, settle_frames=9, height_m=0.03, hover_z_m=0.05)
    assert np.all(out[10 : cs - 9, 2] >= 0.05 - 1e-9)  # open fingers kept above the top, not on it
    assert np.all(np.diff(out[cs - 9 : cs + 1, 2]) < 0) and out[cs, 2] == 0.0  # then straight down
    assert out[0, 2] == 0.0 and np.all(w[10:cs] == 1.0)


def test_limit_carry_acceleration_smooths_a_jerk_but_keeps_the_ends():
    from stages.retargeting.hand_eye import limit_carry_acceleration

    path = np.zeros((90, 3))
    path[45:, 0] = 0.02  # a 2 cm step: ~20 m/s^2 at 30 Hz
    out, sigma = limit_carry_acceleration(path, 20, 70, 1 / 30, max_accel_m_s2=3.0, max_sigma_frames=12)
    acc = np.linalg.norm(np.diff(out[19:71], 2, axis=0), axis=1) * 900
    assert sigma > 0 and acc.max() <= 3.0 + 1e-9
    np.testing.assert_allclose(out[:20], path[:20])
    np.testing.assert_allclose(out[70:], path[70:])


def test_release_motion_holds_lifts_and_rejoins():
    from stages.retargeting.hand_eye import release_motion

    path = np.zeros((80, 3))
    path[:, 0] = np.linspace(0, 0.4, 80)  # the human hand keeps moving after the release
    out, w, end = release_motion(
        path, 30, dwell_frames=10, retreat_frames=8, retreat_height_m=0.04, blend_frames=10
    )
    np.testing.assert_allclose(out[31:41], np.tile(path[30], (10, 1)))
    assert out[48, 2] == pytest.approx(0.04) and np.allclose(out[41:49, :2], path[30, :2])
    assert end == 59 and (w[31:49] == 1).all() and w[50] < 1 and (w[end:] == 0).all()
    np.testing.assert_allclose(out[end:], path[end:])
