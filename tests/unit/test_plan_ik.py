import numpy as np
import pytest

pytest.importorskip("mujoco")

from contracts.locomanip import TaskKeyframe
from stages.locomotion.plan.candidates import Stance
from stages.locomotion.plan.ik import StanceIK


def _kf(phase, t, pos):
    T = np.eye(4)
    T[:3, 3] = pos
    return TaskKeyframe(phase=phase, time_s=t, T_object_hand=T.tolist(), contact=phase == "grasp")


def test_reachable_keyframe_solves_and_matches_forward_kinematics():
    sik = StanceIK("right")
    st = Stance(0.0, 0.0, 0.0, 0.74, 0.0)
    ref = Stance(0.0, 0.0, 0.0, 0.74, 0.0)
    T_obj = np.eye(4)
    T_obj[:3, 3] = [0.3, -0.2, 0.9]
    res, ok, why = sik.solve(
        st, T_obj, [_kf("pre_grasp", 0.0, [0.0, 0.0, 0.05]), _kf("grasp", 1.0, [0, 0, 0])]
    )
    assert ok, why
    T_w = sik.wrist_world(ref, res[1].q_arm)
    np.testing.assert_allclose(T_w[:3, 3], [0.3, -0.2, 0.9], atol=0.02)


def test_unreachable_keyframe_is_reported():
    sik = StanceIK("right")
    T_obj = np.eye(4)
    T_obj[:3, 3] = [3.0, 0.0, 0.8]
    _, ok, why = sik.solve(Stance(0, 0, 0, 0.74, 0.0), T_obj, [_kf("grasp", 0.0, [0, 0, 0])])
    assert not ok and why == "ARM_IK_INFEASIBLE"
