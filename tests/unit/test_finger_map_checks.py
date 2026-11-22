import json

import numpy as np

from robot.asset import JOINT_ORDER_PATH
from robot.finger_map import load_finger_config, map_curls_to_dex3
from stages.manipulation.checks import check_rollout


def test_finger_map_endpoints_and_grouping():
    cfg = load_finger_config()
    assert np.allclose(map_curls_to_dex3(np.zeros(5)), cfg["open"])
    assert np.allclose(map_curls_to_dex3(np.ones(5)), cfg["closed"])
    # ring/pinky curl contributes to the single robot middle finger
    half = map_curls_to_dex3(np.array([0, 0, 1.0, 1.0, 0]))
    assert np.allclose(half[5:], 0.8)
    assert np.allclose(half[:5], 0.0)


def test_finger_map_left_order_matches_robot_joints():
    names = json.loads(JOINT_ORDER_PATH.read_text())["joint_names"]
    cfg = load_finger_config()
    for side, start in (("right", 36), ("left", 22)):
        expected = [f"{side}_hand_{j}_joint" for j in cfg["joint_order"][side]]
        assert names[start : start + 7] == expected


def _arrays(k=100):
    t = np.linspace(0, 2, k)
    obj = np.tile(np.eye(4), (k, 1, 1, 1))
    return {
        "timestamps_s": t,
        "q_command_rad": np.zeros((k, 43)),
        "q_actual_rad": np.zeros((k, 43)),
        "dq_actual_rad_s": np.zeros((k, 43)),
        "wrist_actual_T_sim": np.tile(np.eye(4), (k, 2, 1, 1)),
        "object_T_sim": obj,
    }


def test_checks_flag_nonfinite_and_limits():
    lim = np.tile([-1.0, 1.0], (43, 1))
    a = _arrays()
    assert not check_rollout(a, lim).passed  # a static object is not a pick-place
    a["q_actual_rad"][10, 3] = 2.0
    assert "JOINT_LIMIT_VIOLATION" in check_rollout(a, lim).error_codes
    a["q_actual_rad"][10, 3] = np.nan
    assert check_rollout(a, lim).error_codes == ["NON_FINITE_STATE"]
