"""L.14/L.15: the executor reaches a stance goal within the arrival tolerances."""

import pytest

pytest.importorskip("onnxruntime")
mujoco = pytest.importorskip("mujoco")

from robot.asset import MJCF_PATH
from robot.dwbc import DWBC_ROBOT_DIR, DwbcRobot
from stages.locomotion.executor import POS_TOL_M, YAW_TOL_RAD, goto

pytestmark = pytest.mark.skipif(not (DWBC_ROBOT_DIR / "policy").exists(), reason="submodule missing")


def test_goto_arrives_with_requested_heading():
    m = mujoco.MjSpec.from_file(str(MJCF_PATH)).compile()
    d = mujoco.MjData(m)
    r = DwbcRobot(m, d)
    r.reset_pose(pelvis_z=0.79)
    rep = goto(r, [(1.0, 0.5, 0.0), (1.5, 1.0, 1.57)], timeout_s=40)
    assert rep.arrived, rep.reason
    assert rep.pos_err_m <= POS_TOL_M and rep.yaw_err_rad <= YAW_TOL_RAD
