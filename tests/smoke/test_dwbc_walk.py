"""Decoupled WBC smoke test: the policy stands and walks the G1 on our MJCF asset."""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
mujoco = pytest.importorskip("mujoco")

from robot.asset import MJCF_PATH
from robot.dwbc import DWBC_ROBOT_DIR, DwbcRobot, gravity_in_body

pytestmark = pytest.mark.skipif(
    not (DWBC_ROBOT_DIR / "policy").exists(), reason="GR00T-WholeBodyControl submodule not checked out"
)


def test_gravity_in_body_identity():
    np.testing.assert_allclose(gravity_in_body(np.array([1.0, 0, 0, 0])), [0, 0, -1], atol=1e-9)


def test_stand_then_walk_forward():
    m = mujoco.MjSpec.from_file(str(MJCF_PATH)).compile()
    d = mujoco.MjData(m)
    r = DwbcRobot(m, d)
    r.reset_pose(pelvis_z=0.79)
    for i in range(int(6 / m.opt.timestep)):
        r.cmd.vel[:] = (0, 0, 0) if i * m.opt.timestep < 2 else (0.4, 0, 0)
        r.step()
    assert 0.65 < d.qpos[2] < 0.85
    assert d.qpos[0] > 0.5
