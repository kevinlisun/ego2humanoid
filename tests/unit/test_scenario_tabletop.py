import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stages.locomotion.tabletop import Scenario


def test_object_rests_on_table_and_reset_is_deterministic():
    sc = Scenario.load()
    m, d = sc.build()
    for _ in range(500):
        mujoco.mj_step(m, d)
    z = d.body(sc.cfg["object"]["name"]).xpos[2]
    assert abs(z - (sc.table_top_z + sc.cfg["object"]["half_size"][2])) < 5e-3
    sc.reset_object(m, d)
    np.testing.assert_allclose(d.joint(f"{sc.cfg['object']['name']}_free").qpos[:3], sc.object_start_xyz())


def test_occupancy_blocks_table_not_start():
    sc = Scenario.load()
    g = sc.occupancy(res=0.05)
    assert g.any() and not g[int((0 + 2.0) / 0.05), int((0 + 1.0) / 0.05)]
    assert g[int((0 + 2.0) / 0.05), int((3.0 + 1.0) / 0.05)]
