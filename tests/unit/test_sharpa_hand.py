"""Sharpa Wave five-finger hand: model assembly, actuator layout and MANO curl mapping."""

import numpy as np
import pytest

from robot.finger_map import map_curls_to_sharpa, sharpa_joint_names

mujoco = pytest.importorskip("mujoco")

from robot.layout import RobotLayout
from robot.sharpa import HAND_MJCF, fingertip_geom, fingertip_site
from stages.manipulation.mjcf_scene import SceneConfig, build_scene

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in HAND_MJCF.values()), reason="run scripts/fetch_sharpa_assets.sh"
)


@pytest.fixture(scope="module")
def sharpa_model():
    return build_scene(SceneConfig(table_top_z_m=0.8, hand="sharpa"))


def test_build_scene_hands_out_independent_copies_of_a_cached_compile():
    cfg = SceneConfig(table_top_z_m=0.8)
    a = build_scene(cfg)
    a.geom_group[:] = 5
    b = build_scene(cfg)
    assert b is not a and not (b.geom_group == 5).all()
    table = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, "table")
    lower = build_scene(SceneConfig(table_top_z_m=0.7))
    assert lower.body_pos[table, 2] == pytest.approx(b.body_pos[table, 2] - 0.1)


def test_layout_matches_hand(sharpa_model):
    dex3 = RobotLayout.from_model(build_scene(SceneConfig(table_top_z_m=0.8)))
    assert (dex3.hand, dex3.dof) == ("dex3", 43)
    assert dex3.finger_idx("left").tolist() == list(range(22, 29))

    layout = RobotLayout.from_model(sharpa_model)
    assert (layout.hand, layout.dof) == ("sharpa", 29 + 2 * 22)
    for side in ("left", "right"):
        idx = layout.finger_idx(side)
        assert len(idx) == 22
        assert [layout.actuator_names[i] for i in idx] == sharpa_joint_names(side)


def test_fingertip_sites_and_pads_exist(sharpa_model):
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for obj, name in (
                (mujoco.mjtObj.mjOBJ_SITE, fingertip_site(side, finger)),
                (mujoco.mjtObj.mjOBJ_GEOM, fingertip_geom(side, finger)),
            ):
                assert mujoco.mj_name2id(sharpa_model, obj, name) >= 0, name


def test_curl_mapping_is_monotone_and_within_joint_limits(sharpa_model):
    names = sharpa_joint_names("right")
    jid = [mujoco.mj_name2id(sharpa_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    lo, hi = sharpa_model.jnt_range[jid, 0], sharpa_model.jnt_range[jid, 1]

    curls = np.linspace(0.0, 1.0, 11)[:, None] * np.ones(5)
    q = map_curls_to_sharpa(curls, "right")
    assert q.shape == (11, 22)
    assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
    assert np.all(np.diff(q, axis=0) >= -1e-9)  # closing a human finger never extends a robot joint
    # Out-of-range curls are clipped rather than extrapolated out of the joint limits.
    assert np.allclose(map_curls_to_sharpa(np.full(5, 3.0), "right"), q[-1])


def test_closed_hand_brings_thumb_and_index_tips_together(sharpa_model):
    layout = RobotLayout.from_model(sharpa_model)
    data = mujoco.MjData(sharpa_model)
    qadr = sharpa_model.jnt_qposadr[sharpa_model.actuator_trnid[:, 0]]
    tips = [
        mujoco.mj_name2id(sharpa_model, mujoco.mjtObj.mjOBJ_SITE, fingertip_site("right", f))
        for f in ("thumb", "index")
    ]

    gaps = []
    for curl in (0.0, 1.0):
        data.qpos[:] = 0.0
        data.qpos[qadr[layout.finger_idx("right")]] = map_curls_to_sharpa(np.full(5, curl), "right")
        mujoco.mj_forward(sharpa_model, data)
        gaps.append(float(np.linalg.norm(data.site_xpos[tips[0]] - data.site_xpos[tips[1]])))
    assert gaps[1] < gaps[0]
