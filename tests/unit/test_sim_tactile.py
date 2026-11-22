import mujoco
import numpy as np
import pytest

from robot.floating_hand import build_floating_hand_spec
from stages.manipulation.sim_tactile import REGION_NAMES, SimTactile, support_touch
from stages.reconstruction.hands import tactile as human_tactile
from stages.retargeting.tactile_target import build_tactile_target, compare_tactile


def _hand_with_box(side="left"):
    spec = build_floating_hand_spec(side)
    probe = spec.compile()
    data = mujoco.MjData(probe)
    mujoco.mj_forward(probe, data)
    tip = data.xpos[mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_index_1_link")]
    body = spec.worldbody.add_body(name="box", pos=list(tip))
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03] * 3, mass=0.1)
    model = spec.compile()
    return model, mujoco.MjData(model)


def test_sim_tactile_reads_contact_force_and_flags_missing_regions():
    model, data = _hand_with_box("left")
    tac = SimTactile(model)
    mujoco.mj_forward(model, data)
    tac.update(model, data)
    tac.update(model, data)
    a = tac.arrays()
    assert a["normal_force_n"].shape == (2, 2, len(REGION_NAMES))
    assert a["contact"].dtype == bool
    assert not a["normal_force_n"][:, 1].any()  # right hand absent from the model
    missing = [i for i, r in enumerate(REGION_NAMES) if r.startswith(("ring", "pinky"))]
    assert not a["normal_force_n"][:, 0, missing].any()
    assert [r for r, h in zip(REGION_NAMES, a["has_region"], strict=True) if not h] == [
        REGION_NAMES[i] for i in missing
    ]
    assert a["normal_force_n"][:, 0].sum() > 0 and a["contact"][:, 0].any()
    assert set(a["object_name"][:, 0]) == {"box"} and set(a["object_name"][:, 1]) == {""}


def test_sim_tactile_partner_filter_and_save(tmp_path):
    model, data = _hand_with_box("left")
    mujoco.mj_forward(model, data)
    tac = SimTactile(model, partner="table")
    tac.update(model, data)
    assert not tac.arrays()["contact"].any()
    path = tac.save(tmp_path / "sim_tactile.npz")
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {
            "normal_force_n",
            "tangential_force_n",
            "contact",
            "region_names",
            "has_region",
            "object_name",
            "support_normal_force_n",
            "support_contact",
        }


def test_sim_tactile_keeps_the_support_share_of_the_force_apart():
    model, data = _hand_with_box("left")
    mujoco.mj_forward(model, data)
    grip = SimTactile(model, partner="object+table")  # the box is an object here
    plane = SimTactile(model, partner="object+table", table_name="box")  # ... and the support here
    grip.update(model, data)
    plane.update(model, data)
    g, p = grip.arrays(), plane.arrays()
    assert g["normal_force_n"].sum() > 0 and not g["support_normal_force_n"].any()
    np.testing.assert_allclose(p["support_normal_force_n"], p["normal_force_n"])
    assert p["support_contact"][0, 0].any()
    touch = support_touch(p, np.array([0.0]), "left")
    assert touch["first_s"] == 0.0 and touch["peak_n"] > 0 and touch["regions"]
    assert support_touch(g, np.array([0.0]), "left") is None


def test_sim_tactile_covers_every_region_on_the_five_finger_hand():
    pytest.importorskip("mujoco")
    from robot.sharpa import HAND_MJCF
    from stages.manipulation.mjcf_scene import SceneConfig, build_scene
    from stages.manipulation.sim_tactile import detect_hand

    if not all(p.exists() for p in HAND_MJCF.values()):
        pytest.skip("run scripts/fetch_sharpa_assets.sh")
    model = build_scene(SceneConfig(table_top_z_m=0.8, hand="sharpa"))
    assert detect_hand(model) == "sharpa"
    tac = SimTactile(model, partner="object+table")
    assert tac.hand == "sharpa" and tac.has_region.all()
    covered = {(h, r) for h, r in tac._geom_slot.values()}
    assert covered == {(h, r) for h in (0, 1) for r in range(len(REGION_NAMES))}


def test_sim_tactile_regions_match_the_human_schema():
    assert REGION_NAMES == human_tactile.REGION_NAMES  # compare_tactile matches sim/human by name


def test_tactile_target_scales_and_compares():
    names = ["thumb", "index", "palm"]
    prob = np.zeros((4, 2, 3))
    prob[1:3, 0, 0] = 0.9
    prob[2, 0, 1] = 0.7
    force = prob * 4.0
    t = build_tactile_target(names, prob, force, force_max_n=8.0)
    assert t["contact_target"].sum() == 3
    assert t["force_target_n"].max() == 8.0  # human max maps to Dex3 max
    sim = {
        "contact": t["contact_target"].copy(),
        "normal_force_n": t["force_target_n"] * 0.5,
        "region_names": np.array(names),
        "has_region": np.ones(3, bool),
    }
    sim["contact"][2, 0, 0] = False
    r = compare_tactile(sim, t)
    assert r["thumb"]["precision"] == 1.0 and r["thumb"]["recall"] == 0.5
    assert r["index"]["f1"] == 1.0
    assert np.isnan(r["palm"]["f1"])
