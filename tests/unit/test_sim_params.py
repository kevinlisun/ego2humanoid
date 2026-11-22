import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from contracts.sim_params import SimParams
from skills.params.resolve import diff_params, needs_plan, params_hash, resolve_params

ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_matches_model_defaults():
    on_disk = yaml.safe_load((ROOT / "configs" / "sim_params" / "default.yaml").read_text())
    assert on_disk == json.loads(SimParams().model_dump_json())


def test_overlay_merge_and_precedence(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("contact: {impratio: 20, noslip_iterations: 3}\n")
    p = resolve_params([overlay], ["contact.impratio=30", "pd.groups.hand.kp=10"], env={})
    assert p.contact.impratio == 30 and p.contact.noslip_iterations == 3
    assert p.pd.groups.hand.kp == 10 and p.pd.groups.hand.kd == 0.4  # untouched sibling keeps default


def test_env_overlay_applies_before_files(tmp_path):
    env_file = tmp_path / "e.yaml"
    env_file.write_text("contact: {impratio: 20}\n")
    assert resolve_params(env={"E2H_SIM_PARAMS": str(env_file)}).contact.impratio == 20


def test_accepted_params_sit_below_env_overlay(tmp_path):
    acc = tmp_path / "accepted.yaml"
    acc.write_text("contact: {impratio: 20, noslip_iterations: 3}\n")
    env_file = tmp_path / "e.yaml"
    env_file.write_text("contact: {impratio: 30}\n")
    p = resolve_params(env={"E2H_SIM_PARAMS": str(env_file)}, accepted=acc)
    assert p.contact.impratio == 30 and p.contact.noslip_iterations == 3
    assert resolve_params(env={}, accepted=tmp_path / "missing.yaml") == resolve_params(env={})


@pytest.mark.parametrize(
    "pair",
    [
        "contact.impratio=1000",
        "contact.solref=[0.001,1.0]",
        "contact.typo=1",
        "contact.solimp=[0.99,0.9,0.001]",
    ],
)
def test_invalid_values_are_rejected(pair):
    with pytest.raises(ValidationError):
        resolve_params(sets=[pair], env={})


def test_hash_is_stable_and_diff_tiers():
    base = SimParams()
    assert params_hash(base) == params_hash(SimParams())
    sim_only = resolve_params(sets=["contact.impratio=20"], env={})
    assert params_hash(sim_only) != params_hash(base)
    assert not needs_plan(diff_params(base, sim_only))
    plan = resolve_params(sets=["contact.min_slide_friction=1.5", "objects.held.mass_kg=0.2"], env={})
    changes = {c.path: c for c in diff_params(base, plan)}
    assert needs_plan(changes.values())
    assert changes["objects.held.mass_kg"].affects == {"sim", "plan"}


def test_default_params_reproduce_scene_config_defaults():
    from stages.manipulation.mjcf_scene import ObjectContact, PDConfig, SceneConfig

    p = SimParams()
    assert ObjectContact.from_params(p) == ObjectContact()
    assert PDConfig.from_params(p) == PDConfig()
    built, today = SceneConfig.from_params(p, hand="sharpa"), SceneConfig(hand="sharpa")
    for name in (
        "table_friction",
        "pd",
        "object_contact",
        "noslip_iterations",
        "impratio",
        "cone",
        "integrator",
    ):
        assert getattr(built, name) == getattr(today, name), name


def test_params_reach_the_compiled_model():
    import mujoco

    from stages.manipulation.mjcf_scene import SceneConfig, build_scene_spec

    def opts(hand, *sets):
        cfg = SceneConfig.from_params(resolve_params(sets=sets, env={}), hand=hand)
        return build_scene_spec(cfg).compile().opt

    for hand in ("dex3", "sharpa"):
        base = opts(hand)
        assert (base.impratio, base.cone, base.noslip_iterations) == (10.0, mujoco.mjtCone.mjCONE_ELLIPTIC, 5)
        new = opts(hand, "contact.impratio=25", "contact.cone=pyramidal", "pd.timestep_s=0.001")
        assert new.impratio == 25 and new.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and new.timestep == 0.001


def _box(name, pos=(0.5, 0.0, 0.8), mesh=False):
    import numpy as np

    from stages.manipulation.mjcf_scene import BoxObject

    kw = {}
    if mesh:
        kw["mesh_vertices"] = np.array([[0.02, 0, 0], [0, 0.01, 0], [0, 0, 0.03], [-0.02, 0, 0]])
        kw["mesh_faces"] = np.array([[0, 1, 2], [1, 2, 3]])
    return BoxObject(name, (0.02, 0.02, 0.02), pos, **kw)


def test_legacy_env_maps_to_trajectory_and_cli_wins():
    env = {"E2H_SQUEEZE": "0.015", "E2H_SCENE_OBJECTS": "0", "E2H_PLACE_DWELL": "20"}
    p = resolve_params(env=env)
    assert p.trajectory.squeeze_m == 0.015 and p.trajectory.scene_objects is False
    assert p.trajectory.place_dwell_frames == 20
    assert resolve_params(sets=["trajectory.squeeze_m=0.01"], env=env).trajectory.squeeze_m == 0.01


def test_scene_overrides_drop_move_and_reject_unknown():
    import numpy as np

    from skills.params.scene_overrides import apply_scene_overrides

    objs = [_box("tray"), _box("cup", mesh=True)]
    p = resolve_params(
        sets=[
            "objects.tray.enabled=false",
            "objects.cup.pos_offset_m=[0.01,0,0]",
            "objects.cup.yaw_offset_deg=90",
        ],
        env={},
    )
    out = apply_scene_overrides(objs, p)
    assert [o.name for o in out] == ["cup"] and abs(out[0].pos_m[0] - 0.51) < 1e-9
    assert np.allclose(out[0].mesh_vertices[0], [0, 0.02, 0], atol=1e-9)  # +x turned 90 deg counter-clockwise
    with pytest.raises(ValueError, match="unknown object selector"):
        apply_scene_overrides(objs, resolve_params(sets=["objects.mug.enabled=false"], env={}))
