import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco")

from stages.manipulation.events import detect_pick_place
from stages.manipulation.guard import StateInjectionGuard
from stages.manipulation.mjcf_scene import BoxObject, SceneConfig
from stages.manipulation.rollout import run_rollout
from stages.manipulation.scripted import scripted_pick_place

CUBE = np.array([0.38, -0.12, 0.825])


def _cfg() -> SceneConfig:
    base = np.eye(4)
    base[2, 3] = 0.793
    return SceneConfig(
        T_sim_robot_base=base,
        table_top_z_m=0.80,
        objects=(BoxObject("cube", (0.025,) * 3, tuple(CUBE), mass_kg=0.1),),
    )


def test_scripted_fixture_passes_events_on_most_seeds():
    cfg = _cfg()
    t, q = scripted_pick_place(cfg, CUBE)
    ok = 0
    for seed in range(5):
        r = run_rollout(cfg, t, q, seed=seed)
        ev = detect_pick_place(r.arrays["timestamps_s"], r.arrays["object_T_sim"][:, 0])
        ok += ev.success
    assert ok >= 3


def test_guard_counts_injected_write():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    g = StateInjectionGuard([slice(0, 7)])
    mujoco.mj_step(m, d)
    g.snapshot(d)
    g.check(d)
    assert g.violations == 0
    d.qpos[0] += 0.5
    g.check(d)
    assert g.violations == 1


def test_replay_engine_writes_validated_bundle(tmp_path):
    from stages.manipulation.engine import replay_episode

    cfg = _cfg()
    t, q = scripted_pick_place(cfg, CUBE)
    out = replay_episode(cfg, t, q, tmp_path, "ep_syn", render=True)
    assert out.bundle.rollouts_attempted == 5
    assert out.bundle.input_source == out.report.input_source == "synthetic"
    assert out.report.task_success and not out.report.accepted_for_dataset
    assert out.bundle.replay_validity == "PHYSICS_VALIDATED"
    assert (tmp_path / "actual_states.npz").exists() and (tmp_path / "manifest.json").exists()
    assert (tmp_path / "render.mp4").stat().st_size > 1000
