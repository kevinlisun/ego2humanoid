import numpy as np

from stages.manipulation.mjcf_scene import BoxObject, SceneConfig
from stages.reconstruction.scene.scene_state import load_scene_state, record_scene_state, settle_report


def _cfg(*objs):
    return SceneConfig(table_top_z_m=0.8, table_center_xy_m=(0.5, 0.0), objects=tuple(objs))


def test_resting_object_is_stable_and_floating_object_is_not(tmp_path):
    rest = BoxObject("rest", (0.03, 0.03, 0.05), (0.5, 0.2, 0.8505), mass_kg=0.2)
    air = BoxObject("air", (0.03, 0.03, 0.05), (0.5, -0.2, 1.2), mass_kg=0.2)
    state = record_scene_state(_cfg(rest, air), duration_s=0.5)
    assert state.pos_m.shape == (len(state.time_s), 2, 3)
    report = settle_report(state)
    assert report["objects"]["rest"]["stable"]
    assert not report["objects"]["air"]["stable"]
    assert not report["all_stable"]
    path = tmp_path / "scene_state.npz"
    state.save(path)
    loaded = load_scene_state(path)
    assert loaded.names == ("rest", "air")
    assert np.allclose(loaded.pos_m, state.pos_m)
