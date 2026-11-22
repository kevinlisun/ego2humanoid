import numpy as np

from skills.physics_fidelity import fidelity_report, format_markdown

TOP = 0.8


def _state(T=100):
    t = np.arange(T) / 50.0
    xpos = np.zeros((T, 3, 3))
    xpos[:, 1] = [0.0, 0.0, TOP + 0.03]  # perceived
    xpos[:, 2] = [0.2, 0.0, TOP + 0.02]  # scene_box
    xquat = np.tile([1.0, 0, 0, 0], (T, 3, 1))
    return {"t": t, "body_names": np.array(["world", "perceived", "scene_box"]), "xpos": xpos, "xquat": xquat}


def _run(s, **kw):
    return fidelity_report(s, ["perceived", "scene_box"], TOP, {"perceived": 0.03, "scene_box": 0.02}, **kw)


def test_clean_rest_ok():
    r = _run(_state())
    assert r["ok"] and not r["problems"]
    assert r["events"]["lift_start_s"] is None
    assert "OK" in format_markdown(r, "clip0")


def test_sink_flagged():
    s = _state()
    s["xpos"][50:, 2, 2] = TOP - 0.02
    r = _run(s)
    assert not r["ok"] and r["max_sink_m"] > 0.03


def test_drift_flagged():
    s = _state()
    s["xpos"][:, 2, 0] += np.linspace(0, 0.05, 100)
    r = _run(s)
    assert any("drifts" in p for p in r["problems"])
    assert any("moves" in p for p in r["problems"])


def test_nan_flagged():
    s = _state()
    s["xpos"][10, 1, 0] = np.nan
    s["pad_force_N"] = np.full((100, 2), np.nan)
    r = _run(s)
    assert not r["ok"] and not r["finite"] and r["pad_force"]["nan"]


def test_pad_force_range():
    s = _state()
    s["pad_force_N"] = np.full((100, 2), 9.0)
    assert not _run(s)["pad_force"]["in_range"]


def test_lift_events():
    s = _state()
    s["xpos"][20:60, 1, 2] += 0.05
    c = np.zeros(100, bool)
    c[10:60] = True
    s["object_contact"] = c
    ev = _run(s)["events"]
    assert ev["lift_start_s"] == 0.4 and ev["lift_end_s"] == 59 / 50.0
    assert ev["release_s"] == 59 / 50.0
