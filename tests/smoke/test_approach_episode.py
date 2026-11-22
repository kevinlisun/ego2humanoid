"""L.14-L.16: plan -> walk -> arrive -> drive the arm through the golden keyframes without falling."""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("mujoco")

from robot.dwbc import DWBC_ROBOT_DIR
from stages.locomotion.episode import run_episode
from stages.locomotion.plan.golden import golden_task
from stages.locomotion.tabletop import Scenario

pytestmark = pytest.mark.skipif(not (DWBC_ROBOT_DIR / "policy").exists(), reason="submodule missing")


def test_golden_sharpa_pick_and_place_standing():
    sc = Scenario.load()
    r = run_episode(sc, golden_task(sc))
    assert r["ok"], r.get("failure")
    assert r["arrival"].arrived and r["visible_fraction_measured"] >= 0.8
    assert r["min_pelvis_z"] > 0.65 and r["max_tilt_rad"] < 0.3
    assert max(r["wrist_track_err_m"]) < 0.1
    goal = sc.goal_xy()
    assert np.linalg.norm(r["object_final_xyz"][:2] - goal) < 0.05
