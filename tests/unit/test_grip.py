"""Grip sizing: the force an object needs and the squeeze that produces it on each hand."""

import numpy as np
import pytest

from stages.retargeting.grip import (
    MAX_SQUEEZE_M,
    hold_pinch_stiffness,
    required_grip_force_n,
    squeeze_for_force,
)


def test_required_force_grows_with_mass_and_wedge():
    light = required_grip_force_n(0.03, 1.0, min_force_n=0.0)
    assert light == pytest.approx(2.0 * 0.03 * (9.81 + 3.0) / 2.0)
    assert required_grip_force_n(0.3, 1.0, min_force_n=0.0) == pytest.approx(10 * light)
    assert required_grip_force_n(0.03, 1.0, 0.5, min_force_n=0.0) == pytest.approx(2 * light)
    assert required_grip_force_n(0.03, 1.0) == 4.0  # floor


def test_squeeze_is_force_over_stiffness_within_bounds():
    assert squeeze_for_force(4.0, 400.0, 0.005) == pytest.approx(0.012)
    assert squeeze_for_force(4.0, 4000.0, 0.008) == 0.008
    assert squeeze_for_force(40.0, 400.0, 0.008) == MAX_SQUEEZE_M


def test_sharpa_pinch_is_softer_than_dex3():
    from stages.manipulation.mjcf_scene import SceneConfig, build_scene

    k = {}
    for hand in ("dex3", "sharpa"):
        q = np.zeros(build_scene(SceneConfig(hand=hand)).nu)
        k[hand] = hold_pinch_stiffness(q, "left", hand, 0.03)
    assert 50 < k["sharpa"] < k["dex3"] < 1e4
