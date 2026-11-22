"""Hand subtree lookup and the planning-time hand clearance margin."""

import numpy as np
import pytest

from robot.collision import ArmCollisionChecker, hand_geoms
from stages.manipulation.mjcf_scene import SceneConfig, build_scene

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def model():
    return build_scene(SceneConfig())


def test_hand_geoms_are_per_side_and_disjoint(model):
    left, right = hand_geoms(model, "left"), hand_geoms(model, "right")
    assert len(left) > 0 and len(left) == len(right)
    assert not set(left.tolist()) & set(right.tolist())


def test_hand_clearance_flags_frames_that_plain_contact_misses(model):
    q = np.zeros((1, model.nu))
    assert not ArmCollisionChecker(model).colliding_frames(q).any()
    # A margin wider than the shoulder span must reject the same frame, proving the margin is applied.
    wide = ArmCollisionChecker(model, hand_clearance_m=1.0)
    assert wide.colliding_frames(q).all()
