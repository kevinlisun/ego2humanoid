"""L.8-L.12: the planner finds a feasible stance for the golden task on tabletop_v1."""

import pytest

pytest.importorskip("mujoco")

from stages.locomotion.plan.golden import golden_task
from stages.locomotion.plan.rank import plan_stance
from stages.locomotion.tabletop import Scenario


def test_golden_task_has_a_feasible_stance():
    sc = Scenario.load()
    goal, failure, rejected = plan_stance(sc, golden_task(sc))
    assert failure is None and goal is not None, rejected
    assert goal.visible_fraction >= 0.4
    assert rejected["ik"] > 0
