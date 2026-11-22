"""Manual object-relative keyframes for the golden pick-and-place task (spec §15, owner Q15)."""

from __future__ import annotations

import numpy as np

from contracts.locomanip import ManipulationTask, TaskKeyframe
from stages.locomotion.tabletop import Scenario

# (phase, time_s, wrist position in the object frame; identity orientation). Sharpa power wrap: the hand
# comes in sideways (+y), fingers pointing +x, and the thin box sits in the finger cup at PINCH_OFFSET.
PINCH_OFFSET = (-0.07, -0.05, 0.0)
SIDE_CLEAR_Y = -0.17
GOLDEN_KEYFRAMES = (
    ("pre_grasp", 0.0, (PINCH_OFFSET[0], SIDE_CLEAR_Y, 0.12)),
    ("pre_grasp", 0.5, (PINCH_OFFSET[0], SIDE_CLEAR_Y, 0.0)),
    ("grasp", 1.0, PINCH_OFFSET),
    ("lift", 2.0, (PINCH_OFFSET[0], PINCH_OFFSET[1], 0.10)),
    ("place", 3.0, (PINCH_OFFSET[0], PINCH_OFFSET[1], 0.005)),
    ("release", 4.0, (PINCH_OFFSET[0], PINCH_OFFSET[1] - 0.06, 0.085)),
)


def golden_task(sc: Scenario, hand: str = "right") -> ManipulationTask:
    kfs = []
    for phase, t, pos in GOLDEN_KEYFRAMES:
        T = np.eye(4)
        T[:3, 3] = pos
        kfs.append(
            TaskKeyframe(phase=phase, time_s=t, T_object_hand=T.tolist(), hand=hand, contact=phase == "grasp")
        )
    return ManipulationTask(
        scenario=sc.cfg["name"],
        object_name=sc.cfg["object"]["name"],
        goal_xy_on_table=tuple(sc.cfg["goal_zone"]["xy_on_table"]),
        keyframes=kfs,
    )
