"""Actuator layout of the robot, derived from the compiled model instead of hard-coded indices.

The Dex3 build has 43 actuators (29 body/arm + 7 fingers per hand); the Sharpa Wave build has 73
(29 + 22 per hand). Everything downstream addresses joints through this layout, so swapping hands
does not need index arithmetic anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robot.sharpa import FINGERS

DEX3_FINGER_TOKENS = ("_hand_thumb_", "_hand_index_", "_hand_middle_")
SHARPA_FINGER_TOKENS = tuple(f"_{f}_" for f in FINGERS)


@dataclass(frozen=True)
class RobotLayout:
    hand: str
    actuator_names: tuple[str, ...]

    @staticmethod
    def from_model(model: mujoco.MjModel) -> RobotLayout:
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)
        )
        hand = "sharpa" if any(n.endswith("_MCP_FE") for n in names) else "dex3"
        return RobotLayout(hand, names)

    @property
    def dof(self) -> int:
        return len(self.actuator_names)

    def finger_idx(self, side: str) -> np.ndarray:
        """Actuator indices of one hand's finger joints, in actuator order."""
        tokens = SHARPA_FINGER_TOKENS if self.hand == "sharpa" else DEX3_FINGER_TOKENS
        return np.array(
            [
                i
                for i, n in enumerate(self.actuator_names)
                if n.startswith(f"{side}_") and any(t in n for t in tokens)
            ]
        )

    def joint_idx(self, names: list[str]) -> np.ndarray:
        return np.array([self.actuator_names.index(n) for n in names])
