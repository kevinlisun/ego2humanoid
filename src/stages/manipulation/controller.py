"""Joint-target controller: linear interpolation of ``q_target`` clipped to joint limits."""

import mujoco
import numpy as np


class PositionController:
    """Feeds position-actuator setpoints; actuator order == authoritative joint order."""

    def __init__(self, model: mujoco.MjModel, timestamps_s: np.ndarray, q_target_rad: np.ndarray):
        if np.any(np.diff(timestamps_s) <= 0):
            raise ValueError("timestamps_s must be strictly increasing")
        self.t = np.asarray(timestamps_s, dtype=np.float64)
        self.q = np.asarray(q_target_rad, dtype=np.float64)
        joint_ids = model.actuator_trnid[: model.nu, 0]
        self.lo = model.jnt_range[joint_ids, 0]
        self.hi = model.jnt_range[joint_ids, 1]

    def q_target(self, t_s: float) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.t, t_s, side="right"), 1, len(self.t) - 1)
        a = np.clip((t_s - self.t[idx - 1]) / (self.t[idx] - self.t[idx - 1]), 0.0, 1.0)
        return self.q[idx - 1] * (1 - a) + self.q[idx] * a

    def command(self, t_s: float) -> np.ndarray:
        return np.clip(self.q_target(t_s), self.lo, self.hi)
