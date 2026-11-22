"""Anti-cheat guard: object state may only change through physics stepping."""

import mujoco
import numpy as np


class StateInjectionGuard:
    """Snapshots object qpos after each ``mj_step`` and verifies nothing wrote to it before the next.

    Any external write to a tracked object between two physics steps (teleporting) is counted
    as a violation, which downgrades the run (``state_injection_after_reset=True``).
    """

    def __init__(self, object_qpos_slices: list[slice]):
        self.slices = object_qpos_slices
        self._snap: list[np.ndarray] | None = None
        self.violations = 0

    def snapshot(self, data: mujoco.MjData) -> None:
        self._snap = [data.qpos[s].copy() for s in self.slices]

    def check(self, data: mujoco.MjData) -> None:
        if self._snap is None:
            return
        for s, ref in zip(self.slices, self._snap, strict=True):
            if not np.array_equal(data.qpos[s], ref):
                self.violations += 1
