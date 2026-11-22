"""Final object pose check after the place (Track RSI): is the object left upright, at rest, at height?

Pure numpy on the held object's sim trajectory. The reference is the object's own start pose, which
rests upright on the support (the human demonstration also puts it down upright), so a tilted,
still-moving or floating/sunken end state is a physics or release defect, not a video mismatch.
Thresholds are code constants, not agent-tunable.
"""

from dataclasses import dataclass

import numpy as np

REST_WINDOW_S = 0.5  # the end of the rollout over which "at rest" is measured
TILT_DEG = 15.0  # tilted past this = fallen / leaning (hint fires)
DRIFT_M = 0.005  # moved more than this in the rest window = not settled (hint fires)
HEIGHT_M = 0.01  # resting height differs from the start height by more than this (hint fires)


@dataclass(frozen=True)
class FinalPose:
    tilt_deg: float  # angle between the final and the start "up" axis of the object
    drift_m: float  # object displacement over the last REST_WINDOW_S
    height_m: float | None  # |final z - start z| on a bare table; None with a receptacle

    def defects(self) -> list[str]:
        out = []
        if self.tilt_deg > TILT_DEG:
            out.append("FINAL_TILTED")
        if self.drift_m > DRIFT_M:
            out.append("FINAL_NOT_AT_REST")
        if self.height_m is not None and self.height_m > HEIGHT_M:
            out.append("FINAL_HEIGHT")
        return out

    def detail(self) -> str:
        h = "" if self.height_m is None else f", height off by {100 * self.height_m:.1f} cm"
        return (
            f"tilt {self.tilt_deg:.0f} deg, drift {1000 * self.drift_m:.0f} mm in last {REST_WINDOW_S} s{h}"
        )


def final_pose(t: np.ndarray, object_T_sim: np.ndarray, bare_table: bool) -> FinalPose | None:
    """Metrics of one object's ``(T, 4, 4)`` pose track; None when it has fewer than 2 finite samples."""
    t = np.asarray(t, float)
    T = np.asarray(object_T_sim, float)
    ok = np.isfinite(T).all(axis=(1, 2))
    if ok.sum() < 2:
        return None
    t, T = t[ok], T[ok]
    up0, up1 = T[0, :3, 2], T[-1, :3, 2]
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(up0, up1), -1.0, 1.0))))
    window = t >= t[-1] - REST_WINDOW_S
    drift = float(np.linalg.norm(T[window][:, :3, 3].max(axis=0) - T[window][:, :3, 3].min(axis=0)))
    height = abs(float(T[-1, 2, 3] - T[0, 2, 3])) if bare_table else None
    return FinalPose(tilt, drift, height)
