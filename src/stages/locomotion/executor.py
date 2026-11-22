"""Closed-loop waypoint executor on top of Decoupled WBC (L.14) with an arrival gate (L.15).

Strategy from the L.3 profile: turn in place toward the next waypoint, walk straight,
then turn in place to the final heading and wait for the base to settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot.dwbc import DwbcRobot

MAX_VX = 0.5
MAX_YAW_RATE = 0.4
TURN_ONLY_ERR = 0.35  # rad: above this heading error, do not walk
FALL_Z = 0.4
POS_TOL_M = 0.08
YAW_TOL_RAD = 0.12
SHUFFLE_TOL_M = 0.05  # tighter arrival for clips: the arm has little reach margin
NEAR_GOAL_M = 0.35  # shuffle mode: inside this radius, creep omnidirectionally instead of turning around
NEAR_MAX_V = 0.25
NEAR_MIN_V = 0.15  # the gait has a dead band below this command
SETTLE_S = 0.5
SETTLE_LIN_VEL = 0.05
SETTLE_ANG_VEL = 0.1


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def base_yaw(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@dataclass
class ArrivalReport:
    arrived: bool
    fell: bool
    pos_err_m: float
    yaw_err_rad: float
    time_s: float
    reason: str = ""
    log: list = field(default_factory=list, repr=False)


def goto(
    robot: DwbcRobot,
    waypoints: list[tuple[float, float, float]],
    timeout_s: float = 60.0,
    log_every_s: float = 0.1,
    shuffle: bool = False,
) -> ArrivalReport:
    """Drive through ``waypoints`` (x, y, heading); the last one is the stance goal."""
    m, d = robot.model, robot.data
    dt = m.opt.timestep
    log, settled_since, t = [], None, 0.0
    steps_per_log = max(1, int(log_every_s / dt))
    for wi, (gx, gy, gyaw) in enumerate(waypoints):
        final = wi == len(waypoints) - 1
        tol = (SHUFFLE_TOL_M if shuffle else POS_TOL_M) if final else 0.2
        while t < timeout_s:
            x, y, yaw = d.qpos[0], d.qpos[1], base_yaw(d.qpos)
            dx, dy = gx - x, gy - y
            dist = float(np.hypot(dx, dy))
            herr = wrap(np.arctan2(dy, dx) - yaw)
            if shuffle and dist > tol and final and dist < NEAR_GOAL_M:
                c, s_ = np.cos(yaw), np.sin(yaw)
                bx, by = c * dx + s_ * dy, -s_ * dx + c * dy
                near = lambda e: float(  # noqa: E731
                    np.sign(e) * np.clip(abs(2.5 * e), NEAR_MIN_V, NEAR_MAX_V) if abs(e) > 0.015 else 0.0
                )
                vel = (near(bx), near(by), float(np.clip(1.5 * wrap(gyaw - yaw), -0.3, 0.3)))
            elif dist > tol:
                vx = 0.0 if abs(herr) > TURN_ONLY_ERR else min(MAX_VX, 0.8 * dist + 0.1)
                wz = float(np.clip(1.5 * herr, -MAX_YAW_RATE, MAX_YAW_RATE))
                vel = (vx, 0.0, wz)
            elif final:
                herr = wrap(gyaw - yaw)
                vel = (
                    0.0,
                    0.0,
                    0.0 if abs(herr) < 0.5 * YAW_TOL_RAD else float(np.clip(1.5 * herr, -0.3, 0.3)),
                )
                lin = float(np.linalg.norm(d.qvel[:2]))
                still = lin < SETTLE_LIN_VEL and abs(d.qvel[5]) < SETTLE_ANG_VEL
                ok = abs(herr) <= YAW_TOL_RAD and still
                settled_since = (settled_since if settled_since is not None else t) if ok else None
                if settled_since is not None and t - settled_since >= SETTLE_S:
                    return ArrivalReport(True, False, dist, abs(herr), t, "", log)
            else:
                break
            robot.cmd.vel[:] = vel
            robot.step()
            t += dt
            if d.qpos[2] < FALL_Z:
                return ArrivalReport(False, True, dist, 0.0, t, "FALL", log)
            if int(round(t / dt)) % steps_per_log == 0:
                log.append((t, *d.qpos[:2], base_yaw(d.qpos), *vel))
    dist = float(np.hypot(waypoints[-1][0] - d.qpos[0], waypoints[-1][1] - d.qpos[1]))
    return ArrivalReport(
        False, False, dist, abs(wrap(waypoints[-1][2] - base_yaw(d.qpos))), t, "TIMEOUT", log
    )
