"""Scripted right-hand top-down pinch pick-and-place on the G1+Dex3 model.

Produces a 43-DoF joint target trajectory (actuator/joint order) from Cartesian waypoints via
damped-least-squares IK. It is a test fixture proving robot, simulator and controller can
physically lift and move an object; it is not part of the production pipeline.
"""

import numpy as np

from common.transforms import rot_axis_angle
from robot.kinematics import ArmIK
from stages.manipulation.mjcf_scene import SceneConfig, build_scene

# Cube centre expressed in the wrist-yaw frame during the grasp (palm facing down).
CUBE_IN_WRIST_M = np.array([0.13, 0.06, 0.0])
R_PALM_DOWN = rot_axis_angle([1.0, 0.0, 0.0], -np.pi / 2)
FINGERS_OPEN = np.zeros(7)  # thumb0..2, index0..1, middle0..1
FINGERS_CLOSED = np.array([0.0, 0.6, -1.0, 1.2, 1.2, 1.2, 1.2])
RIGHT_FINGER_SLICE = slice(36, 43)
CONTROL_RATE_HZ = 50.0


def scripted_pick_place(
    cfg: SceneConfig,
    cube_center_m: np.ndarray,
    delta_xy_m: tuple[float, float] = (0.0, -0.12),
    hover_m: float = 0.12,
    clearance_m: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (timestamps_s (M,), q_target_rad (M, 43))."""
    model = build_scene(SceneConfig(T_sim_robot_base=cfg.T_sim_robot_base, pd=cfg.pd))
    ik = ArmIK(model, "right")
    cube = np.asarray(cube_center_m, dtype=np.float64)
    dest = cube + [delta_xy_m[0], delta_xy_m[1], 0.0]
    up = np.array([0.0, 0.0, hover_m])
    lo = np.array([0.0, 0.0, clearance_m])
    # (duration_s, cube-centre target, finger targets)
    waypoints = [
        (1.0, cube + up, FINGERS_OPEN),
        (1.0, cube + lo, FINGERS_OPEN),
        (0.8, cube + lo, FINGERS_CLOSED),
        (1.0, cube + up, FINGERS_CLOSED),
        (1.5, dest + up, FINGERS_CLOSED),
        (1.2, dest + lo, FINGERS_CLOSED),
        (0.6, dest + lo, FINGERS_OPEN),
        (0.8, dest + up, FINGERS_OPEN),
    ]
    q = np.zeros(43)

    def pose(center: np.ndarray, fingers: np.ndarray) -> np.ndarray:
        res = ik.solve(q, center - R_PALM_DOWN @ CUBE_IN_WRIST_M, R_PALM_DOWN)
        q[ik.arm_idx] = res.q_arm
        q[RIGHT_FINGER_SLICE] = fingers
        return q.copy()

    prev_c, prev_f = waypoints[0][1], waypoints[0][2]
    rows = [pose(prev_c, prev_f)]
    for dur, c, f in waypoints:
        n = int(round(dur * CONTROL_RATE_HZ))
        for k in range(1, n + 1):
            a = k / n
            s = a * a * (3 - 2 * a)
            rows.append(pose(prev_c + (c - prev_c) * s, prev_f + (f - prev_f) * s))
        prev_c, prev_f = c, f
    q_target = np.asarray(rows)
    return np.arange(len(q_target)) / CONTROL_RATE_HZ, q_target
