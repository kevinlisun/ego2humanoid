"""L.9: stationary arm IK per keyframe for a candidate stance (feet/base/waist fixed)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from contracts.locomanip import TaskKeyframe
from robot.asset import load_fixed_base_spec
from robot.kinematics import ArmIK
from stages.locomotion.plan.candidates import Stance

POS_TOL_M = 0.02
ROT_TOL_RAD = 0.25
MAX_KEYFRAME_STEP_RAD = 2.0
LIMIT_MARGIN_RAD = 0.02
FIXED_PELVIS_Z = 0.793  # pelvis height of the fixed-base model


@dataclass
class KeyframeIK:
    phase: str
    q_arm: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    feasible: bool


class StanceIK:
    """Solves wrist targets given in the world frame from the pelvis pose of a stance."""

    def __init__(self, side: str = "right"):
        self.model = load_fixed_base_spec().compile()
        self.ik = ArmIK(self.model, side)
        joints = self.model.actuator_trnid[:, 0].tolist()
        self.waist_pitch_idx = joints.index(self.model.joint("waist_pitch_joint").id)

    def world_to_pelvis_frame(self, stance: Stance, T_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(stance.yaw), np.sin(stance.yaw)
        R_wp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        p = R_wp.T @ (T_world[:3, 3] - np.array([stance.x, stance.y, stance.height_m]))
        return p + np.array([0, 0, FIXED_PELVIS_Z]), R_wp.T @ T_world[:3, :3]

    def solve(
        self,
        stance: Stance,
        T_world_object: np.ndarray,
        keyframes: list[TaskKeyframe],
        limit_margin: float = LIMIT_MARGIN_RAD,
    ):
        """IK for each keyframe, warm-started from the previous one; returns (results, ok, reason)."""
        q = np.zeros(self.model.nu)
        q[self.waist_pitch_idx] = stance.waist_pitch_rad
        out: list[KeyframeIK] = []
        for kf in keyframes:
            T_o = T_world_object(kf.phase) if callable(T_world_object) else T_world_object
            T_w = T_o @ np.asarray(kf.T_object_hand)
            p, R = self.world_to_pelvis_frame(stance, T_w)
            res = self.ik.solve(q, p, R)
            near_limit = bool(
                np.any(res.q_arm < self.ik.lo + limit_margin) or np.any(res.q_arm > self.ik.hi - limit_margin)
            )
            ok = res.pos_err_m <= POS_TOL_M and res.rot_err_rad <= ROT_TOL_RAD and not near_limit
            out.append(KeyframeIK(kf.phase, res.q_arm, res.pos_err_m, res.rot_err_rad, ok))
            q[self.ik.arm_idx] = res.q_arm
        if not all(k.feasible for k in out):
            return out, False, "ARM_IK_INFEASIBLE"
        steps = [float(np.abs(b.q_arm - a.q_arm).max()) for a, b in zip(out, out[1:], strict=False)]
        if steps and max(steps) > MAX_KEYFRAME_STEP_RAD:
            return out, False, "ARM_IK_BRANCH_JUMP"
        return out, True, ""

    def approach_path(
        self, stance: Stance, T_world_target: np.ndarray, retreat_m=(-0.2, 0.0, 0.2), n: int = 8
    ) -> tuple[list[np.ndarray], bool]:
        """Arm waypoints [staging, ..., target]: from a raised pose close to the body down to the target.

        Joint-space interpolation between two independent IK solutions can swing the long hand through the
        table edge. Solving the target first and walking Cartesian-ly *away* from it (warm-started) keeps
        every waypoint on the target's IK branch; the caller moves the idle arm pose to the staging
        waypoint first.
        """
        q = np.zeros(self.model.nu)
        q[self.waist_pitch_idx] = stance.waist_pitch_rad
        p1, R1 = self.world_to_pelvis_frame(stance, T_world_target)
        res = self.ik.solve(q, p1, R1)
        ok = res.pos_err_m <= POS_TOL_M and res.rot_err_rad <= ROT_TOL_RAD
        out = [res.q_arm.copy()]
        q[self.ik.arm_idx] = res.q_arm
        p_stage = p1 + self.world_to_pelvis_frame(stance, np.eye(4))[1] @ np.array(retreat_m)
        for i in range(1, n + 1):
            res = self.ik.solve(q, p1 + (p_stage - p1) * i / n, R1)
            q[self.ik.arm_idx] = res.q_arm
            out.append(res.q_arm.copy())
            ok &= res.pos_err_m <= POS_TOL_M and res.rot_err_rad <= ROT_TOL_RAD
        return out[::-1], bool(ok)

    def wrist_world(self, stance: Stance, q_arm: np.ndarray) -> np.ndarray:
        """Forward check: wrist pose in world for the solved arm (used by tests and visibility)."""
        q = np.zeros(self.model.nu)
        q[self.waist_pitch_idx] = stance.waist_pitch_rad
        q[self.ik.arm_idx] = q_arm
        p, R = self.ik.wrist_pose(q)
        c, s = np.cos(stance.yaw), np.sin(stance.yaw)
        R_wp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = R_wp @ R
        T[:3, 3] = R_wp @ (p - np.array([0, 0, FIXED_PELVIS_Z])) + np.array(
            [stance.x, stance.y, stance.height_m]
        )
        return T


_ = mujoco  # re-exported for callers that build models alongside StanceIK
