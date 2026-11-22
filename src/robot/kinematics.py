"""Damped least-squares IK on the fixed-pelvis G1 model (kinematic, no contacts)."""

from dataclasses import dataclass

import mujoco
import numpy as np

from common.transforms import rotation_angle

ARM_JOINTS = {
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
}
WRIST_BODY = {"right": "right_wrist_yaw_link", "left": "left_wrist_yaw_link"}


@dataclass
class IKResult:
    q_arm: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    converged: bool


SHOULDER_ROLL_INWARD_LIMIT_RAD = 0.05


class ArmIK:
    """IK for the wrist-yaw body of one arm; other joints stay at ``q_ref``."""

    def __init__(self, model: mujoco.MjModel, side: str = "right"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY[side])
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[side]]
        self.dof = np.array([model.jnt_dofadr[i] for i in ids])
        act_joints = model.actuator_trnid[:, 0].tolist()
        self.act_qadr = model.jnt_qposadr[act_joints]  # q vectors are in actuator (joint-order) layout
        self.arm_idx = np.array([act_joints.index(i) for i in ids])
        self.lo = model.jnt_range[ids, 0]
        self.hi = model.jnt_range[ids, 1].copy()
        self.lo = self.lo.copy()
        roll = ARM_JOINTS[side].index(
            f"{side}_shoulder_roll_joint"
        )  # beyond this the arm presses into the torso
        if side == "right":
            self.hi[roll] = min(self.hi[roll], SHOULDER_ROLL_INWARD_LIMIT_RAD)
        else:
            self.lo[roll] = max(self.lo[roll], -SHOULDER_ROLL_INWARD_LIMIT_RAD)

    def wrist_pose(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.act_qadr] = q_full
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.xpos[self.body].copy(), self.data.xmat[self.body].reshape(3, 3).copy()

    def solve(
        self,
        q_full: np.ndarray,
        pos_m: np.ndarray,
        R: np.ndarray,
        iters: int = 200,
        damping: float = 1e-2,
        tol_m: float = 1e-3,
        tol_rad: float = 2e-3,
    ) -> IKResult:
        q = q_full.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            p, Rc = self.wrist_pose(q)
            e_p = pos_m - p
            e_r_mat = R @ Rc.T  # world-frame rotation error
            e_r = 0.5 * np.array(
                [
                    e_r_mat[2, 1] - e_r_mat[1, 2],
                    e_r_mat[0, 2] - e_r_mat[2, 0],
                    e_r_mat[1, 0] - e_r_mat[0, 1],
                ]
            )
            converged = np.linalg.norm(e_p) < tol_m and rotation_angle(e_r_mat) < tol_rad
            if converged:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body)
            J = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])
            err = np.concatenate([e_p, e_r])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            dq = J_pinv @ err
            q[self.arm_idx] = np.clip(q[self.arm_idx] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, Rc = self.wrist_pose(q)
        pos_err = float(np.linalg.norm(pos_m - p))
        rot_err = float(rotation_angle(R @ Rc.T))
        return IKResult(
            q[self.arm_idx].copy(), pos_err, rot_err, pos_err < 5 * tol_m and rot_err < 5 * tol_rad
        )
