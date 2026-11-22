"""Thumb-index pinch geometry: fingertip points and a small solver that sets the pinch gap.

``PinchSolver`` works on the Dex3 hand (tip = farthest mesh vertex of the distal link);
``SharpaPinchSolver`` works on the Sharpa Wave hand, whose MJCF already carries fingertip sites.
"""

import mujoco
import numpy as np
from scipy.optimize import least_squares

from robot.finger_map import load_finger_config, sharpa_joint_names
from robot.kinematics import ArmIK
from robot.sharpa import fingertip_site

PINCH_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1")
LEAN_WEIGHT = 0.5  # residual weight per metre of distal-link lean, relative to 1 per metre of gap error
SHARPA_LEAN_WEIGHT = 1.0  # same, for the inward lean only of the Sharpa distal links


def _tip_local(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Farthest mesh vertex of the body's geoms from the body origin (body frame)."""
    best, best_d = np.zeros(3), -1.0
    for g in np.where(model.geom_bodyid == body)[0]:
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        v = v @ R.reshape(3, 3).T + model.geom_pos[g]
        d = np.linalg.norm(v, axis=1)
        if d.max() > best_d:
            best_d, best = d.max(), v[d.argmax()]
    return best


class PinchSolver:
    def __init__(self, ik: ArmIK, side: str, side_slice: slice):
        self.ik, self.side, self.slice = ik, side, side_slice
        m = ik.model
        self.bodies = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_{n}_link")
            for n in ("thumb_2", "index_1")
        ]
        self.tips_local = [_tip_local(m, b) for b in self.bodies]
        self.middle_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_middle_1_link")
        self.middle_local = _tip_local(m, self.middle_body)
        order = load_finger_config()["joint_order"][
            side
        ]  # canonical names, one per actuator in the side slice
        self.idx = np.array([order.index(p) for p in PINCH_JOINTS])
        self.index_gi = side_slice.start + np.array([order.index(p) for p in ("index_0", "index_1")])
        self.middle_gi = side_slice.start + np.array([order.index(p) for p in ("middle_0", "middle_1")])
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{n}_joint") for n in order]
        self.lo = m.jnt_range[jids, 0][self.idx]
        self.hi = m.jnt_range[jids, 1][self.idx]

    def tips(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.ik.wrist_pose(q_full)
        d = self.ik.data
        return tuple(
            d.xpos[b] + d.xmat[b].reshape(3, 3) @ t for b, t in zip(self.bodies, self.tips_local, strict=True)
        )

    def middle_tip(self, q_full: np.ndarray) -> np.ndarray:
        self.ik.wrist_pose(q_full)
        d = self.ik.data
        return d.xpos[self.middle_body] + d.xmat[self.middle_body].reshape(3, 3) @ self.middle_local

    def mirror_middle(self, q_full: np.ndarray) -> np.ndarray:
        """The middle finger (beside the index, same joint axes and limits) closes like the index."""
        q = q_full.copy()
        q[self.middle_gi] = q[self.index_gi]
        return q

    def lean(self, q_full: np.ndarray) -> np.ndarray:
        """Inward lean (m, along the pinch axis) of the thumb and index distal links, tip to joint."""
        a, b = self.tips(q_full)
        u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        d = self.ik.data
        return np.array([(d.xpos[self.bodies[0]] - a) @ u, (b - d.xpos[self.bodies[1]]) @ u])

    def solve(self, q_full: np.ndarray, gap_m: float, reg: float = 0.02) -> np.ndarray:
        """Return ``q_full`` with the five pinch joints set so the tip gap is ``gap_m``, the distal links
        standing square to the pinch axis, closest to the current pose otherwise.

        The tips are the farthest vertices of the distal links. Left free, the solver bent the links
        into a V with the joint ends 2-5 cm inside the tips: on a 4 cm block the link faces met the
        top edge instead of the sides and the squeeze wedged the block down and out of the hand
        (clips 100, 104, 114)."""
        q = q_full.copy()
        gi = self.slice.start + self.idx
        x0 = np.clip(q[gi], self.lo + 1e-3, self.hi - 1e-3)

        def res(x):
            q[gi] = x
            a, b = self.tips(q)
            gap_res = [np.linalg.norm(a - b) - gap_m]
            return np.concatenate([gap_res, LEAN_WEIGHT * self.lean(q), reg * (x - x0)])

        sol = least_squares(res, x0, bounds=(self.lo, self.hi), max_nfev=120, diff_step=1e-3)
        q[gi] = sol.x
        return q


class SharpaPinchSolver:
    """Same contract as ``PinchSolver`` for the Sharpa Wave hand, using its fingertip sites."""

    def __init__(self, ik: ArmIK, side: str, finger_idx: np.ndarray):
        self.ik, self.side = ik, side
        m = ik.model
        self.sites = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, fingertip_site(side, f))
            for f in ("thumb", "index")
        ]
        self.bodies = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{f}_DP") for f in ("thumb", "index")
        ]
        order = sharpa_joint_names(side)
        names = [f"{side}_{j}" for j in load_finger_config()["sharpa"]["pinch_joints"]]
        self.gi = finger_idx[[order.index(n) for n in names]]
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.lo, self.hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]

    def tips(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.ik.wrist_pose(q_full)
        return tuple(self.ik.data.site_xpos[s].copy() for s in self.sites)

    def lean(self, q_full: np.ndarray) -> np.ndarray:
        """Inward lean (m, along the pinch axis) of the thumb and index distal links, tip to joint."""
        a, b = self.tips(q_full)
        u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        d = self.ik.data
        return np.array([(d.xpos[self.bodies[0]] - a) @ u, (b - d.xpos[self.bodies[1]]) @ u])

    def solve(self, q_full: np.ndarray, gap_m: float, reg: float = 0.02) -> np.ndarray:
        """Tip gap ``gap_m`` with neither distal link leaning in over the object. Left free, the index
        distal joint sat 1 cm inside its tip, level with the top of a 4 cm object: the link met the top
        edge and pressed the tongs down out of the grip (clip 100). Leaning out is harmless."""
        q = q_full.copy()
        x0 = np.clip(q[self.gi], self.lo + 1e-3, self.hi - 1e-3)

        def res(x):
            q[self.gi] = x
            a, b = self.tips(q)
            lean = SHARPA_LEAN_WEIGHT * np.maximum(self.lean(q), 0.0)
            return np.concatenate([[np.linalg.norm(a - b) - gap_m], lean, reg * (x - x0)])

        sol = least_squares(res, x0, bounds=(self.lo, self.hi), max_nfev=120, diff_step=1e-3)
        q[self.gi] = sol.x
        return q
