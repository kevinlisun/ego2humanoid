"""Arm contact detection for the G1 model: the two arms must not hit each other, nor the torso."""

import mujoco
import numpy as np


def geom_sides(model: mujoco.MjModel, exclude_bodies: set[str] = frozenset()) -> dict[int, str]:
    """Map each robot geom id to ``"left"``/``"right"`` by the name of its body (other geoms are omitted)."""
    out = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if name in exclude_bodies:
            continue
        if name.startswith("left"):
            out[g] = "left"
        elif name.startswith("right"):
            out[g] = "right"
    return out


def arms_touching(data: mujoco.MjData, side_of: dict[int, str]) -> bool:
    for i in range(data.ncon):
        a, b = side_of.get(int(data.contact[i].geom1)), side_of.get(int(data.contact[i].geom2))
        if a and b and a != b:
            return True
    return False


def hand_geoms(model: mujoco.MjModel, side: str) -> np.ndarray:
    """Geom ids of everything below ``<side>_wrist_yaw_link`` (works for Dex3 and Sharpa alike)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    if root < 0:
        return np.zeros(0, int)
    in_hand = np.zeros(model.nbody, bool)
    in_hand[root] = True
    for b in range(root + 1, model.nbody):  # bodies are stored parents-first
        in_hand[b] = in_hand[model.body_parentid[b]]
    return np.where(in_hand[model.geom_bodyid])[0]


class ArmCollisionChecker:
    """Counts frames of an actuator-order joint trajectory in which the two arms touch.

    ``hand_clearance_m`` additionally rejects frames in which the two *hands* come closer than the given
    distance without touching yet. The five-finger hands track their command with a few degrees of PD
    error, so a commanded pose that only just clears collides once it is actually executed; the margin is
    deliberately limited to the hands, because inflating the whole arm moves the idle arm far enough to
    sweep objects off the table.
    """

    def __init__(self, model: mujoco.MjModel, hand_clearance_m: float = 0.0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.side_of = geom_sides(model)
        self.hand_clearance_m = float(hand_clearance_m)
        self._left = hand_geoms(model, "left")
        self._right = hand_geoms(model, "right")
        self.body_geoms = torso_geoms(model)

    def _hands_too_close(self) -> bool:
        margin = self.hand_clearance_m
        pos_l, pos_r = self.data.geom_xpos[self._left], self.data.geom_xpos[self._right]
        rb_l, rb_r = self.model.geom_rbound[self._left], self.model.geom_rbound[self._right]
        gap = np.linalg.norm(pos_l[:, None] - pos_r[None], axis=-1) - rb_l[:, None] - rb_r[None]
        for i, j in zip(*np.where(gap < margin), strict=True):  # bounding spheres first, exact after
            g1, g2 = int(self._left[i]), int(self._right[j])
            if mujoco.mj_geomDistance(self.model, self.data, g1, g2, margin, None) < margin:
                return True
        return False

    def colliding_frames(self, q_traj: np.ndarray) -> np.ndarray:
        out = np.zeros(len(q_traj), bool)
        for i, q in enumerate(q_traj):
            self.data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self.data)
            out[i] = arms_touching(self.data, self.side_of)
            if not out[i] and self.hand_clearance_m > 0:
                out[i] = self._hands_too_close()
        return out

    def count(self, q_traj: np.ndarray) -> int:
        return int(self.colliding_frames(q_traj).sum())

    def torso_frames(self, q_traj: np.ndarray, side: str | None = None) -> np.ndarray:
        """(N,) frames in which an arm (``side`` only, if given) touches the torso, pelvis or head.

        The arm IK only knows the wrist target, so on a target close in front of the chest the elbow
        or the wrist can fold into the torso; the PD controller then cannot reach the target at all.
        """
        out = np.zeros(len(q_traj), bool)
        for i, q in enumerate(q_traj):
            self.data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self.data)
            for c in self.data.contact[: self.data.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                arm = g2 if g1 in self.body_geoms else g1 if g2 in self.body_geoms else None
                if arm is not None and self.side_of.get(arm) in ((side,) if side else ("left", "right")):
                    out[i] = True
                    break
        return out


def torso_geoms(model: mujoco.MjModel) -> set[int]:
    """Geom ids of the robot's unsided bodies (pelvis, waist, torso, head), found from the pelvis down."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if root < 0:
        return set()
    out = set()
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if model.body_rootid[b] == model.body_rootid[root] and not name.startswith(("left", "right")):
            out.add(g)
    return out
