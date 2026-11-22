"""L.11: terminal validators for a stance + arm posture (no forbidden penetration, no pre-contact)."""

from __future__ import annotations

import mujoco
import numpy as np

from robot.kinematics import ARM_JOINTS, WRIST_BODY
from stages.locomotion.plan.candidates import Stance
from stages.locomotion.plan.visibility import set_stance

PENETRATION_TOL_M = 0.005
LEG_PARTS = ("hip", "knee", "ankle", "pelvis")  # legs are at model default here; checked at arrival


def _is_leg(name: str) -> bool:
    return any(p in name for p in LEG_PARTS)


def tucked_arm(side: str) -> np.ndarray:
    """Posture of the arm that does not manipulate: held close to the hip, away from the table."""
    return np.array([0.5, 0.3 if side == "left" else -0.3, 0.0, 0.2, 0.0, 0.0, 0.0])


def set_arm(model, data, side: str, q_arm: np.ndarray) -> None:
    other = "left" if side == "right" else "right"
    for sd, q_side in ((side, q_arm), (other, tucked_arm(other))):
        for n, q in zip(ARM_JOINTS[sd], q_side, strict=True):
            data.qpos[model.joint(n).qposadr[0]] = q
    mujoco.mj_kinematics(model, data)


def _in_hand(model, body: int, side: str) -> bool:
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY[side])
    while body > 0:
        if body == wrist:
            return True
        body = int(model.body_parentid[body])
    return False


def forbidden_contacts(
    model, data, stance: Stance, side: str, q_arm, object_name: str, allow_object: bool, hand_table_ok=False
):
    """Names of geom pairs in forbidden penetration (arm/torso vs table, obstacles, itself, object).

    ``hand_table_ok`` lets the manipulating hand touch the table (fingers on a tabletop for flat objects).
    """
    set_stance(model, data, stance)
    set_arm(model, data, side, q_arm)
    mujoco.mj_forward(model, data)
    root = int(model.jnt_bodyid[0])
    bad = []
    for c in data.contact[: data.ncon]:
        if c.dist > -PENETRATION_TOL_M:
            continue
        b1, b2 = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
        r1, r2 = (int(model.body_rootid[b]) == root for b in (b1, b2))
        n1, n2 = (
            model.geom(g).name or f"body:{model.body(model.geom_bodyid[g]).name}" for g in (c.geom1, c.geom2)
        )
        bn1, bn2 = (model.body(b).name for b in (b1, b2))
        if r1 and r2:
            if not (_is_leg(bn1) or _is_leg(bn2)):
                bad.append((n1, n2))
            continue
        other, other_body, robot_body = (n2, bn2, bn1) if r1 else (n1, bn1, bn2)
        if not (r1 or r2) or other == "floor" or _is_leg(robot_body):
            continue
        if other_body == object_name and allow_object:
            continue
        if hand_table_ok and other == "table_top" and _in_hand(model, b1 if r1 else b2, side):
            continue
        bad.append((n1, n2))
    return bad
