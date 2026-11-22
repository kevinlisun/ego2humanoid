"""Grip sizing for a position-controlled thumb-index pinch.

The fingers are PD position servos, so the grip force is not commanded directly: it is the finger
stiffness times how far the commanded tip gap is closed *past* the object surface (the squeeze). A
constant squeeze gives very different forces on the two hands -- measured at the hold pose, the
Sharpa thumb-index pair is ~340 N/m in series against ~2350 N/m for the Dex3, so the same 12 mm held
a 30 g object with 4 N on one and 28 N on the other, and the Sharpa let it slide out of a wedge-shaped
pinch during the carry (clips 100, 117). The squeeze is therefore sized per clip from the force the
object needs and the stiffness the hand has at that pose.

Force needed. Two opposing contacts each carry at most ``mu`` times the normal force tangentially.
When the contact normals lean by ``alpha`` from the pinch axis (a non-parallel, wedge-shaped pair of
faces), part of each normal force pushes the object out along the wedge, so only ``mu - tan(alpha)``
of it holds. Carrying mass ``m`` at up to ``a`` m/s^2 on top of gravity with a safety factor ``SF``::

    N_req = SF * m * (g + a) / (2 * (mu - tan(alpha)))

Stiffness. With joint stiffness ``Kp`` (the position-actuator gains) and the two fingertip
Jacobians, the relative tip compliance along the unit pinch axis ``n`` is
``n^T (J_t - J_i) Kp^-1 (J_t - J_i)^T n``; the arm and wrist joints are shared by both tips and
cancel in ``J_t - J_i``, so only the finger joints count.
"""

from __future__ import annotations

import mujoco
import numpy as np

GRAVITY_M_S2 = 9.81
CARRY_ACCEL_M_S2 = 3.0  # wrist acceleration the grip must ride out (matches the carry limiter)
SAFETY_FACTOR = 2.0
MIN_GRIP_N = 4.0  # below this a 1-2 mm tracking error already loses the object
MAX_TAN_ALPHA = 0.7  # a steeper wedge is not held by friction at all; size for this and rely on C
CONTACT_ALLOWANCE_M = 0.002  # soft-contact penetration taken before the force builds up
MAX_SQUEEZE_M = 0.02  # beyond this the fingers cross inside the object and roll it out instead


def required_grip_force_n(
    mass_kg: float,
    mu: float,
    tan_alpha: float = 0.0,
    accel_m_s2: float = CARRY_ACCEL_M_S2,
    safety: float = SAFETY_FACTOR,
    min_force_n: float = MIN_GRIP_N,
) -> float:
    """Normal force per contact that holds ``mass_kg`` between two fingertips (see module doc)."""
    hold = max(mu - min(max(tan_alpha, 0.0), MAX_TAN_ALPHA), 0.1)
    return max(safety * mass_kg * (GRAVITY_M_S2 + accel_m_s2) / (2.0 * hold), min_force_n)


def squeeze_for_force(
    force_n: float,
    stiffness_n_m: float,
    min_squeeze_m: float,
    max_squeeze_m: float = MAX_SQUEEZE_M,
    allowance_m: float = CONTACT_ALLOWANCE_M,
) -> float:
    """Commanded gap reduction below the object width that produces ``force_n`` at ``stiffness_n_m``."""
    return float(np.clip(force_n / max(stiffness_n_m, 1.0) + allowance_m, min_squeeze_m, max_squeeze_m))


def actuated_stiffness(model: mujoco.MjModel) -> np.ndarray:
    """(nv,) position-servo stiffness per DoF (Nm/rad); 0 where no joint actuator acts."""
    kp = np.zeros(model.nv)
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        dof = model.jnt_dofadr[model.actuator_trnid[a, 0]]
        kp[dof] += model.actuator_gainprm[a, 0]
    return kp


def pinch_stiffness_n_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tip_bodies: tuple[int, int],
    tip_points: tuple[np.ndarray, np.ndarray],
) -> float:
    """Series stiffness (N/m) of two fingertips squeezed towards each other at the current pose.

    ``data`` must already hold the pose (``mj_kinematics`` done); ``tip_points`` are world points on
    ``tip_bodies``. Un-actuated DoFs are treated as rigid.
    """
    jac = []
    for body, point in zip(tip_bodies, tip_points, strict=True):
        jp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, None, np.asarray(point, float), int(body))
        jac.append(jp)
    n = np.asarray(tip_points[1], float) - np.asarray(tip_points[0], float)
    n /= max(np.linalg.norm(n), 1e-9)
    row = n @ (jac[0] - jac[1])
    kp = actuated_stiffness(model)
    act = kp > 0
    compliance = float(np.sum(row[act] ** 2 / kp[act]))
    return 1.0 / max(compliance, 1e-9)


def hold_pinch_stiffness(q_full: np.ndarray, side: str, hand: str, gap_m: float) -> float:
    """Pinch stiffness (N/m) of ``hand`` with its fingers solved to ``gap_m`` at arm pose ``q_full``."""
    from robot.kinematics import ArmIK
    from robot.layout import RobotLayout
    from robot.pinch import PinchSolver, SharpaPinchSolver
    from stages.manipulation.mjcf_scene import SceneConfig, build_scene
    from stages.retargeting.kinematic import SIDE_SLICE

    model = build_scene(SceneConfig(hand=hand))
    ik = ArmIK(model, side)
    solver = (
        SharpaPinchSolver(ik, side, RobotLayout.from_model(model).finger_idx(side))
        if hand == "sharpa"
        else PinchSolver(ik, side, SIDE_SLICE[side])
    )
    q = solver.solve(np.asarray(q_full, float), gap_m)
    tips = solver.tips(q)
    return pinch_stiffness_n_m(ik.model, ik.data, tuple(solver.bodies[:2]), tips)
