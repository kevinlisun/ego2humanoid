"""Fit the synthetic table to the robot so the scene is graspable and the arms start clear of it.

The simulated table does not have to match the video: it only has to carry the object where the robot
can grip it. Two failure modes come from copying the human geometry instead:

* The table was raised to the human grip height. The monocular grip is often 7-16 cm above the
  resting hands, which put the top at 0.96 m on clip 1, above the G1 elbow (0.90 m at rest): the
  shoulder and wrist links sat 38-77 mm inside the table on every frame and the object was never lifted.
* The hand landmarks are kept above the table, but the robot wrist and palm are bulkier than a human
  hand, so the resting wrist still started 11-15 mm inside the table (clips 0 and 10).

``fit_table_height`` picks the table top (and the matching global vertical shift of the whole motion)
by reachability and robot-table clearance. ``table_penetration`` measures the executed geometry and
drives the clearance loop in ``retarget_clip``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.retargeting.support_clearance import Support, lift_above_support

TABLE_Z_GRID_M = tuple(np.round(np.arange(0.70, 0.861, 0.02), 3))  # top stays below the G1 elbow
PENETRATION_TOL_M = 0.002  # contacts shallower than this are resting contact, not embedding
CLEARANCE_MARGIN_M = 0.01  # extra lift on top of the measured penetration
PENETRATION_WEIGHT = 3.0  # metres of IK error one metre of mean penetration is worth
PASSIVE_WEIGHT = 0.3
SHIFT_WEIGHT = 0.05  # mild preference for the height the video suggests
# The reconstructed object is a thin slab gripped at mid-height, so a top-down pinch pushes the other
# fingers into the table (clip 1 sharpa: 15-40 mm through the whole hold, no lift). The simulated
# object may reach further down to the table than the video shows; its top and the grip stay put.
# A taller object is a fallback only: on clips that already clear the table it cost grasps (clip 10
# extended 4 cm: success -> NO_LIFT on both hands), so it is tried only when the reconstructed height
# leaves the robot this deep in the table on average (clip 1: 35 mm; clips 0/10/104: <= 9 mm).
OBJECT_EXTEND_GRID_M = (0.02, 0.04, 0.06)
EXTEND_MIN_PENETRATION_M = 0.02
EXTEND_WEIGHT = 0.1
SIDES = ("left", "right")
BODY_CLEARANCE_M = 0.02  # arm links, torso and legs keep this far from the table edge
FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")  # Dex3 and Sharpa finger body names


@dataclass(frozen=True)
class TableFit:
    top_z_m: float
    shift_z_m: float  # added to every landmark (and object) height
    ik_err_m: float
    penetration_m: float  # mean robot-table penetration over the clip at that height
    object_half_z_m: float  # half height of the (possibly extended) object resting on the table


@lru_cache(maxsize=8)
def _table_model(hand: str, top_z_m: float, lo_xy: tuple, hi_xy: tuple, base_z_m: float):
    base = np.eye(4)
    base[2, 3] = base_z_m
    c, h = (np.asarray(lo_xy) + np.asarray(hi_xy)) / 2, (np.asarray(hi_xy) - np.asarray(lo_xy)) / 2
    model = build_scene(
        SceneConfig(
            T_sim_robot_base=base,
            hand=hand,
            table_top_z_m=top_z_m,
            table_center_xy_m=tuple(c),
            table_half_size_xy_m=tuple(h),
        )
    )
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    side_of = np.full(model.ngeom, -1)
    finger = np.zeros(model.ngeom, bool)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        finger[g] = any(
            k in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for k in FINGER_KEYS
        )
        while b > 0:  # the first ancestor named left_*/right_* decides the arm
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith(SIDES):
                side_of[g] = SIDES.index(name.split("_", 1)[0])
                break
            b = model.body_parentid[b]
    qadr = model.jnt_qposadr[model.actuator_trnid[: model.nu, 0]]
    return model, table, side_of, finger, qadr


def table_penetration(q_rad: np.ndarray, support: Support, hand: str, base_z_m: float) -> np.ndarray:
    """(N, DoF) joint targets -> (N, 2, 2) deepest robot-table penetration in metres, indexed by arm
    (left, right) and part (0: arm, wrist and palm; 1: fingers)."""
    model, table, side_of, finger, qadr = _table_model(
        hand, round(support.top_z_m, 4), tuple(support.lo_xy_m), tuple(support.hi_xy_m), base_z_m
    )
    data = mujoco.MjData(model)
    out = np.zeros((len(q_rad), 2, 2))
    for k, q in enumerate(q_rad):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        for c in data.contact[: data.ncon]:
            if table not in (c.geom1, c.geom2) or c.dist >= 0:
                continue
            g = c.geom2 if c.geom1 == table else c.geom1
            if side_of[g] >= 0:
                out[k, side_of[g], int(finger[g])] = max(out[k, side_of[g], int(finger[g])], -c.dist)
    return out


def arm_penetration(pen: np.ndarray, active_side: str, exempt: np.ndarray | None) -> np.ndarray:
    """(N, 2, 2) -> (N, 2) per-arm penetration that must be cleared: on ``exempt`` frames the active
    hand's fingers may rest on the table while it grips, its wrist and palm may not."""
    out = pen.max(axis=2)
    if exempt is not None:
        si = SIDES.index(active_side)
        ex = np.asarray(exempt, bool)
        out[ex, si] = pen[ex, si, 0]
    return out


def clearance_lift(penetration_m: np.ndarray) -> np.ndarray:
    """(N,) penetration -> (N,) smooth upward shift that clears it with a margin."""
    need = np.where(penetration_m > PENETRATION_TOL_M, penetration_m + CLEARANCE_MARGIN_M, 0.0)
    if not need.any():
        return need
    wide = maximum_filter1d(need, size=9, mode="nearest")
    return np.maximum(gaussian_filter1d(wide, 3.0, mode="nearest"), need)


def fit_table_height(
    lm_active_sim: np.ndarray,
    lm_passive_sim: np.ndarray | None,
    side: str,
    grip_z_m: float,
    object_half_z_m: float,
    support_xy: tuple[tuple[float, float], tuple[float, float]],
    hand: str,
    waist_pitch_rad: float,
    base_z_m: float,
    exempt: np.ndarray | None = None,
    hold: np.ndarray | None = None,
    pinch_gap_m: np.ndarray | None = None,
    stride: int = 4,
    z_grid_m: tuple[float, ...] = TABLE_Z_GRID_M,
    extend_grid_m: tuple[float, ...] = OBJECT_EXTEND_GRID_M,
) -> TableFit:
    """Choose the table top (and how far the object reaches down to it) so the grip is reachable and
    the robot clears the table.

    The grip sits ``object_half_z_m`` below the object's top. With the object extended downwards by
    ``e`` the table is at ``grip - object_half_z_m - e``, so a table at ``z`` shifts the whole human
    motion by ``z + object_half_z_m + e - grip_z_m``. Extending the object leaves room under the grip
    for the fingers that do not take part in the pinch; it is only tried when the unextended object
    leaves the robot more than ``EXTEND_MIN_PENETRATION_M`` deep in the table. ``exempt`` (N,) marks
    the frames where the active hand is at the object (approach and hold): only its wrist and palm
    must clear the table there. ``hold`` (N,) marks the gripping frames, where its fingers must clear
    the table as well; ``pinch_gap_m`` (N,) gives them the posture they will actually be executed with.
    """
    from stages.locomotion.base_placement import OTHER_SIDE, ik_waist
    from stages.retargeting.kinematic import retarget_hand

    lo, hi = support_xy
    a = lm_active_sim[::stride]
    p = None if lm_passive_sim is None else lm_passive_sim[::stride]
    ex = np.zeros(len(a), bool) if exempt is None else np.asarray(exempt, bool)[::stride]
    ho = np.zeros(len(a), bool) if hold is None else np.asarray(hold, bool)[::stride]
    gaps = None if pinch_gap_m is None else np.asarray(pinch_gap_m, float)[::stride]
    si = SIDES.index(side)
    waist = ik_waist(_table_model(hand, 0.0, (0.0, 0.0), (0.1, 0.1), base_z_m)[0])
    best = None
    for grid in ((0.0,), extend_grid_m):
        if best is not None and best[1].penetration_m < EXTEND_MIN_PENETRATION_M:
            break
        for z in z_grid_m:
            sup = Support(float(z), tuple(lo), tuple(hi))
            for e in grid:
                dz = float(z + object_half_z_m + e - grip_z_m)
                ik_a = retarget_hand(
                    lift_above_support(a + [0, 0, dz], sup, ex),
                    side,
                    pinch_gap_m=gaps,
                    hand=hand,
                    waist_pitch_rad=waist_pitch_rad,
                )
                q, err = ik_a.q_target_rad.copy(), float(ik_a.pos_err_m.mean())
                if p is not None:
                    ik_p = retarget_hand(
                        lift_above_support(p + [0, 0, dz], sup),
                        OTHER_SIDE[side],
                        hand=hand,
                        waist_pitch_rad=waist_pitch_rad,
                    )
                    q = q + ik_p.q_target_rad
                    q[:, waist] = (
                        waist_pitch_rad  # each arm solve carries the tilt; it must not be added twice
                    )
                    err += PASSIVE_WEIGHT * float(ik_p.pos_err_m.mean())
                raw = table_penetration(q, sup, hand, base_z_m)
                pen = arm_penetration(raw, side, ex)
                pen[ho, si] = raw[ho, si].max(axis=1)
                pen_m = float(pen.max(axis=1).mean())
                cost = err + PENETRATION_WEIGHT * pen_m + SHIFT_WEIGHT * abs(dz) + EXTEND_WEIGHT * e
                if best is None or cost < best[0]:
                    best = (cost, TableFit(float(z), dz, err, pen_m, float(object_half_z_m + e / 2)))
    return best[1]


def near_edge_clearance(
    q_rad: np.ndarray,
    support: Support,
    hand: str,
    base_z_m: float,
    max_lo_x_m: float,
    margin_m: float = BODY_CLEARANCE_M,
    stride: int = 3,
    max_iters: int = 12,
) -> tuple[float, float]:
    """(N, DoF) joint targets -> (near-edge x, remaining deficit) for the robot body.

    ``table_penetration`` only sees contact, and the clearance loop can only lift a hand; an elbow or
    upper arm resting on the near edge (clip 1: upper arm 3 mm into it, elbow 3 mm above it at frame 0)
    is neither. Every non-finger geom whose nearest table point is on the near edge or face pushes
    that edge forward until it is clear. The edge never passes ``max_lo_x_m`` (object and grasp path).
    """
    lo, hi = np.array(support.lo_xy_m, float), np.array(support.hi_xy_m, float)
    for _ in range(max_iters):
        model, table, _, finger, qadr = _table_model(
            hand, round(support.top_z_m, 4), tuple(np.round(lo, 4)), tuple(np.round(hi, 4)), base_z_m
        )
        data = mujoco.MjData(model)
        body = [g for g in range(model.ngeom) if g != table and model.geom_bodyid[g] > 0 and not finger[g]]
        need, fromto = 0.0, np.zeros(6)
        for q in q_rad[::stride]:
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            for g in body:
                dist = mujoco.mj_geomDistance(model, data, g, table, margin_m, fromto)
                # Only a robot point in front of the near edge is helped by moving it; a hand hovering
                # over the top is the clearance loop's job (and is close on purpose while gripping).
                if dist < margin_m and fromto[0] < lo[0]:
                    need = max(need, margin_m - dist)
        if need <= 1e-4 or lo[0] >= max_lo_x_m:
            break
        # Past a rounded edge the gain in distance is less than the step, so overshoot a little.
        lo[0] = min(lo[0] + 1.5 * need + 2e-3, max_lo_x_m)
    return float(lo[0]), float(need)
