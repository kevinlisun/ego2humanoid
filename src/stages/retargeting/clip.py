"""Clip-level kinematic retargeting: landmarks (camera frame) + gravity -> 43-DoF joint targets."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from robot.collision import ArmCollisionChecker
from stages.locomotion.base_placement import OTHER_SIDE, PlaceGoal, Placement, ik_waist, place_hand_motion
from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.reconstruction.align.frames import gravity_aligned_rotation
from stages.reconstruction.align.table_fit import (
    PENETRATION_TOL_M,
    SIDES,
    arm_penetration,
    clearance_lift,
    table_penetration,
)
from stages.retargeting.kinematic import (
    PELVIS_HEIGHT_M,
    RetargetResult,
    landmarks_camera_to_sim,
    retarget_hand,
)
from stages.retargeting.support_clearance import Support, lift_above_support

CONTROL_RATE_HZ = 50.0
AVOID_STEP_M = 0.02
MAX_AVOID_ITERS = 16  # 8 left the clip 104 Dex3 idle wrist 5 cm inside the torso
MAX_AVOID_M = 0.25
MAX_CLEARANCE_ITERS = 3  # IK -> robot-table penetration -> lift the offending hand -> IK
# While gripping, a palm (or a tucked finger) in the table is first cleared by rolling the hand about the
# thumb-index axis, which keeps the grip height; lifting would close the fingers above the object.
ROLL_LEVER_M = 0.06  # penetration -> roll angle: depth over this lever (palm ~6 cm off the pinch axis)
MAX_PINCH_ROLL_RAD = np.radians(60.0)
FINGER_ROLL_TOL_M = 0.01  # the pinching fingertips may touch the table; deeper finger contact rolls
# Clearance demanded between the two hands while planning, per hand model. The five-finger hands are
# bulky and track their command with a few degrees of PD error, so a pose that only just clears in the
# plan collides once executed; Dex3 keeps 0.0 so its trajectories stay bit-identical.
# Measured on clip 104 with the *executed* (post z-offset) trajectory: 0.0 -> 628 arm-contact steps,
# lift 0.199 m, carry 0.235 m; 0.008 -> 634 steps, lift 0.202 m, carry 0.337 m. The margin does not cut
# the executed self-contacts (those come from the hand pressing on the torso, see H.8) but it does keep
# the idle arm out of the carry, so it is kept at 0.008. Beyond ~0.012 the avoidance loop displaces the
# idle wrist far enough to break the grasp.
# Sharpa: clip 100 (left active) had the idle right hand 12 mm from the active one during the carry, which
# looks like a collision in the render; 0.03 keeps >= 31 mm there and the lift/placement are unchanged.
HAND_CLEARANCE_M = {"dex3": 0.0, "sharpa": 0.03}

BASE_POSE = np.eye(4)
BASE_POSE[2, 3] = PELVIS_HEIGHT_M


@dataclass
class ClipRetarget:
    timestamps_s: np.ndarray  # (M,) at CONTROL_RATE_HZ
    q_target_rad: np.ndarray  # (M, DoF)
    placement: Placement
    ik: RetargetResult  # active hand, at the input frame rate
    ik_passive: RetargetResult | None = None


def _avoid_arm_collisions(
    lm_active_sim: np.ndarray,
    lm_passive_sim: np.ndarray,
    side: str,
    ik_active: RetargetResult,
    hand: str = "dex3",
    waist_pitch_rad: float = 0.0,
    supports: Sequence[Support] = (),
    cache: dict | None = None,
) -> tuple[RetargetResult, np.ndarray]:
    """Retarget the idle hand, pushing its wrist away from the active hand while the two robot hands collide.

    Dex3 hands are larger than the human hands they imitate, so two human hands that nearly touch make the
    robot hands interpenetrate. Each solve leaves the other arm's joints at zero, so the two add up.
    A push can move the hand over a table object the landmarks were clear of, so the pushed hand is
    kept above ``supports`` again (clip 104 Sharpa: the torso push put the idle fingers 1.4 cm into
    the welded bowl, and its reaction twisted the waist enough to miss the grasp).
    """
    from scipy.ndimage import maximum_filter1d

    checker = ArmCollisionChecker(
        build_scene(SceneConfig(T_sim_robot_base=BASE_POSE, hand=hand)),
        hand_clearance_m=HAND_CLEARANCE_M.get(hand, 0.0),
    )
    other = OTHER_SIDE[side]
    away = lm_passive_sim[:, 0] - lm_active_sim[:, 0]
    away = away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-6)
    offset = np.zeros_like(away)
    waist = ik_waist(checker.model)

    def _both(ik_p):
        q = ik_active.q_target_rad + ik_p.q_target_rad
        q[:, waist] = waist_pitch_rad  # each arm solve carries the tilt; it must not be added twice
        return q

    ik_p = retarget_hand(lm_passive_sim, other, hand=hand, waist_pitch_rad=waist_pitch_rad, cache=cache)
    # The idle arm is also kept out of the torso: forwards and outwards, away from the chest.
    out_dir = np.array([1.0, 1.0 if other == "left" else -1.0, 0.0]) / np.sqrt(2.0)
    for _ in range(MAX_AVOID_ITERS):
        q_b = _both(ik_p)
        bad = checker.colliding_frames(q_b)
        in_torso = checker.torso_frames(q_b, other)
        if not (bad.any() or in_torso.any()):
            break
        push = maximum_filter1d(bad.astype(float), size=9)[:, None] * away  # neighbours too: smooth path
        push = push + maximum_filter1d(in_torso.astype(float), size=9)[:, None] * out_dir
        offset = np.clip(offset + AVOID_STEP_M * push, -MAX_AVOID_M, MAX_AVOID_M)
        offset = gaussian_filter1d(offset, 2.0, axis=0, mode="nearest")
        shifted = lm_passive_sim + offset[:, None]
        rise = lift_above_support(shifted, supports)[:, 0, 2] - shifted[:, 0, 2]
        ik_p = retarget_hand(
            lm_passive_sim,
            other,
            wrist_offset_m=offset + rise[:, None] * np.array([0.0, 0.0, 1.0]),
            hand=hand,
            waist_pitch_rad=waist_pitch_rad,
            cache=cache,
        )
    return ik_p, _both(ik_p)


def _worse_roll_runs(
    grip: np.ndarray, roll: np.ndarray, prev_roll: np.ndarray, pen: np.ndarray, prev_pen: np.ndarray
) -> np.ndarray:
    """(N,) mask of the grip runs whose roll changed and whose summed hand penetration grew."""
    out = np.zeros(len(grip), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, grip.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        changed = not np.allclose(roll[a:b], prev_roll[a:b])
        if changed and pen[a:b].sum() > prev_pen[a:b].sum() + PENETRATION_TOL_M:
            out[a:b] = True
    return out


def retarget_clip(
    landmarks_camera: np.ndarray,
    timestamps_s: np.ndarray,
    up_in_camera: np.ndarray,
    side: str,
    table_top_z_m: float = 0.80,
    pinch_gap_m: np.ndarray | None = None,
    placement: Placement | None = None,
    offset_m: np.ndarray | None = None,  # (N, 3) per-frame displacement added to the active hand motion
    passive_landmarks_camera: np.ndarray | None = None,  # (N, 21, 3) other hand, same frames
    hand: str = "dex3",
    pinch_uv_ref: np.ndarray | None = None,  # (N, 2) human pinch in the source image, normalised
    support: Support | None = None,  # table: both robot arms are kept clear of it (before ``offset_m``)
    clearance_exempt: np.ndarray | None = None,  # (N,) bool, active hand at the object (approach, grasp)
    obstacles: Sequence[Support] = (),  # other objects on the table: the hands are kept above them too
    passive_obstacles: Sequence[Support] | None = None,  # the same for the other hand (default: obstacles)
    place_goal: PlaceGoal | None = None,  # steers the standing spot towards a hard-to-reach receptacle
) -> ClipRetarget:
    R = gravity_aligned_rotation(up_in_camera)
    placement = placement or place_hand_motion(
        landmarks_camera,
        R,
        side,
        table_top_z_m,
        passive_landmarks_camera=passive_landmarks_camera,
        pinch_uv_ref=pinch_uv_ref,
        hand=hand,
        place_goal=place_goal,
    )
    pitch = placement.waist_pitch_rad
    table = () if support is None else (support,)
    supports = (*table, *obstacles)
    lm_sim = lift_above_support(
        landmarks_camera_to_sim(landmarks_camera, placement.T_sim_camera), supports, clearance_exempt
    )
    if offset_m is not None:
        lm_sim = lm_sim + np.asarray(offset_m, float)[:, None, :]
    lm_p = None
    if passive_landmarks_camera is not None:
        p_supports = supports if passive_obstacles is None else (*table, *passive_obstacles)
        lm_p = lift_above_support(
            landmarks_camera_to_sim(passive_landmarks_camera, placement.T_sim_camera), p_supports
        )
    # The landmarks clear the table, but the robot wrist and palm are bulkier than a human hand: the
    # executed geometry is checked against the table and the offending hand lifted until it clears.
    si, oi = SIDES.index(side), 1 - SIDES.index(side)
    lift = np.zeros((len(lm_sim), 2))
    roll = np.zeros(len(lm_sim))
    grip = np.zeros(len(lm_sim), bool)
    if clearance_exempt is not None and pinch_gap_m is not None:
        grip = np.asarray(clearance_exempt, bool) & np.isfinite(np.asarray(pinch_gap_m, float))
    frozen = np.zeros(len(lm_sim), bool)  # grip runs where rolling made the penetration worse
    prev_roll, prev_pen = None, None
    # The clearance rounds re-solve nearly the same chains: only the frames after the first change are redone.
    active_cache, passive_cache = {}, {}
    for it in range(MAX_CLEARANCE_ITERS + 2):
        up_a = np.array([0.0, 0.0, 1.0]) * lift[:, si, None]
        ik = retarget_hand(
            lm_sim + up_a[:, None],
            side,
            pinch_gap_m=pinch_gap_m,
            hand=hand,
            waist_pitch_rad=pitch,
            pinch_roll_rad=roll if roll.any() else None,
            cache=active_cache,
        )
        q_target = ik.q_target_rad
        ik_passive = None
        if lm_p is not None:
            up_p = np.array([0.0, 0.0, 1.0]) * lift[:, oi, None]
            ik_passive, q_target = _avoid_arm_collisions(
                lm_sim + up_a[:, None], lm_p + up_p[:, None], side, ik, hand, pitch, p_supports, passive_cache
            )
        if support is None or it == MAX_CLEARANCE_ITERS + 1:
            break
        raw = table_penetration(q_target, support, hand, PELVIS_HEIGHT_M)
        pen = arm_penetration(raw, side, clearance_exempt)
        hand_pen = np.maximum(raw[:, si, 0], np.where(raw[:, si, 1] > FINGER_ROLL_TOL_M, raw[:, si, 1], 0.0))
        # The roll swings the palm up but also turns the thumb or little-finger chain down; on a thin
        # slab (Sharpa clip 108, 1.6 cm) a 1.2 cm fingertip contact rolled the thumb base 3 cm into the
        # table, which held the fingers off the object. A grip run keeps its roll only if it helped.
        if prev_roll is not None:
            worse = _worse_roll_runs(grip, roll, prev_roll, hand_pen, prev_pen)
            if worse.any():
                roll[worse], frozen[worse] = prev_roll[worse], True
                hand_pen[worse] = prev_pen[worse]
                prev_roll, prev_pen = roll.copy(), hand_pen
                continue  # solve again with the reverted roll
        if it >= MAX_CLEARANCE_ITERS:
            break
        prev_roll, prev_pen = roll.copy(), hand_pen
        can_roll = grip & ~frozen & (roll < MAX_PINCH_ROLL_RAD - 1e-6) & (hand_pen > PENETRATION_TOL_M)
        if can_roll.any():
            add = clearance_lift(np.where(can_roll, hand_pen, 0.0)) / ROLL_LEVER_M
            roll = np.where(grip, np.minimum(roll + add, MAX_PINCH_ROLL_RAD), 0.0)
            pen[can_roll, si] = 0.0  # rolled this round instead of lifted
        elif not (pen > PENETRATION_TOL_M).any():
            break
        for s_ in (si, oi):
            lift[:, s_] += clearance_lift(pen[:, s_])
    t_out = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / CONTROL_RATE_HZ)
    q_out = np.stack(
        [np.interp(t_out, timestamps_s, q_target[:, j]) for j in range(q_target.shape[1])], axis=1
    )
    return ClipRetarget(t_out, q_out, placement, ik, ik_passive)
