"""Kinematic hand -> G1+Dex3 retargeting from 21 hand landmarks (OpenPose/MANO order).

Wrist pose is solved with damped least squares on the arm; finger curls from landmark bend angles are
mapped onto the 7 Dex3 joints. Contact-aware refinement (SPIDER) is a separate, later step.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from common.transforms import apply_se3
from robot.collision import ArmCollisionChecker
from robot.finger_map import map_curls_to_dex3, map_curls_to_sharpa
from robot.kinematics import ArmIK, IKResult
from robot.layout import RobotLayout
from robot.pinch import PinchSolver, SharpaPinchSolver
from stages.manipulation.mjcf_scene import SceneConfig, build_scene

FINGER_CHAINS = {  # landmark indices: wrist, then base to tip
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
RESTART_POS_ERR_M = 0.01  # a warm-started IK missing by more is retried from the neutral arm
HELD_MAX_ARM_STEP_RAD = 0.5  # largest per-frame arm joint change a restart may make while holding
HELD_RESTART_POS_ERR_M = 0.05  # ...unless the warm-started IK misses by more than this
REPAIR_POS_ERR_M = 0.005  # a frame re-solved on the other branch may miss its target by this much more
BRANCH_MERGE_RAD = 0.1  # the re-solved branch has rejoined the original one when this close
BRANCH_BLEND_FRAMES = 15  # free-motion frames over which a branch change moved out of a hold is blended
PINCH_TUCK_CURL = 0.9  # middle/ring/pinky curl while a Sharpa hand pinches
LEVEL_RAMP_FRAMES = 10  # pinch frames over which the thumb-index axis is turned level
ROLL_AXIS_MIN_GAP_M = 0.01  # a closer tip pair gives no usable direction; the last wider one is kept
TURN_FALLBACK_FRACS = (1.0, 0.5, 0.25, 0.0)  # share of the pinch level/roll turn tried when unreachable
TURN_FRAC_RATE = 0.05  # largest change of that share per frame
FULL_CURL_RAD = {"thumb": 1.6, "index": 3.3, "middle": 3.3, "ring": 3.3, "pinky": 3.3}
PELVIS_HEIGHT_M = 0.793  # standing G1 pelvis; table top at 0.80 m
SIDE_SLICE = {"right": slice(36, 43), "left": slice(22, 29)}


# Whether the right hand's palm normal has to be flipped to match the robot's right wrist frame.
# ``cross(index - wrist, pinky - wrist)`` is chirality-dependent: it points out of the *palm* of a right
# hand but out of the *back* of a left hand. Whether that has to be undone depends on the robot hand:
# running the same formula on the compiled model at qpos0 and expressing the result in the wrist frame
# gives y_local = +y for both sides of both hands, so neither needs a flip. (The Sharpa right hand
# used to be mounted palm-outward, which made y_local = -y and forced a flip here; that cost the right
# arm most of its workspace and is now fixed at the mount, see robot/sharpa.py::ATTACH_R.)
# `tests/unit/test_retarget_align.py::test_right_y_flip_matches_the_compiled_hand` re-derives this.
RIGHT_Y_FLIP = {"dex3": False, "sharpa": False}


def hand_frame(lm: np.ndarray, side: str, hand: str = "dex3") -> np.ndarray:
    """Robot wrist-frame axes of a human hand: x along fingers, y out of the palm or the back. (...,3,3).

    The y direction follows ``RIGHT_Y_FLIP[hand]``: a wrong sign commands that hand 180 deg rolled about
    the finger axis (palm up instead of palm down).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if hand not in RIGHT_Y_FLIP:
        raise ValueError(f"unknown hand {hand!r}")
    w = lm[..., 0, :]
    x = lm[..., 9, :] - w
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(lm[..., 5, :] - w, lm[..., 17, :] - w)
    y = y - (y * x).sum(-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    if side == "right" and RIGHT_Y_FLIP[hand]:
        y = -y
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def turn_rotation(u: np.ndarray, v: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """(3, 3) rotation turning direction ``u`` by ``weight`` of the angle towards direction ``v``."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    k = np.cross(u, v)
    s = np.linalg.norm(k)
    if s < 1e-9:
        return np.eye(3)
    angle = weight * np.arctan2(s, float(u @ v))
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def level_rotation(axis: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """(3, 3) rotation turning ``axis`` by ``weight`` of its elevation into the horizontal plane,
    keeping its heading."""
    if np.linalg.norm(axis[:2]) < 1e-9:
        return np.eye(3)
    return turn_rotation(axis, np.array([axis[0], axis[1], 0.0]), weight)


def axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """(3, 3) rotation by ``angle_rad`` about the unit ``axis`` (Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def finger_curls(lm: np.ndarray) -> np.ndarray:
    """(N, 21, 3) -> (N, 5) curl in [0, 1] (thumb, index, middle, ring, pinky) from summed bend angles."""
    out = []
    for name, chain in FINGER_CHAINS.items():
        bones = np.diff(lm[:, list(chain), :], axis=1)
        bones = bones / np.linalg.norm(bones, axis=-1, keepdims=True)
        cos = np.clip((bones[:, :-1] * bones[:, 1:]).sum(-1), -1.0, 1.0)
        out.append(np.arccos(cos).sum(-1) / FULL_CURL_RAD[name])
    return np.clip(np.stack(out, axis=-1), 0.0, 1.0)


@dataclass
class RetargetResult:
    q_target_rad: np.ndarray  # (N, DoF) actuator order (43 for Dex3, 73 for Sharpa)
    pos_err_m: np.ndarray  # (N,)
    rot_err_rad: np.ndarray  # (N,)
    ik_converged: np.ndarray  # (N,) bool


def _solve_wrist(
    ik: ArmIK,
    q: np.ndarray,
    pos: np.ndarray,
    R: np.ndarray,
    in_torso: Callable[[np.ndarray], bool],
    held: bool = False,
) -> IKResult:
    """Warm-started IK, restarted from the neutral arm when the warm start gets stuck.

    Tracking frame to frame can follow the arm into a branch from which a fast motion can no longer be
    reached: carrying the object back towards the chest on clip 104 (Dex3) the error grew to 0.3 m over
    a few frames although every one of those poses is reachable from the neutral arm. The restart is
    only taken if it keeps the arm out of the torso (the neutral seed can fold the elbow into it).
    While the fingers hold an object (``held``) a restart onto another branch is a one-frame whip:
    Sharpa clip 100 lifted with the warm start 1.5 cm off, the restart met the target on a branch
    1.6 rad away and flung the object 1 m. A restart that far is only taken there when the warm
    start is badly lost.
    """
    res = ik.solve(q, pos, R)
    if res.pos_err_m > RESTART_POS_ERR_M:
        q0 = q.copy()
        q0[ik.arm_idx] = 0.0
        alt = ik.solve(q0, pos, R)
        step = float(np.abs(alt.q_arm - q[ik.arm_idx]).max())
        if held and step > HELD_MAX_ARM_STEP_RAD and res.pos_err_m < HELD_RESTART_POS_ERR_M:
            return res
        if alt.pos_err_m < res.pos_err_m:
            q0[ik.arm_idx] = alt.q_arm
            if not in_torso(q0):
                return alt
    return res


def held_frames(gaps: np.ndarray) -> np.ndarray:
    """(N,) bool: frames whose commanded pinch gap is narrower than the open (approach) gap.

    These are the frames on which the fingers are closing on, holding or just letting go of the object;
    an approach-only gap curve (a constant gap) has none.
    """
    fin = np.isfinite(gaps)
    if not fin.any():
        return np.zeros(len(gaps), bool)
    return fin & (np.where(fin, gaps, np.inf) < np.nanmax(gaps) - 1e-6)


def _repair_held_flips(
    ik: ArmIK,
    q_out: np.ndarray,
    targets: list[tuple[np.ndarray, np.ndarray]],
    held: np.ndarray,
    pe: np.ndarray,
    re: np.ndarray,
    ok: np.ndarray,
) -> None:
    """Move an arm branch change out of a hold, in place.

    Frame-to-frame tracking can enter the hold on a branch that runs into the joint limits later in
    the carry (Sharpa clip 100: wrist roll pinned at +1.97 rad and the elbow at +2.09 rad while
    lifting); the only continuation is then another branch, reached in one frame, which flings the
    object (1.9 rad elbow step, the object thrown 1 m). The frames before the flip are re-solved
    backwards, each seeded from the next, on the branch the hold ends up on. The walk stops where it
    rejoins the original branch, or where it tracks the target worse than the original frame did;
    the remaining change of branch then happens before the grip closes and is spread over the
    free-motion frames before that point.
    """
    arm = ik.arm_idx
    for _ in range(int(held.sum())):
        steps = np.abs(np.diff(q_out[:, arm], axis=0)).max(axis=1)
        bad = np.flatnonzero((steps > HELD_MAX_ARM_STEP_RAD) & held[1:])
        if not len(bad):
            return
        k = int(bad[0]) + 1
        seed, new, merged = q_out[k].copy(), {}, False
        for j in range(k - 1, -1, -1):
            qj = q_out[j].copy()
            qj[arm] = seed[arm]
            res = ik.solve(qj, *targets[j])
            if (
                res.pos_err_m > pe[j] + REPAIR_POS_ERR_M
                or np.abs(res.q_arm - seed[arm]).max() > HELD_MAX_ARM_STEP_RAD
            ):
                break
            qj[arm] = res.q_arm
            new[j], seed = (qj, res), qj
            if np.abs(res.q_arm - q_out[j][arm]).max() < BRANCH_MERGE_RAD:
                merged = True
                break
        if not new or (held[min(new)] and not merged):
            return  # the flip cannot be moved out of the hold
        for j, (qj, res) in new.items():
            q_out[j], pe[j], re[j], ok[j] = qj, res.pos_err_m, res.rot_err_rad, res.converged
        j0 = min(new)
        if not merged and j0 > 0:
            b0 = max(j0 - BRANCH_BLEND_FRAMES, 0)
            b = q_out[j0][arm].copy()
            for jj in range(b0 + 1, j0):
                w = (jj - b0) / (j0 - b0)
                w = w * w * (3.0 - 2.0 * w)
                q_out[jj][arm] = (1.0 - w) * q_out[jj][arm] + w * b


def rate_limited_share(share: np.ndarray, rate: float) -> np.ndarray:
    """Largest curve that never exceeds ``share`` (N,) and changes by at most ``rate`` per frame."""
    k = np.arange(len(share))
    return np.clip(np.min(share[None, :] + rate * np.abs(k[:, None] - k[None, :]), axis=1), 0.0, 1.0)


def retarget_hand(
    landmarks_sim: np.ndarray,
    side: str,
    T_sim_robot_base: np.ndarray | None = None,
    pinch_gap_m: np.ndarray | None = None,
    wrist_offset_m: np.ndarray | None = None,
    hand: str = "dex3",
    waist_pitch_rad: float = 0.0,
    pinch_roll_rad: np.ndarray | None = None,
    cache: dict | None = None,
) -> RetargetResult:
    """``landmarks_sim``: (N, 21, 3) in the simulation frame (already aligned).

    ``cache`` (a dict the caller keeps between calls) makes repeated solves of a slightly changed input
    cheap: the frame loop is a deterministic chain, so the frames before the first changed input are
    taken from the previous call and the chain resumes from its saved state. The result is bit-identical.

    ``pinch_gap_m`` (N,), NaN where unused: on those frames the thumb/index joints are solved so the Dex3 tip
    gap equals the value (object width minus a squeeze) and the wrist is shifted so the Dex3 pinch point lands
    on the human pinch point (Dex3 is not human-sized, so plain wrist matching misses small objects).
    ``waist_pitch_rad`` holds the waist at a constant tilt (it re-aims the head camera); the arm IK
    then solves with the shoulders where that tilt puts them, and every output row carries the tilt.
    ``wrist_offset_m`` (N, 3) displaces the wrist target (used to push the idle hand out of collisions).
    """
    if T_sim_robot_base is None:
        T_sim_robot_base = np.eye(4)
        T_sim_robot_base[2, 3] = PELVIS_HEIGHT_M
    cfg = SceneConfig(T_sim_robot_base=T_sim_robot_base, hand=hand)
    model = build_scene(cfg)
    layout = RobotLayout.from_model(model)
    finger_idx = layout.finger_idx(side)
    ik = ArmIK(model, side)
    checker: list[ArmCollisionChecker] = []  # built on the first IK restart only

    def in_torso(q_full: np.ndarray) -> bool:
        if not checker:
            checker.append(ArmCollisionChecker(model))
        return bool(checker[0].torso_frames(q_full[None], side)[0])

    R = hand_frame(landmarks_sim, side, hand)
    pos = landmarks_sim[:, 0, :] + (0.0 if wrist_offset_m is None else wrist_offset_m)
    curls = finger_curls(landmarks_sim)
    n = len(pos)
    gaps = np.full(n, np.nan) if pinch_gap_m is None else np.asarray(pinch_gap_m, dtype=float)
    if hand == "sharpa":
        # Sharpa is human-sized: a relaxed ring/pinky hangs below the pinch point and hits the object
        # before the thumb and index reach it (clip 0: pinky tip pushed the block 3 cm aside). On the
        # pinching hand they stay tucked into the palm, as in a human precision grip, for the whole
        # clip: tucking them only for the grasp swats an object already beside the hand (clip 10).
        # The middle tip sits between the thumb and index tips and 1.3 cm below them, so on a wide
        # object it lands on the top face and holds the pinch above it (clip 100: tongs); it is
        # tucked as well, leaving a thumb-index precision pinch.
        if pinch_gap_m is not None:
            curls[:, 2:] = np.maximum(curls[:, 2:], PINCH_TUCK_CURL)
        pinch = SharpaPinchSolver(ik, side, finger_idx)
        fingers = map_curls_to_sharpa(curls, side)
    else:
        pinch = PinchSolver(ik, side, SIDE_SLICE[side])
        fingers = map_curls_to_dex3(curls, side)
    human_pinch = 0.5 * (landmarks_sim[:, 4] + landmarks_sim[:, 8])
    held = held_frames(gaps)
    roll_in = np.zeros(n) if pinch_roll_rad is None else np.asarray(pinch_roll_rad, float)
    # Everything frame i reads except the chain state: the target, fingers, gap, hold flag and roll.
    frame_sig = np.concatenate(
        [landmarks_sim.reshape(n, -1), pos, fingers, gaps[:, None], roll_in[:, None], held[:, None]], axis=1
    )
    chain_key = (side, hand, float(waist_pitch_rad), T_sim_robot_base.tobytes(), pinch_roll_rad is None)
    if cache is not None and cache.get("key") != chain_key:
        cache.clear()
        cache["key"] = chain_key

    def solve(fixed_frac: np.ndarray | None) -> tuple[RetargetResult, np.ndarray]:
        q = np.zeros(layout.dof)
        if waist_pitch_rad:
            q[layout.joint_idx(["waist_pitch_joint"])] = waist_pitch_rad
        q_out = np.zeros((n, layout.dof))
        pe, re, ok = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        used = np.ones(n)
        run = 0
        roll_axis, roll_sign = None, 0.0
        last_shift, last_turn, since = np.zeros(3), np.zeros(3), LEVEL_RAMP_FRAMES
        targets: list[tuple[np.ndarray, np.ndarray]] = []
        frac_col = np.full((n, 1), np.nan) if fixed_frac is None else fixed_frac[:, None]
        sig = np.concatenate([frame_sig, frac_col], axis=1)
        slot = (
            None
            if cache is None
            else cache.setdefault("pass_none" if fixed_frac is None else "pass_frac", {})
        )
        start = 0
        if slot and slot["sig"].shape == sig.shape:
            same = np.all((slot["sig"] == sig) | (np.isnan(slot["sig"]) & np.isnan(sig)), axis=1)
            start = n if same.all() else int(np.argmin(same))
        if start > 0:
            k = start - 1
            sn = slot["snaps"][k]
            q = sn["q"].copy()
            run, roll_axis, roll_sign, since = sn["run"], sn["roll_axis"], sn["roll_sign"], sn["since"]
            last_shift, last_turn = sn["last_shift"], sn["last_turn"]
            q_out[:start], pe[:start], re[:start], ok[:start] = (
                slot[x][:start] for x in ("q_out", "pe", "re", "ok")
            )
            used[:start] = slot["used"][:start]
            targets = list(slot["targets"][:start])
        snaps = [] if start == 0 else list(slot["snaps"][:start])
        for i in range(start, n):
            q[finger_idx] = fingers[i]
            # After a pinch the wrist eases back to the human one over the leveling ramp: dropping the
            # pinch shift and turn in one frame swung the Dex3 wrist 1.1 rad at the release on clip 101
            # and threw the object 12 cm back towards its start.
            since += 1
            fade = max(0.0, 1.0 - since / LEVEL_RAMP_FRAMES) if not np.isfinite(gaps[i]) else 0.0
            turn_back = Rotation.from_rotvec(fade * last_turn).as_matrix()
            target_i = (pos[i] + fade * last_shift, turn_back @ R[i])
            res = _solve_wrist(ik, q, *target_i, in_torso)
            q[ik.arm_idx] = res.q_arm
            if np.isfinite(gaps[i]):
                run = run + 1 if i > 0 and np.isfinite(gaps[i - 1]) else 0
                q = pinch.solve(q, gaps[i])
                tip_a, tip_b = pinch.tips(q)
                mid = 0.5 * (tip_a + tip_b)
                # Level the grip: a pinch tilted like the human wrist puts one finger on the top face and
                # squeezes partly downwards (Dex3 clip 101: +47 deg, the index pressed on top and the
                # block never left the table). The hand turns rigidly about the pinch point. Sharpa
                # pinches with thumb and index only: their axis is made horizontal. On Dex3 the thumb
                # opposes the index and middle side by side, so a level thumb-index axis alone stacks
                # the middle finger 3-7 cm below it, onto the top face; the middle mirrors the index and
                # the plane of the three tips is made horizontal instead.
                w_lev = min(1.0, (run + 1) / LEVEL_RAMP_FRAMES)
                if isinstance(pinch, PinchSolver):
                    q = pinch.mirror_middle(q)
                    normal = np.cross(tip_b - tip_a, pinch.middle_tip(q) - tip_a)
                    normal *= np.sign(normal[2]) or 1.0
                    lev = turn_rotation(normal, np.array([0.0, 0.0, 1.0]), w_lev)
                else:
                    lev = level_rotation(tip_b - tip_a, w_lev)
                wrist = ik.wrist_pose(q)[0]
                arm, rot = lev @ (wrist - mid), lev
                # Rolling the hand about the thumb-index axis keeps both tips where they are and swings
                # the palm up; the clip-level clearance uses it where the palm would sink into the table
                # at the grip height (clip 101 Sharpa: palm 3.2 cm deep, the hand was lifted instead and
                # the fingers shut 4.5 cm above the object's centre).
                if run == 0 or np.linalg.norm(tip_b - tip_a) >= ROLL_AXIS_MIN_GAP_M or roll_axis is None:
                    roll_axis = lev @ (tip_b - tip_a)
                if pinch_roll_rad is not None and pinch_roll_rad[i] > 0:
                    if run == 0 or not roll_sign:
                        roll_sign = float(np.sign(np.cross(roll_axis, arm)[2])) or 1.0
                    turn = axis_rotation(roll_axis, roll_sign * pinch_roll_rad[i])
                    arm, rot = turn @ arm, turn @ lev
                else:
                    roll_sign = 0.0
                # The leveled/rolled wrist can lie outside the arm's workspace although the human one did
                # not (Dex3 clip 0: 47% of frames unreachable, the pinch stalled 8 cm off the object).
                # The turn is then scaled back until the wrist is reachable again.
                rotvec = Rotation.from_matrix(rot).as_rotvec()
                base = wrist - mid
                # Seeded from the previous leveled pose, not from this frame's unleveled one: the two
                # orientations differ by the whole turn, and solving across it can hop to another arm
                # branch mid-grasp (Sharpa clip 116: wrist roll +1.4 rad in one frame flung the object).
                best = None
                q_arm0 = q_out[i - 1][ik.arm_idx].copy() if run > 0 else q[ik.arm_idx].copy()
                fracs = TURN_FALLBACK_FRACS if fixed_frac is None else (fixed_frac[i],)
                for frac in fracs:
                    rot_f = Rotation.from_rotvec(frac * rotvec).as_matrix()
                    target = human_pinch[i] + rot_f @ base
                    q[ik.arm_idx] = q_arm0
                    cand = _solve_wrist(ik, q, target, rot_f @ R[i], in_torso, held=bool(held[i]))
                    q[ik.arm_idx] = cand.q_arm
                    cand.pos_err_m = float(np.linalg.norm(target - ik.wrist_pose(q)[0]))
                    if best is None or cand.pos_err_m < best.pos_err_m:
                        best, used[i] = cand, frac
                        target_i = (target, rot_f @ R[i])
                    if cand.pos_err_m <= RESTART_POS_ERR_M:
                        break
                res = best
                q[ik.arm_idx] = res.q_arm
                turn_used = Rotation.from_rotvec(used[i] * rotvec).as_matrix()
                last_shift = human_pinch[i] + turn_used @ base - pos[i]
                last_turn, since = used[i] * rotvec, 0
            q_out[i] = q
            targets.append(target_i)
            pe[i], re[i], ok[i] = res.pos_err_m, res.rot_err_rad, res.converged
            if slot is not None:
                snaps.append(
                    {"q": q.copy(), "run": run, "roll_axis": roll_axis, "roll_sign": roll_sign,
                     "last_shift": last_shift.copy(), "last_turn": last_turn.copy(), "since": since}
                )  # fmt: skip
        if slot is not None:  # saved before the in-place repair below
            slot.update(sig=sig, snaps=snaps, q_out=q_out.copy(), pe=pe.copy(), re=re.copy(), ok=ok.copy(),
                        used=used.copy(), targets=list(targets))  # fmt: skip
        _repair_held_flips(ik, q_out, targets, held, pe, re, ok)
        return RetargetResult(q_out, pe, re, ok), used

    result, used = solve(None)
    if used.min() >= 1.0:
        return result
    # Picking the reachable share frame by frame flips it (1 -> 0 -> 0.25 on Dex3 clip 101), and every
    # flip swings the wrist about the pinch point: 15 cm and 50-100 deg in two frames, which knocks the
    # object over before the grip. The share is replaced by the largest rate-limited curve under the
    # per-frame one and solved again.
    return solve(rate_limited_share(used, TURN_FRAC_RATE))[0]


def robot_pinch_axis_xy(
    landmarks_sim: np.ndarray, side: str, gap_m: float = 0.04, hand: str = "dex3"
) -> np.ndarray:
    """Horizontal unit axis along which the robot thumb and index tips face each other for these frames.

    The robot pinch direction in the wrist frame is fixed per hand, so it can differ from the human
    thumb-index direction (Sharpa by ~50 deg on clip 1); the object has to be oriented to the axis of
    the hand that will close on it, or the commanded gap is narrower than the object along it.
    """
    n = len(landmarks_sim)
    res = retarget_hand(landmarks_sim, side, pinch_gap_m=np.full(n, gap_m), hand=hand)
    T = np.eye(4)
    T[2, 3] = PELVIS_HEIGHT_M
    model = build_scene(SceneConfig(T_sim_robot_base=T, hand=hand))
    ik = ArmIK(model, side)
    pinch = (
        SharpaPinchSolver(ik, side, RobotLayout.from_model(model).finger_idx(side))
        if hand == "sharpa"
        else PinchSolver(ik, side, SIDE_SLICE[side])
    )
    vecs = []
    for q in res.q_target_rad:
        a, b = pinch.tips(q)
        v = (b - a)[:2]
        vecs.append(v if not vecs or v @ vecs[0] >= 0 else -v)
    axis = np.median(vecs, axis=0)
    return axis / np.linalg.norm(axis)


def commanded_pinch_axis_xy(
    q_rad: np.ndarray, side: str, hand: str, T_sim_robot_base: np.ndarray
) -> np.ndarray:
    """Median horizontal thumb-to-index unit axis of already solved poses (N, DoF), in the sim frame.

    Where the wrist orientation is out of reach (rotation error ~1 rad on clip 1 Sharpa) the twist the
    arm ends up with depends on the IK branch of the whole trajectory, so the axis that the fingers
    actually close along can only be read off the final solution.
    """
    model = build_scene(SceneConfig(T_sim_robot_base=T_sim_robot_base, hand=hand))
    ik = ArmIK(model, side)
    pinch = (
        SharpaPinchSolver(ik, side, RobotLayout.from_model(model).finger_idx(side))
        if hand == "sharpa"
        else PinchSolver(ik, side, SIDE_SLICE[side])
    )
    vecs: list[np.ndarray] = []
    for q in q_rad:
        a, b = pinch.tips(q)
        v = (b - a)[:2]
        vecs.append(v if not vecs or v @ vecs[0] >= 0 else -v)
    axis = np.median(vecs, axis=0)
    return axis / np.linalg.norm(axis)


def landmarks_camera_to_sim(landmarks_camera: np.ndarray, T_sim_camera: np.ndarray) -> np.ndarray:
    return apply_se3(T_sim_camera, landmarks_camera)
