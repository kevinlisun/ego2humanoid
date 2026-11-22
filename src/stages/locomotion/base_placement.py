"""Choose where the (fixed-pelvis) robot stands relative to the hand motion.

The camera->sim anchor is searched over a small grid of horizontal target positions; the candidate with
the lowest wrist IK error over a subsample of frames wins (both hands when the passive one is given).
Reports unreachable frames.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from robot.collision import ArmCollisionChecker
from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.reconstruction.align.camera_fit import HeadCamera, score_pitch
from stages.reconstruction.align.frames import anchor_to_table
from stages.retargeting.kinematic import PELVIS_HEIGHT_M, landmarks_camera_to_sim, retarget_hand

X_GRID_M = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)  # >= 0.20 keeps the hand clear of the torso
Y_GRID_M = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
UNREACHABLE_POS_M = 0.03
NOMINAL_PINCH_GAP_M = 0.04  # typical held-object width for scoring a standing spot
PASSIVE_WEIGHT = 0.2  # the active hand drives the anchor; the other hand must stay reachable
COLLISION_WEIGHT_M = 0.1  # metres of wrist error one fully colliding trajectory is worth
CAMERA_PITCHES_RAD = (-0.3, -0.15, 0.0, 0.15, 0.3)  # static waist tilts tried per standing spot
IK_SLACK_M = 0.01  # placements within this score of the most reachable one compete on camera view
UNREACH_SLACK = 0.01  # ... provided they add (almost) no unreachable frames
_BASE = np.eye(4)
_BASE[2, 3] = PELVIS_HEIGHT_M
OTHER_SIDE = {"left": "right", "right": "left"}


def ik_waist(model) -> int:
    """Actuator index of the waist pitch joint (q vectors are in actuator order)."""
    from robot.layout import RobotLayout

    return int(RobotLayout.from_model(model).joint_idx(["waist_pitch_joint"])[0])


@dataclass
class Placement:
    T_sim_camera: np.ndarray
    target_xy_m: tuple[float, float]
    mean_pos_err_m: float
    unreachable_fraction: float
    passive_unreachable_fraction: float = 0.0
    waist_pitch_rad: float = 0.0  # static waist tilt that aims the head camera at the manipulation
    camera_coverage: float = 1.0  # fraction of frames whose pinch stays inside the camera's safe box
    camera_layout_err: float = 0.0  # mean normalised-image distance to the human pinch layout
    place_unreachable_fraction: float = 0.0  # unreachable share of the frames from the place start on


@dataclass
class PlaceGoal:
    """The place phase is asked to move the hand by ``offset_sim_m`` over the frames ``start..release``."""

    offset_sim_m: tuple[float, float, float]
    start: int
    release: int
    reach_weight_m: float = 0.1  # wrist error one fully unreachable place window adds to a spot's score

    def apply(self, landmarks_camera: np.ndarray, R_sim_camera: np.ndarray) -> np.ndarray:
        w = np.clip((np.arange(len(landmarks_camera)) - self.start) / max(self.release - self.start, 1), 0, 1)
        w[self.release :] = 1.0
        return landmarks_camera + w[:, None, None] * (R_sim_camera.T @ np.asarray(self.offset_sim_m))


_CTX: dict = {}
_WORKER: dict = {}


def _workers() -> int:
    """Processes for the candidate grid: $E2H_WORKERS, default all cores (1 = serial)."""
    return max(1, int(os.environ.get("E2H_WORKERS", os.cpu_count() or 1)))


def _score_spot(xy: tuple[float, float]) -> list[tuple[float, float, Placement]]:
    """Every waist pitch of one standing spot, scored as ``place_hand_motion`` ranks them."""
    c = _CTX
    x, y = xy
    if _WORKER.get("ctx_id") != id(c):  # one model per process and call
        model = build_scene(SceneConfig(T_sim_robot_base=_BASE, hand=c["hand"]))
        _WORKER.update(
            ctx_id=id(c),
            model=model,
            checker=ArmCollisionChecker(model),
            head=HeadCamera(model) if c["pinch_uv_ref"] is not None else None,
        )
    model, checker, head = _WORKER["model"], _WORKER["checker"], _WORKER["head"]
    landmarks_camera, passive_landmarks_camera = c["landmarks_camera"], c["passive_landmarks_camera"]
    side, hand, stride, pinch_uv_ref = c["side"], c["hand"], c["stride"], c["pinch_uv_ref"]
    scored: list[tuple[float, float, Placement]] = []
    pitches = (0.0,) if head is None else CAMERA_PITCHES_RAD
    T = anchor_to_table(c["R_sim_camera"], landmarks_camera, side, c["table_top_z_m"], (x, y))
    lm = landmarks_camera_to_sim(c["ik_landmarks_camera"][::stride], T)
    place_frames = (
        None
        if c["place_goal"] is None
        else np.arange(len(landmarks_camera))[::stride] >= c["place_goal"].start
    )
    lm_p = None
    if passive_landmarks_camera is not None:  # the other arm moves with the same camera anchor
        lm_p = landmarks_camera_to_sim(passive_landmarks_camera[::stride], T)
    lm_view = landmarks_camera_to_sim(landmarks_camera[::stride], T)  # the head camera sees the human path
    pinch = 0.5 * (lm_view[:, 4] + lm_view[:, 8])
    for pitch in pitches:
        cam_score = None
        if head is not None:
            cam_score = score_pitch(
                head, pitch, np.stack([pinch, lm_view[:, 0]], axis=1), pinch, pinch_uv_ref[::stride]
            )
        # Scored as the hand will be driven: with its own geometry and a pinch held open
        # around an object, which moves the wrist (and the arm links) by several centimetres.
        gap = np.full(len(lm), NOMINAL_PINCH_GAP_M)
        ik = retarget_hand(lm, side, hand=hand, waist_pitch_rad=pitch, pinch_gap_m=gap)
        err = ik.pos_err_m
        p_err, collisions, q_both = np.zeros(1), 0.0, ik.q_target_rad
        if lm_p is not None:
            p_ik = retarget_hand(lm_p, OTHER_SIDE[side], hand=hand, waist_pitch_rad=pitch)
            p_err = p_ik.pos_err_m
            q_both = ik.q_target_rad + p_ik.q_target_rad
            q_both[:, ik_waist(model)] = pitch  # each arm solve carries the tilt; not added twice
            collisions = checker.count(q_both) / len(lm)
        # A target close in front of the chest folds the active elbow/wrist into the torso, which
        # the PD controller cannot follow (clip 0: the pinch stalled 12 cm above the object).
        collisions += float(checker.torso_frames(q_both, side).mean())
        place_unreach = 0.0
        if place_frames is not None and place_frames.any():
            place_unreach = float((err[place_frames] > UNREACHABLE_POS_M).mean())
        cand = Placement(
            T,
            (x, y),
            float(err.mean()),
            float((err > UNREACHABLE_POS_M).mean()),
            float((p_err > UNREACHABLE_POS_M).mean()),
            pitch,
            1.0 if cam_score is None else cam_score.coverage_fraction,
            0.0 if cam_score is None else cam_score.layout_err,
            place_unreach,
        )
        score = cand.mean_pos_err_m + PASSIVE_WEIGHT * float(p_err.mean()) + COLLISION_WEIGHT_M * collisions
        if place_frames is not None:  # the place window is a few frames of the clip: weigh it on its own
            score += c["place_goal"].reach_weight_m * place_unreach
        scored.append((score, 0.0 if cam_score is None else cam_score.cost, cand))
    return scored


def place_hand_motion(
    landmarks_camera: np.ndarray,
    R_sim_camera: np.ndarray,
    side: str,
    table_top_z_m: float = 0.80,
    stride: int = 4,
    passive_landmarks_camera: np.ndarray | None = None,
    pinch_uv_ref: np.ndarray | None = None,
    hand: str = "dex3",
    place_goal: PlaceGoal | None = None,
) -> Placement:
    """``place_goal``: where the place phase has to reach beyond the human motion (see ``PlaceGoal``); the
    standing spot is then scored on the corrected path, so a receptacle the arm cannot reach from the
    default spot moves the robot instead of cutting the correction.

    ``pinch_uv_ref`` (N, 2): normalised image position of the human pinch in the source video. When
    given, the head camera's waist tilt and the standing position are chosen together, so the hand and
    the manipulation stay in view at a similar place in the picture as in the video."""
    global _CTX
    _CTX = {
        "landmarks_camera": landmarks_camera,
        "ik_landmarks_camera": landmarks_camera
        if place_goal is None
        else place_goal.apply(landmarks_camera, R_sim_camera),
        "R_sim_camera": R_sim_camera,
        "side": side,
        "table_top_z_m": table_top_z_m,
        "stride": stride,
        "passive_landmarks_camera": passive_landmarks_camera,
        "pinch_uv_ref": pinch_uv_ref,
        "hand": hand,
        "place_goal": place_goal,
    }
    grid = [(x, y) for x in X_GRID_M for y in Y_GRID_M]
    workers = min(_workers(), len(grid))
    if workers > 1:
        # The spots are independent: fork so each worker inherits the inputs and builds its own model.
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(_score_spot, grid, chunksize=1))
    else:
        results = [_score_spot(xy) for xy in grid]
    _CTX = {}
    scored = [item for spot in results for item in spot]
    return select_placement(scored)


def select_placement(scored: list[tuple[float, float, Placement]]) -> Placement:
    """``scored``: (reach score, camera cost, candidate). The best camera view among the reachable ones."""
    best_ik = min(s for s, _, _ in scored)
    best_unreach = min(cand.unreachable_fraction for _, _, cand in scored)
    # Reachability comes first; the head camera only chooses among near-equally reachable standing spots,
    # and never one that leaves frames out of reach (clip 108: a 7 mm mean error hid 6% unreachable
    # frames at the grasp, and the hand stalled 4 cm short of the pen).
    near = [
        (c, s, cand)
        for s, c, cand in scored
        if s <= best_ik + IK_SLACK_M and cand.unreachable_fraction <= best_unreach + UNREACH_SLACK
    ]
    if not near:
        # The lowest-error spot leaves frames out of reach and every spot that does not is well behind
        # it on error (Dex3 clip 0 once the pinch leveling backs off where it is unreachable): the
        # reachable ones compete among themselves.
        reach = [
            (s, c, cand) for s, c, cand in scored if cand.unreachable_fraction <= best_unreach + UNREACH_SLACK
        ]
        best_r = min(s for s, _, _ in reach)
        near = [(c, s, cand) for s, c, cand in reach if s <= best_r + IK_SLACK_M]
    return min(near, key=lambda e: (e[0], e[1]))[2]
