"""L.17: move the robot only when the head camera or the arm cannot do the job from the original stance.

The retarget places the human motion in front of a robot standing at the origin (``place_hand_motion``);
that placement is the original use case and is kept whenever it is good enough. It is good enough when the
pinch and wrist stay inside the narrow head-camera view during the *critical* frames (approach, grasp,
transport, release) and the arm does not lose reachability. If not, the robot is moved (x, y, yaw, within
``LEVELS``) and the head is re-aimed (waist pitch), preferring the smallest move that works. Moving the
robot by ``Delta`` is the same as re-anchoring the human motion by ``Delta^-1``, so every downstream
stage (table, object, place target) keeps working in the robot frame; ``start_xyyaw`` is where the
original robot pose lies in that frame, i.e. where the robot walks from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from common.transforms import make_se3
from stages.locomotion.base_placement import _BASE, UNREACHABLE_POS_M, Placement
from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.reconstruction.align.camera_fit import HeadCamera, score_pitch
from stages.retargeting.kinematic import landmarks_camera_to_sim, retarget_hand

PITCHES_RAD = (0.0, 0.15, 0.3, 0.45)
LEVELS = (
    (0.0, 0.0),
    (0.1, 0.2),
    (0.3, 0.4),
    (0.5, 0.6),
)  # (max |xy| move m, max |yaw| rad) per escalation level
COVERAGE_OK = 0.9  # critical frames inside the view
UNREACH_SLACK = 0.02  # a move may not add more unreachable frames than this
WINDOW_LEAD_S, WINDOW_TAIL_S = 1.0, 0.5  # critical frames: grasp - lead .. release + tail
IK_CANDIDATES = 12
NOMINAL_GAP_M = 0.04


@dataclass
class StanceAdjustment:
    placement: Placement
    delta_xyyaw: tuple[float, float, float]  # new robot pose in the original robot frame
    start_xyyaw: tuple[float, float, float]  # original robot pose in the new robot frame (walk start)
    coverage_before: float
    coverage_after: float
    unreachable_before: float
    unreachable_after: float
    level: int  # 0: kept the original stance (head pitch only), 1..3 moved the robot
    visible_ok: bool
    reason: str

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "placement"}
        d["waist_pitch_rad"] = self.placement.waist_pitch_rad
        return d


def critical_window(t_s: np.ndarray, grasp_t_s: float, release_t_s: float) -> np.ndarray:
    return (t_s >= grasp_t_s - WINDOW_LEAD_S) & (t_s <= release_t_s + WINDOW_TAIL_S)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def reanchor(T_sim_camera: np.ndarray, dx: float, dy: float, yaw: float) -> np.ndarray:
    """``T_sim_camera`` seen from a robot moved by (dx, dy, yaw) in the old frame."""
    R = _rz(-yaw)
    return make_se3(R, -R @ np.array([dx, dy, 0.0])) @ T_sim_camera


def _start_pose(dx: float, dy: float, yaw: float) -> tuple[float, float, float]:
    R = _rz(-yaw)
    p = -R @ np.array([dx, dy, 0.0])
    return float(p[0]), float(p[1]), float(-yaw)


def _unreachable(lm_sim: np.ndarray, side: str, hand: str, pitch: float, stride: int) -> float:
    gap = np.full(len(lm_sim[::stride]), NOMINAL_GAP_M)
    ik = retarget_hand(lm_sim[::stride], side, hand=hand, waist_pitch_rad=pitch, pinch_gap_m=gap)
    return float((ik.pos_err_m > UNREACHABLE_POS_M).mean())


def adjust_stance(
    landmarks_camera: np.ndarray,
    side: str,
    placement: Placement,
    window: np.ndarray,
    pinch_uv_ref: np.ndarray | None,
    hand: str = "sharpa",
    stride: int = 4,
) -> StanceAdjustment:
    """Keep ``placement`` if its critical-frame coverage and reachability are fine, else move the robot."""
    head = HeadCamera(build_scene(SceneConfig(T_sim_robot_base=_BASE, hand=hand)))

    def coverage(T: np.ndarray, pitch: float) -> float:
        lm = landmarks_camera_to_sim(landmarks_camera, T)
        pinch = 0.5 * (lm[:, 4] + lm[:, 8])
        ref = None if pinch_uv_ref is None else pinch_uv_ref[window]
        pts = np.stack([pinch, lm[:, 0]], axis=1)[window]
        return score_pitch(head, pitch, pts, pinch[window], ref).coverage_fraction

    T0 = placement.T_sim_camera
    cov0 = coverage(T0, placement.waist_pitch_rad)
    unr0 = placement.unreachable_fraction
    kept = StanceAdjustment(
        placement,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        cov0,
        cov0,
        unr0,
        unr0,
        0,
        cov0 >= COVERAGE_OK,
        "original",
    )
    if kept.visible_ok:
        return kept
    best: StanceAdjustment | None = None
    for level, (max_xy, max_yaw) in enumerate(LEVELS):
        xs = np.linspace(-max_xy, max_xy, 5) if max_xy else (0.0,)
        yaws = np.linspace(-max_yaw, max_yaw, 5) if max_yaw else (0.0,)
        cands = []
        for dx in xs:
            for dy in xs:
                for yaw in yaws:
                    T = reanchor(T0, dx, dy, yaw)
                    for pitch in PITCHES_RAD:
                        cands.append(
                            (coverage(T, pitch), float(np.hypot(dx, dy) + abs(yaw)), pitch, T, (dx, dy, yaw))
                        )
        cands.sort(key=lambda c: (-round(c[0], 3), c[1], c[2]))
        for cov, _, pitch, T, (dx, dy, yaw) in cands[:IK_CANDIDATES]:
            lm = landmarks_camera_to_sim(landmarks_camera, T)
            unr = _unreachable(lm, side, hand, pitch, stride)
            if unr > unr0 + UNREACH_SLACK:
                continue
            adj = StanceAdjustment(
                replace(placement, T_sim_camera=T, waist_pitch_rad=pitch, unreachable_fraction=unr),
                (float(dx), float(dy), float(yaw)),
                _start_pose(dx, dy, yaw),
                cov0,
                cov,
                unr0,
                unr,
                level,
                cov >= COVERAGE_OK,
                f"moved at level {level}" if cov >= COVERAGE_OK else "best effort, view below threshold",
            )
            if best is None or cov > best.coverage_after + 1e-6:
                best = adj
            if adj.visible_ok:
                return adj
    if best is not None and best.coverage_after > cov0 + 1e-6:
        return best
    return replace(kept, reason="VISIBILITY_UNACHIEVABLE within the stance limits")
