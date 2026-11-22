"""Camera -> simulation frame: gravity alignment plus table anchoring.

The simulation frame is z-up with the table top at ``table_top_z_m`` and the robot pelvis near the origin.
Absolute camera-frame depth from monocular RGB is biased (focal error), so the horizontal and vertical
position is anchored to the table and the robot workspace rather than trusted metrically.
"""

import numpy as np

from common.transforms import make_se3

TIP_IDX = (4, 8, 12, 16, 20)
REST_FRAC = 0.05  # first and last share of the clip taken as the hand's rest pose
REST_RADIUS_M = 0.08  # the hand is working once its wrist is this far (horizontally) from both rests
MIN_WORK_FRAC = 0.1


def working_frames(pts: np.ndarray) -> np.ndarray:
    """(N, 21, 3) gravity-aligned landmarks -> (N,) bool, frames with the hand away from its rest poses.

    A hand resting in the lap is not at the table: anchoring the table to those fingertips put the
    work 20 cm high (clip 108), which the table fit then lowered back out of the arm's reach."""
    wrist = pts[:, 0, :2]
    n = max(1, int(round(REST_FRAC * len(pts))))
    rests = (wrist[:n].mean(axis=0), wrist[-n:].mean(axis=0))
    away = np.min([np.linalg.norm(wrist - r_, axis=1) for r_ in rests], axis=0) > REST_RADIUS_M
    return away if away.mean() >= MIN_WORK_FRAC else np.ones(len(pts), bool)


def gravity_aligned_rotation(up_in_camera: np.ndarray, forward_in_camera=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation R_sim_camera whose z axis is gravity-up and x axis the horizontal camera forward."""
    z = np.asarray(up_in_camera, dtype=float)
    z = z / np.linalg.norm(z)
    f = np.asarray(forward_in_camera, dtype=float)
    x = f - (f @ z) * z
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("camera looks along gravity; heading is undefined")
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows are sim axes expressed in camera coordinates


def anchor_to_table(
    R_sim_camera: np.ndarray,
    landmarks_camera: np.ndarray,
    side: str,
    table_top_z_m: float = 0.80,
    target_xy_m: tuple[float, float] = (0.35, 0.18),
    tip_clearance_m: float = 0.01,
) -> np.ndarray:
    """Return T_sim_camera (4x4) placing the active hand's motion in the robot workspace.

    ``landmarks_camera`` is (N, 21, 3). Vertical: the lowest 2% of fingertip heights while the hand is
    working (away from its rest poses, see ``working_frames``) sit on the table.
    Horizontal: the mean wrist position lands at ``target_xy_m`` (mirrored for the left hand).
    """
    pts = np.einsum("ij,nkj->nki", R_sim_camera, landmarks_camera)
    tips_z = pts[working_frames(pts)][:, TIP_IDX, 2]
    shift_z = table_top_z_m + tip_clearance_m - np.percentile(tips_z, 2)
    wrist_xy = pts[:, 0, :2].mean(axis=0)
    tx, ty = target_xy_m
    ty = ty if side == "left" else -ty
    t = np.array([tx - wrist_xy[0], ty - wrist_xy[1], shift_z])
    return make_se3(R_sim_camera, t)
