"""Choose the robot head-camera tilt so the hand and the manipulation stay in view like in the video.

The Unitree D435 is rigidly mounted on the torso, so the only physical way to re-aim it is to pitch the
waist. This module scores a static waist pitch by projecting the pinch trajectory through the real D435
intrinsics and comparing it with where the human pinch appeared in the original (human camera) image.

Two terms, both in normalised image coordinates (u, v) in [0, 1]:
  * coverage: a hinge that is zero while the hand stays inside the central ``1 - 2 * SAFE_MARGIN`` box;
  * layout: squared distance between the robot's and the human's normalised pinch position, so the hand
    sits at a similar place in the picture as in the source video.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from stages.manipulation.mjcf_scene import D435_IMAGE_HW

WAIST_PITCH_JOINT = "waist_pitch_joint"
PITCH_GRID_RAD = np.linspace(-0.45, 0.45, 19)  # the joint range is +-0.52 rad
SAFE_MARGIN = 0.10  # the hand must stay this far (fraction of the image) from every border
NEAR_M = 0.15  # closer than this the pinch is inside the lens
COVERAGE_WEIGHT = 10.0
LAYOUT_WEIGHT = 1.0
IMAGE_HW = D435_IMAGE_HW  # the ego panel's render size; only the aspect ratio matters


@dataclass(frozen=True)
class CameraScore:
    waist_pitch_rad: float
    cost: float
    coverage_fraction: float  # frames whose anchors all lie inside the safe box
    layout_err: float  # mean normalised-image distance to the human layout (1.0 = full image width)


def human_uv(points_camera: np.ndarray, focal_px: float, image_wh: tuple[float, float]) -> np.ndarray:
    """(N, 3) points in the OpenCV human camera frame -> (N, 2) normalised pixel coordinates."""
    w, h = float(image_wh[0]), float(image_wh[1])
    z = np.maximum(points_camera[:, 2], 1e-6)
    return np.stack(
        [(focal_px * points_camera[:, 0] / z + w / 2) / w, (focal_px * points_camera[:, 1] / z + h / 2) / h],
        1,
    )


class HeadCamera:
    """Forward kinematics of the D435 as a function of the waist pitch (pelvis fixed)."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, WAIST_PITCH_JOINT)
        self.qadr = int(model.jnt_qposadr[jid])
        self.fovy_rad = float(np.radians(model.cam_fovy[self.cam]))

    def pose(self, waist_pitch_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """World position (3,) and rotation (3, 3) of the camera; it looks along its own -z, +y up."""
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qadr] = waist_pitch_rad
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        return self.data.cam_xpos[self.cam].copy(), self.data.cam_xmat[self.cam].reshape(3, 3).copy()

    def project(self, points_world: np.ndarray, waist_pitch_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) world points -> normalised (u, v) (N, 2) and depth along the optical axis (N,)."""
        pos, rot = self.pose(waist_pitch_rad)
        p = (np.asarray(points_world, float) - pos) @ rot  # rows are R^T (x - c)
        depth = -p[:, 2]
        tan_y = np.tan(self.fovy_rad / 2)
        tan_x = tan_y * IMAGE_HW[1] / IMAGE_HW[0]
        d = np.maximum(depth, 1e-6)
        return np.stack([0.5 + 0.5 * p[:, 0] / (d * tan_x), 0.5 - 0.5 * p[:, 1] / (d * tan_y)], 1), depth


def score_pitch(
    cam: HeadCamera,
    waist_pitch_rad: float,
    anchors_world: np.ndarray,  # (N, K, 3) points that must stay visible, K per frame
    layout_world: np.ndarray,  # (N, 3) the pinch point
    layout_uv_ref: np.ndarray | None,  # (N, 2) where the human pinch appeared
) -> CameraScore:
    n, k, _ = anchors_world.shape
    uv, depth = cam.project(anchors_world.reshape(-1, 3), waist_pitch_rad)
    uv = uv.reshape(n, k, 2)
    depth = depth.reshape(n, k)
    lo, hi = SAFE_MARGIN, 1.0 - SAFE_MARGIN
    excess = np.maximum.reduce([lo - uv[..., 0], uv[..., 0] - hi, lo - uv[..., 1], uv[..., 1] - hi])
    excess = np.where(depth < NEAR_M, 1.0, np.maximum(excess, 0.0))  # behind or inside the lens: out
    inside = (excess <= 0).all(axis=1)
    coverage_cost = float((excess**2).mean() + excess.max())
    layout_err, layout_cost = 0.0, 0.0
    if layout_uv_ref is not None:
        uv_p, _ = cam.project(layout_world, waist_pitch_rad)
        dist = np.linalg.norm(uv_p - layout_uv_ref, axis=1)
        layout_err, layout_cost = float(dist.mean()), float((dist**2).mean())
    cost = COVERAGE_WEIGHT * coverage_cost + LAYOUT_WEIGHT * layout_cost
    return CameraScore(float(waist_pitch_rad), cost, float(inside.mean()), layout_err)


def best_waist_pitch(
    cam: HeadCamera,
    anchors_world: np.ndarray,
    layout_world: np.ndarray,
    layout_uv_ref: np.ndarray | None,
    grid: np.ndarray = PITCH_GRID_RAD,
) -> CameraScore:
    """Static waist pitch (rad) with the lowest cost; ties go to the smallest tilt."""
    scores = [score_pitch(cam, float(p), anchors_world, layout_world, layout_uv_ref) for p in grid]
    return min(scores, key=lambda s: (round(s.cost, 9), abs(s.waist_pitch_rad)))
