"""L.10: head-camera visibility of object points from a candidate stance (static posture)."""

from __future__ import annotations

import mujoco
import numpy as np

from robot.head_camera import HeadCamera
from stages.locomotion.plan.candidates import Stance

BORDER_MARGIN = 0.15
MIN_VISIBLE_FRACTION = 0.8
MIN_VISIBLE_FRACTION_GOAL = 0.4  # placing: the object is in hand and the narrow FOV rarely covers both
OCCLUSION_EPS_M = 0.02


def set_stance(model, data, stance: Stance) -> None:
    """Place the free-base robot at the stance (legs stay at the model default; refined at arrival)."""
    data.qpos[:3] = (stance.x, stance.y, stance.height_m)
    data.qpos[3:7] = (np.cos(stance.yaw / 2), 0.0, 0.0, np.sin(stance.yaw / 2))
    data.qpos[model.joint("waist_pitch_joint").qposadr[0]] = stance.waist_pitch_rad
    mujoco.mj_kinematics(model, data)


def visible_fraction(
    model, data, cam: HeadCamera, stance: Stance, pts_world: np.ndarray, margin: float = BORDER_MARGIN
) -> float:
    """Fraction of ``pts_world`` inside the image (with a border margin) and not blocked by world geometry."""
    set_stance(model, data, stance)
    T = cam.T_world_cam(model, data)
    uv, z = cam.project(T, pts_world)
    inside = (
        (z > 0.05)
        & (uv[:, 0] >= margin * cam.width)
        & (uv[:, 0] <= (1 - margin) * cam.width)
        & (uv[:, 1] >= margin * cam.height)
        & (uv[:, 1] <= (1 - margin) * cam.height)
    )
    ok = 0
    for p, ins in zip(pts_world, inside, strict=True):
        if not ins:
            continue
        vec = p - T[:3, 3]
        dist = float(np.linalg.norm(vec))
        gid = np.zeros(1, dtype=np.int32)
        hit = mujoco.mj_ray(model, data, T[:3, 3], vec / dist, None, 1, -1, gid)
        if hit < 0 or hit >= dist - OCCLUSION_EPS_M or model.geom_bodyid[gid[0]] != 0:
            ok += 1  # robot self-occlusion by arms is checked later with the IK posture
    return ok / len(pts_world)


def box_surface_points(center, half_size, n_per_axis: int = 3) -> np.ndarray:
    """Grid samples on the six faces of an axis-aligned box."""
    g = np.linspace(-1, 1, n_per_axis)
    pts = []
    for ax in range(3):
        for s in (-1, 1):
            a, b = [i for i in range(3) if i != ax]
            for u in g:
                for v in g:
                    p = np.zeros(3)
                    p[ax], p[a], p[b] = s, u, v
                    pts.append(np.asarray(center) + p * np.asarray(half_size))
    return np.unique(np.round(pts, 9), axis=0)
