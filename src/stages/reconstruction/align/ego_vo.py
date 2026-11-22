"""Lightweight monocular visual odometry for walking ego clips.

Chains frame-to-frame SIFT matches; metric scale comes from per-frame depth
(PnP on depth-lifted keypoints). Dynamic pixels (hands, held object) are masked
out of feature extraction. Poses are camera-to-world with frame 0 = identity,
in the OpenCV camera convention (x right, y down, z forward).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VOResult:
    poses: np.ndarray  # (T, 4, 4) T_world_cam
    inliers: np.ndarray  # (T,) inlier count of the pair (i-1, i); 0 for frame 0 / failures
    ok: np.ndarray  # (T,) bool, False where the motion was extrapolated


def _features(gray: np.ndarray, mask: np.ndarray | None, n: int):
    sift = cv2.SIFT_create(nfeatures=n)
    valid = None if mask is None else (~mask).astype(np.uint8) * 255
    return sift.detectAndCompute(gray, valid)


def _match(da: np.ndarray, db: np.ndarray, ratio: float = 0.75):
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    return [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]


def _lift(kps, matches, depth: np.ndarray, K: np.ndarray, scale_xy: float):
    """Pixels of image A (in matches order) lifted to 3D with depth sampled at depth resolution."""
    uv = np.array([kps[m.queryIdx].pt for m in matches], dtype=np.float64)
    h, w = depth.shape
    ix = np.clip((uv[:, 0] * scale_xy).round().astype(int), 0, w - 1)
    iy = np.clip((uv[:, 1] * scale_xy).round().astype(int), 0, h - 1)
    d = depth[iy, ix].astype(np.float64)
    good = np.isfinite(d) & (d > 0.05)
    xyz = np.stack([(uv[:, 0] - K[0, 2]) * d / K[0, 0], (uv[:, 1] - K[1, 2]) * d / K[1, 1], d], axis=1)
    return xyz, good


def _estimate_chain(
    frames: np.ndarray,
    K: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray | None = None,
    n_features: int = 2000,
    min_inliers: int = 30,
    reproj_px: float = 3.0,
) -> VOResult:
    """frames (T,H,W,3) uint8 RGB; depths (T,h,w) metric z-depth (any resolution); masks (T,H,W) True = ignore."""
    T = len(frames)
    scale_xy = depths.shape[2] / frames.shape[2]
    feats = []
    for i in range(T):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        feats.append(_features(g, None if masks is None else masks[i], n_features))

    poses = np.tile(np.eye(4), (T, 1, 1))
    inl = np.zeros(T, dtype=int)
    ok = np.ones(T, dtype=bool)
    last_step = np.eye(4)  # T_prev_cur of the last good pair, reused when tracking fails
    dist = np.zeros(4)
    for i in range(1, T):
        ka, da = feats[i - 1]
        kb, db = feats[i]
        step = None
        if da is not None and db is not None and len(ka) > 8 and len(kb) > 8:
            m = _match(da, db)
            if len(m) >= min_inliers:
                # Lift previous-frame keypoints, solve current-frame pose by PnP.
                xyz, good = _lift(ka, m, depths[i - 1], K, scale_xy)
                uv_b = np.array([kb[x.trainIdx].pt for x in m], dtype=np.float64)
                if good.sum() >= min_inliers:
                    found, rvec, tvec, idx = cv2.solvePnPRansac(
                        xyz[good], uv_b[good], K, dist, reprojectionError=reproj_px, iterationsCount=200
                    )
                    if found and idx is not None and len(idx) >= min_inliers:
                        T_cur_prev = np.eye(4)
                        T_cur_prev[:3, :3] = cv2.Rodrigues(rvec)[0]
                        T_cur_prev[:3, 3] = tvec[:, 0]
                        step = np.linalg.inv(T_cur_prev)  # T_prev_cur
                        inl[i] = len(idx)
        if step is None:
            ok[i] = False
            step = last_step
        else:
            last_step = step
        poses[i] = poses[i - 1] @ step
    return VOResult(poses=poses, inliers=inl, ok=ok)


def smooth_poses(poses: np.ndarray, window: int = 15) -> np.ndarray:
    """Low-pass the camera path: moving average of positions and of rotation vectors (about frame 0)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial.transform import Rotation

    pos = uniform_filter1d(poses[:, :3, 3], window, axis=0, mode="nearest")
    rv = Rotation.from_matrix(poses[:, :3, :3]).as_rotvec()
    rv = np.unwrap(rv, axis=0)  # keep the rotation vectors continuous before averaging
    rot = Rotation.from_rotvec(uniform_filter1d(rv, window, axis=0, mode="nearest")).as_matrix()
    out = np.tile(np.eye(4), (len(poses), 1, 1))
    out[:, :3, :3] = rot
    out[:, :3, 3] = pos
    return out


def points_to_world(points_cam: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Map per-frame camera-frame points (T, ..., 3) into the frame-0 camera frame."""
    shape = points_cam.shape
    flat = points_cam.reshape(shape[0], -1, 3)
    out = np.einsum("tij,tnj->tni", poses[:, :3, :3], flat) + poses[:, None, :3, 3]
    return out.reshape(shape)


def estimate_poses(
    frames: np.ndarray,
    K: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray | None = None,
    stride: int = 1,
    **kwargs,
) -> VOResult:
    """Chain over every ``stride``-th frame (longer baselines, less drift) and interpolate in between."""
    if stride <= 1:
        return _estimate_chain(frames, K, depths, masks, **kwargs)
    from scipy.spatial.transform import Rotation, Slerp

    T = len(frames)
    idx = np.unique(np.r_[np.arange(0, T, stride), T - 1])
    sub = _estimate_chain(frames[idx], K, depths[idx], None if masks is None else masks[idx], **kwargs)
    t = np.arange(T)
    poses = np.tile(np.eye(4), (T, 1, 1))
    poses[:, :3, :3] = Slerp(idx, Rotation.from_matrix(sub.poses[:, :3, :3]))(t).as_matrix()
    for k in range(3):
        poses[:, k, 3] = np.interp(t, idx, sub.poses[:, k, 3])
    inl = np.zeros(T, dtype=int)
    ok = np.ones(T, dtype=bool)
    for j in range(1, len(idx)):
        seg = slice(idx[j - 1] + 1, idx[j] + 1)
        inl[seg] = sub.inliers[j]
        ok[seg] = sub.ok[j]
    return VOResult(poses=poses, inliers=inl, ok=ok)
