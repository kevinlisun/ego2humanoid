"""Hand-pose estimation vs ground truth. GT is an evaluation reference only, never a pipeline input.

Joint arrays are (N, J, 3) in metres with a ``valid`` mask (N,) marking frames where both exist.
"""

import numpy as np

from common.transforms import apply_se3, rotation_angle, umeyama_sim3

WRIST_JOINT = 0


def mpjpe_m(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Mean per-joint position error with no alignment."""
    return float(_err(pred, gt, valid).mean())


def root_relative_mpjpe_m(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    """MPJPE after subtracting each frame's wrist joint (removes global translation error)."""
    p = pred - pred[:, WRIST_JOINT : WRIST_JOINT + 1]
    g = gt - gt[:, WRIST_JOINT : WRIST_JOINT + 1]
    return mpjpe_m(p, g, valid)


def sim3_aligned_mpjpe_m(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None
) -> tuple[float, float]:
    """One Sim3 fitted over the whole sequence (monocular scale/world ambiguity); returns (mpjpe_m, scale)."""
    m = _mask(pred, valid)
    T = umeyama_sim3(pred[m].reshape(-1, 3), gt[m].reshape(-1, 3), with_scale=True)
    aligned = apply_se3(T, pred.reshape(-1, 3)).reshape(pred.shape)
    return mpjpe_m(aligned, gt, valid), float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


def pa_mpjpe_m(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Per-frame Procrustes-aligned MPJPE (articulation quality, independent of pose and scale)."""
    errs = []
    for i in np.flatnonzero(_mask(pred, valid)):
        T = umeyama_sim3(pred[i], gt[i], with_scale=True)
        errs.append(np.linalg.norm(apply_se3(T, pred[i]) - gt[i], axis=-1).mean())
    return float(np.mean(errs))


def wrist_pose_errors(
    T_pred: np.ndarray, T_gt: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, float]:
    """Wrist SE3 (N, 4, 4) translation (m) and rotation (deg) errors, mean and max."""
    m = _mask(T_pred, valid)
    dt = np.linalg.norm(T_pred[m, :3, 3] - T_gt[m, :3, 3], axis=-1)
    dR = np.degrees(rotation_angle(np.swapaxes(T_gt[m, :3, :3], -1, -2) @ T_pred[m, :3, :3]))
    return {
        "wrist_trans_mean_m": float(dt.mean()),
        "wrist_trans_max_m": float(dt.max()),
        "wrist_rot_mean_deg": float(dR.mean()),
        "wrist_rot_max_deg": float(dR.max()),
    }


def evaluate_hands(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    valid: np.ndarray | None = None,
    pred_wrist: np.ndarray | None = None,
    gt_wrist: np.ndarray | None = None,
) -> dict[str, float]:
    """Full report. Raises if no frame is valid so an empty overlap can never look like a pass."""
    if not _mask(pred_joints, valid).any():
        raise ValueError("no valid frames to evaluate")
    sim3_err, scale = sim3_aligned_mpjpe_m(pred_joints, gt_joints, valid)
    out = {
        "frames": int(_mask(pred_joints, valid).sum()),
        "mpjpe_m": mpjpe_m(pred_joints, gt_joints, valid),
        "root_relative_mpjpe_m": root_relative_mpjpe_m(pred_joints, gt_joints, valid),
        "sim3_aligned_mpjpe_m": sim3_err,
        "sim3_scale": scale,
        "pa_mpjpe_m": pa_mpjpe_m(pred_joints, gt_joints, valid),
    }
    if pred_wrist is not None and gt_wrist is not None:
        out.update(wrist_pose_errors(pred_wrist, gt_wrist, valid))
    return out


def _mask(a: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    ok = np.isfinite(a.reshape(len(a), -1)).all(axis=1)
    return ok if valid is None else ok & np.asarray(valid, dtype=bool)


def _err(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    m = _mask(pred, valid) & _mask(gt, None)
    return np.linalg.norm(pred[m] - gt[m], axis=-1)
