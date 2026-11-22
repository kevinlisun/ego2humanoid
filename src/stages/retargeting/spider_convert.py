"""Our bundles -> SPIDER processed-data layout (human ``mano`` keypoints + ``task_info.json``).

Layout: ``<dataset_dir>/processed/<dataset>/mano/<embodiment>/<task>/<data_id>/trajectory_keypoints.npz``.
Poses are 7-vectors ``[x, y, z, qw, qx, qy, qz]`` in the sim frame; fingertips are ordered
thumb, index, middle, ring, pinky and carry an identity orientation (landmarks give positions only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DATASET = "ego2humanoid"
TIP_LANDMARKS = (4, 8, 12, 16, 20)  # thumb..pinky tips in the 21-landmark layout
FAR_AWAY_M = 10.0  # parking spot for an absent object


def mat_to_pose7(T: np.ndarray) -> np.ndarray:
    """(...,4,4) -> (...,7) as [pos, quat wxyz]."""
    T = np.asarray(T, float)
    q = Rotation.from_matrix(T[..., :3, :3]).as_quat()  # xyzw
    return np.concatenate([T[..., :3, 3], q[..., [3, 0, 1, 2]]], axis=-1)


def _hand_arrays(wrist_T: np.ndarray, landmarks_sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrist = mat_to_pose7(wrist_T)
    tips = np.asarray(landmarks_sim, float)[:, TIP_LANDMARKS]
    ident = np.tile([1.0, 0, 0, 0], (*tips.shape[:2], 1))
    return wrist, np.concatenate([tips, ident], axis=-1)


def convert_clip(
    dataset_dir: str | Path,
    task: str,
    data_id: int,
    ref_dt_s: float,
    wrist_T_sim: dict[str, np.ndarray],
    landmarks_sim: dict[str, np.ndarray],
    object_T_sim: np.ndarray | None,
    object_side: str,
    contact: dict[str, np.ndarray] | None = None,
    object_mesh_dir: str | None = None,
    embodiment: str = "bimanual",
    table_top_z_m: float = 0.0,
) -> Path:
    """Write one clip. ``wrist_T_sim``/``landmarks_sim`` hold per-side (T,4,4)/(T,21,3) arrays.

    ``table_top_z_m`` is subtracted from every z (the runner adds it back).
    ``contact`` maps side to (T,5) per-fingertip flags (SPIDER's ``contact_<side>``); when omitted the
    contact arrays are not written and SPIDER treats every fingertip as always in contact.
    """
    sides = ("right", "left")
    shift = np.eye(4)  # SPIDER puts the floor at z = 0: move the table top there
    shift[2, 3] = -table_top_z_m
    wrist_T_sim = {k: shift @ np.asarray(v, float) for k, v in wrist_T_sim.items()}
    landmarks_sim = {k: np.asarray(v, float) - [0, 0, table_top_z_m] for k, v in landmarks_sim.items()}
    if object_T_sim is not None:
        object_T_sim = shift @ np.asarray(object_T_sim, float)
    if set(wrist_T_sim) != set(sides) or set(landmarks_sim) != set(sides):
        raise ValueError("both hands are required (the passive hand avoids arm collisions)")
    n = len(wrist_T_sim["right"])
    if object_side not in sides:
        raise ValueError(object_side)
    out = {}
    for s in sides:
        out[f"qpos_wrist_{s}"], out[f"qpos_finger_{s}"] = _hand_arrays(wrist_T_sim[s], landmarks_sim[s])
        parked = np.tile([FAR_AWAY_M, 0, 0, 1, 0, 0, 0], (n, 1)).astype(float)
        out[f"qpos_obj_{s}"] = parked
        if contact is not None:
            out[f"contact_{s}"] = np.asarray(contact.get(s, np.zeros((n, 5))), float)
            out[f"contact_pos_{s}"] = np.zeros((5, 3))
    if object_T_sim is not None:
        out[f"qpos_obj_{object_side}"] = mat_to_pose7(object_T_sim)

    task_dir = Path(dataset_dir) / "processed" / DATASET / "mano" / embodiment / task
    clip_dir = task_dir / str(data_id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    np.savez(clip_dir / "trajectory_keypoints.npz", **out)
    info = {
        "task": task,
        "dataset_name": DATASET,
        "robot_type": "mano",
        "hand_type": embodiment,
        "data_id": data_id,
        "right_object_mesh_dir": object_mesh_dir if object_side == "right" else None,
        "left_object_mesh_dir": object_mesh_dir if object_side == "left" else None,
        "ref_dt": ref_dt_s,
        "sim_dt": 0.01,
        "act_sim_dt": 0.005,
    }
    (task_dir / "task_info.json").write_text(json.dumps(info, indent=2))
    return clip_dir
