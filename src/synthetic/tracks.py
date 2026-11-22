"""Synthetic pick-and-place object, hand and robot trajectories (world frame, z up, metres)."""

import numpy as np

from common.timebase import frame_timestamps
from common.transforms import make_se3, rot_axis_angle
from contracts.enums import EXPECTED_G1_DEX3_DOF

TABLE_Z_M = 0.0


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def pick_place_positions(
    n: int,
    start_xy: tuple[float, float] = (0.0, 0.0),
    end_xy: tuple[float, float] = (0.3, 0.1),
    half_height_m: float = 0.03,
    lift_m: float = 0.12,
) -> np.ndarray:
    """Object centre positions (n, 3): rest, lift, translate, lower, rest."""
    u = np.linspace(0.0, 1.0, n)
    lift = _smoothstep((u - 0.15) / 0.2) - _smoothstep((u - 0.65) / 0.2)
    move = _smoothstep((u - 0.35) / 0.3)
    xy = np.asarray(start_xy) + move[:, None] * (np.asarray(end_xy) - np.asarray(start_xy))
    z = TABLE_Z_M + half_height_m + lift_m * lift
    return np.column_stack([xy, z])


def object_tracks(n: int = 150, fps: float = 30.0, **kw) -> dict[str, np.ndarray]:
    """ObjectMotionBundle arrays for one object with a pick-and-place motion."""
    pos = pick_place_positions(n, **kw)
    T = np.tile(np.eye(4), (n, 1, 1, 1))
    T[:, 0, :3, 3] = pos
    return {
        "timestamps_s": frame_timestamps(n, fps),
        "T_world_object": T,
        "valid": np.ones((n, 1), bool),
        "visibility": np.ones((n, 1)),
    }


def hand_tracks(n: int = 150, fps: float = 30.0, **kw) -> dict[str, np.ndarray]:
    """HandTrackBundle arrays: right hand (index 1) follows the object, left hand is invalid."""
    pos = pick_place_positions(n, **kw)
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.04, 0.04, size=(21, 3))
    joints = np.zeros((n, 2, 21, 3))
    joints[:, 1] = pos[:, None, :] + offsets[None] + np.array([0.0, 0.0, 0.05])
    valid = np.zeros((n, 2), bool)
    valid[:, 1] = True
    return {
        "timestamps_s": frame_timestamps(n, fps),
        "frame_ids": np.arange(n, dtype=np.int64),
        "joints2d_px": np.zeros((n, 2, 21, 2)),
        "joints3d_est": joints,
        "valid": valid,
        "confidence": valid.astype(np.float64),
    }


def alignment_arrays(n: int = 150) -> dict[str, np.ndarray]:
    flags = np.zeros((n, 2), bool)
    flags[int(0.2 * n) : int(0.8 * n), 1] = True
    return {
        "T_sim_world_Rt": np.eye(4),
        "scale_sim_per_world": np.float64(1.0),
        "T_sim_robot_base": make_se3(rot_axis_angle([0, 0, 1], np.pi), [0.6, 0.0, 0.0]),
        "contact_flags": flags,
    }


def robot_trajectory(m: int = 100, fps: float = 30.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Smooth 43-DoF target trajectory with consistent velocities."""
    rng = np.random.default_rng(seed)
    t = frame_timestamps(m, fps)
    amp = rng.uniform(0.0, 0.3, size=EXPECTED_G1_DEX3_DOF)
    freq = rng.uniform(0.2, 0.6, size=EXPECTED_G1_DEX3_DOF)
    q = amp * np.sin(2 * np.pi * freq * t[:, None])
    qd = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t[:, None])
    wrist = np.tile(np.eye(4), (m, 2, 1, 1))
    wrist[:, :, :3, 3] = [0.4, 0.0, 0.2]
    return {
        "timestamps_s": t,
        "q_target_rad": q,
        "qvel_target_rad_s": qd,
        "wrist_T_sim": wrist,
        "base_T_sim": np.tile(np.eye(4), (m, 1, 1)),
    }
