"""Simulation-side scene state: per-frame object pose, velocity, contacts and a settle report.

The state is read straight from MuJoCo (no vision-based tracking). ``record_scene_state`` steps a scene
that contains the robot at its home pose (gravity compensated, so it stays put) and the reconstructed
objects, and ``settle_report`` summarises whether every object rests stably on the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from stages.manipulation.mjcf_scene import SceneConfig, build_scene


@dataclass(frozen=True)
class SceneState:
    names: tuple[str, ...]
    time_s: np.ndarray  # (T,)
    pos_m: np.ndarray  # (T, N, 3)
    quat_wxyz: np.ndarray  # (T, N, 4)
    linvel_mps: np.ndarray  # (T, N, 3)
    angvel_rps: np.ndarray  # (T, N, 3)
    on_table: np.ndarray  # (T, N) bool: object touches the table top
    max_penetration_m: np.ndarray  # (T, N) deepest contact of the object (>= 0)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, names=np.array(self.names), **{k: getattr(self, k) for k in _ARRAYS})


_ARRAYS = ("time_s", "pos_m", "quat_wxyz", "linvel_mps", "angvel_rps", "on_table", "max_penetration_m")


def load_scene_state(path: str | Path) -> SceneState:
    z = np.load(path)
    return SceneState(names=tuple(str(n) for n in z["names"]), **{k: z[k] for k in _ARRAYS})


def record_scene_state(cfg: SceneConfig, duration_s: float = 1.0, fps: float = 30.0) -> SceneState:
    model = build_scene(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = tuple(o.name for o in cfg.objects)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    obj_of_geom = {g: i for i, b in enumerate(body_ids) for g in np.where(model.geom_bodyid == b)[0]}
    steps_per_frame = max(1, round(1.0 / fps / model.opt.timestep))
    n_frames = int(round(duration_s * fps)) + 1
    n = len(names)
    out = {
        "pos_m": np.zeros((n_frames, n, 3)),
        "quat_wxyz": np.zeros((n_frames, n, 4)),
        "linvel_mps": np.zeros((n_frames, n, 3)),
        "angvel_rps": np.zeros((n_frames, n, 3)),
        "on_table": np.zeros((n_frames, n), bool),
        "max_penetration_m": np.zeros((n_frames, n)),
    }
    vel = np.zeros(6)
    for f in range(n_frames):
        for i, b in enumerate(body_ids):
            out["pos_m"][f, i] = data.xpos[b]
            out["quat_wxyz"][f, i] = data.xquat[b]
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            out["angvel_rps"][f, i], out["linvel_mps"][f, i] = vel[:3], vel[3:]
        for c in data.contact[: data.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                i = obj_of_geom.get(g)
                if i is None:
                    continue
                out["max_penetration_m"][f, i] = max(out["max_penetration_m"][f, i], -c.dist)
                out["on_table"][f, i] |= other == table_geom
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
    return SceneState(names=names, time_s=np.arange(n_frames) / fps, **out)


def settle_report(
    state: SceneState,
    max_drift_m: float = 0.01,
    max_speed_mps: float = 0.02,
    max_penetration_m: float = 0.005,
) -> dict:
    """Per-object stability: drift from the initial pose, final speed, penetration and table support."""
    objects = {}
    for i, name in enumerate(state.names):
        drift = float(np.linalg.norm(state.pos_m[-1, i] - state.pos_m[0, i]))
        speed = float(np.linalg.norm(state.linvel_mps[-1, i]))
        pen = float(state.max_penetration_m[:, i].max())
        supported = bool(state.on_table[-1, i])
        objects[name] = {
            "drift_m": drift,
            "final_speed_mps": speed,
            "max_penetration_m": pen,
            "supported_on_table": supported,
            "stable": drift <= max_drift_m
            and speed <= max_speed_mps
            and pen <= max_penetration_m
            and supported,
        }
    return {"objects": objects, "all_stable": all(o["stable"] for o in objects.values())}
