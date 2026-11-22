"""Action-controlled rollout: joint targets in, achieved physical states out.

The only writes to simulator state happen at reset (robot initial pose, object placement).
After reset the world advances exclusively through ``mj_step`` under actuator commands.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from common.transforms import make_se3, rotation_angle
from robot.collision import arms_touching, geom_sides
from stages.manipulation.controller import PositionController
from stages.manipulation.events import EventThresholds
from stages.manipulation.guard import StateInjectionGuard
from stages.manipulation.mjcf_scene import BoxObject, SceneConfig, build_scene
from stages.manipulation.sim_tactile import SimTactile

WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


@dataclass(frozen=True)
class Perturbation:
    """Seeded reset-time randomisation (domain randomisation of the estimated scene)."""

    object_xy_m: float = 0.005
    friction_scale: float = 0.2
    mass_scale: float = 0.2

    def apply(self, cfg: SceneConfig, seed: int) -> SceneConfig:
        if seed == 0:
            return cfg
        rng = np.random.default_rng(seed)
        objs = []
        for o in cfg.objects:
            if o.static:  # a welded receptacle is part of the scene, not a randomised prop
                objs.append(o)
                continue
            dxy = rng.uniform(-self.object_xy_m, self.object_xy_m, 2)
            fr = 1.0 + rng.uniform(-self.friction_scale, self.friction_scale)
            ms = 1.0 + rng.uniform(-self.mass_scale, self.mass_scale)
            objs.append(
                replace(
                    o,
                    pos_m=(o.pos_m[0] + dxy[0], o.pos_m[1] + dxy[1], o.pos_m[2]),
                    friction=(o.friction[0] * fr, o.friction[1], o.friction[2]),
                    mass_kg=o.mass_kg * ms,
                )
            )
        return replace(cfg, objects=tuple(objs))


@dataclass
class RolloutResult:
    arrays: dict[str, np.ndarray]
    physics_step_count: int
    seed: int
    contact_steps_with_object: int
    state_injection_after_reset: bool
    arm_collision_steps: int = 0  # steps where a left-arm/hand geom touches a right-arm/hand geom
    tactile: SimTactile | None = None  # filled only when run_rollout got a ``tactile`` factory


def run_rollout(
    cfg: SceneConfig,
    timestamps_s: np.ndarray,
    q_target_rad: np.ndarray,
    seed: int = 0,
    perturbation: Perturbation | None = None,
    record_rate_hz: float = 50.0,
    settle_tail_s: float = 0.5,
    settle_max_s: float = 1.5,
    on_record: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    tactile: Callable[[mujoco.MjModel], SimTactile] | None = None,
) -> RolloutResult:
    """Replay ``q_target_rad`` (M, DoF) given at ``timestamps_s`` (M,) through PD actuators.

    The rollout runs ``settle_tail_s`` past the last target and then on, up to ``settle_max_s`` more,
    until every object has been at rest for the settle window (clip 101: a dropped roll was still
    tumbling inside its case when the episode ended, and was scored NOT_SETTLED).

    ``DoF`` is the actuator count of the compiled robot (43 for Dex3, 73 for the Sharpa hands);
    scene objects are free bodies and add no actuators.
    """
    cfg = (perturbation or Perturbation()).apply(cfg, seed)
    model = build_scene(cfg)
    n_act = model.nu
    if q_target_rad.shape != (len(timestamps_s), n_act):
        raise ValueError(f"q_target_rad must have shape (M, {n_act})")
    data = mujoco.MjData(model)
    ctrl = PositionController(model, timestamps_s, q_target_rad)
    # Joint order is the actuator order; qpos/qvel addresses follow the kinematic tree instead.
    joint_ids = model.actuator_trnid[:n_act, 0]
    qadr, dofadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]

    data.qpos[qadr] = ctrl.q_target(timestamps_s[0])  # reset-time pose only
    mujoco.mj_forward(model, data)
    obj_slices = _object_qpos_slices(model, cfg.objects)
    guard = StateInjectionGuard(obj_slices)
    wrist_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in WRIST_BODIES]
    obj_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o.name) for o in cfg.objects]
    # Contact statistics are about the manipulated objects; touching the receptacle is not a grasp.
    obj_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{o.name}_geom")
        for o in cfg.objects
        if not o.static
    }
    # Every collision geom of each object (convex pieces included), for the per-record contact flags.
    geom_obj = {
        g: obj_ids.index(int(model.geom_bodyid[g]))
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in obj_ids
    }
    non_robot = {"world", "table", *(o.name for o in cfg.objects)}
    robot_geoms = {
        g
        for g in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) not in non_robot
    }

    side_of = geom_sides(model, non_robot)
    t_end = float(timestamps_s[-1] - timestamps_s[0]) + settle_tail_s
    n_steps = int(round(t_end / model.opt.timestep))
    rec_every = max(1, int(round(1.0 / (record_rate_hz * model.opt.timestep))))
    rec: dict[str, list] = {k: [] for k in ("t", "qc", "qa", "dqa", "wrist", "obj", "touch")}
    contact_steps = 0
    arm_collision_steps = 0
    sim_tactile = tactile(model) if tactile is not None else None
    guard.snapshot(data)
    th = EventThresholds()
    n_settle = max(2, int(round(th.settle_window_s * record_rate_hz)) + 1)
    n_max = n_steps + int(round(settle_max_s / model.opt.timestep))
    for step in range(n_max):
        t = timestamps_s[0] + data.time
        data.ctrl[:n_act] = ctrl.command(t)
        guard.check(data)
        mujoco.mj_step(model, data)
        guard.snapshot(data)
        if _touching(data, obj_geoms, robot_geoms):
            contact_steps += 1
        if arms_touching(data, side_of):
            arm_collision_steps += 1
        if step % rec_every == 0:
            rec["t"].append(timestamps_s[0] + data.time)
            rec["qc"].append(data.ctrl[:n_act].copy())
            rec["qa"].append(data.qpos[qadr].copy())
            rec["dqa"].append(data.qvel[dofadr].copy())
            rec["wrist"].append(_poses(data, wrist_ids))
            rec["obj"].append(_poses(data, obj_ids))
            rec["touch"].append(_robot_touch(data, geom_obj, robot_geoms, len(obj_ids)))
            if sim_tactile is not None:
                sim_tactile.update(model, data)
            if on_record is not None:
                on_record(model, data)
            if step >= n_steps and _at_rest(rec["obj"][-n_settle:], n_settle, record_rate_hz, th):
                break
    n_steps = step + 1
    arrays = {
        "timestamps_s": np.asarray(rec["t"]) - timestamps_s[0],
        "q_command_rad": np.asarray(rec["qc"]),
        "q_actual_rad": np.asarray(rec["qa"]),
        "dq_actual_rad_s": np.asarray(rec["dqa"]),
        "wrist_actual_T_sim": np.asarray(rec["wrist"]),
        "object_T_sim": np.asarray(rec["obj"]),
        "object_robot_contact": np.asarray(rec["touch"], bool).reshape(len(rec["t"]), len(obj_ids)),
    }
    return RolloutResult(
        arrays, n_steps, seed, contact_steps, guard.violations > 0, arm_collision_steps, sim_tactile
    )


def _at_rest(poses: list[np.ndarray], n: int, rate_hz: float, th: EventThresholds) -> bool:
    """Whether no object moved faster than the settle thresholds over the last ``n`` records."""
    if len(poses) < n:
        return False
    T = np.asarray(poses)
    if T.shape[1] == 0:  # no object in the scene
        return True
    speed = np.linalg.norm(np.diff(T[:, :, :3, 3], axis=0), axis=2) * rate_hz
    rel = np.swapaxes(T[:-1, :, :3, :3], 2, 3) @ T[1:, :, :3, :3]
    ang = rotation_angle(rel) * rate_hz
    return bool(speed.max() <= th.settle_speed_m_s and ang.max() <= th.settle_ang_speed_rad_s)


def _object_qpos_slices(model: mujoco.MjModel, objects: tuple[BoxObject, ...]) -> list[slice]:
    out = []
    for o in objects:
        if o.static:  # welded to the world, so it has no free joint to guard
            continue
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{o.name}_free")
        adr = model.jnt_qposadr[j]
        out.append(slice(adr, adr + 7))
    return out


def _poses(data: mujoco.MjData, body_ids: list[int]) -> np.ndarray:
    out = np.tile(np.eye(4), (len(body_ids), 1, 1))
    for i, b in enumerate(body_ids):
        out[i] = make_se3(data.xmat[b].reshape(3, 3), data.xpos[b])
    return out


def _touching(data: mujoco.MjData, obj_geoms: set[int], robot_geoms: set[int]) -> bool:
    """True if any robot geom is in contact with an object geom (table contacts excluded)."""
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        if (g1 in obj_geoms and g2 in robot_geoms) or (g2 in obj_geoms and g1 in robot_geoms):
            return True
    return False


def _robot_touch(
    data: mujoco.MjData, geom_obj: dict[int, int], robot_geoms: set[int], n_obj: int
) -> np.ndarray:
    """(n_obj,) bool: whether any robot geom touches each object right now."""
    out = np.zeros(n_obj, bool)
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        if g1 in geom_obj and g2 in robot_geoms:
            out[geom_obj[g1]] = True
        elif g2 in geom_obj and g1 in robot_geoms:
            out[geom_obj[g2]] = True
    return out
