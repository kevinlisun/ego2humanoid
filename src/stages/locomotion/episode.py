"""Approach episode: plan stance -> walk -> raise arm -> final alignment -> drive arm through keyframes.

The arm is raised before the final alignment because raising it while standing makes the Balance policy
step and yaw by ~0.6 rad (a raised carry pose first keeps the long Sharpa hand above the table edge).
Keyframes are re-solved from the *measured* arrival state (L.15). Fingers follow the keyframe phases:
fist while walking, open at the pre-grasp, pinch (touch -> squeeze) at the grasp, open at the release.
"""

from __future__ import annotations

import mujoco
import numpy as np

from contracts.locomanip import ArrivalReport as ArrivalContract
from contracts.locomanip import LocoManipFailure, ManipulationTask
from robot.dwbc import BODY_JOINT_NAMES, NUM_POLICY_JOINTS, DwbcRobot
from robot.head_camera import HeadCamera
from robot.kinematics import ARM_JOINTS, WRIST_BODY
from stages.locomotion.executor import base_yaw, goto
from stages.locomotion.plan.candidates import Stance
from stages.locomotion.plan.ik import StanceIK
from stages.locomotion.plan.rank import EXTENT, RES, ROBOT_RADIUS_M, plan_stance
from stages.locomotion.plan.validators import tucked_arm
from stages.locomotion.plan.visibility import box_surface_points, visible_fraction
from stages.locomotion.route import plan_route
from stages.locomotion.tabletop import Scenario
from stages.locomotion.video import VideoRecorder
from stages.retargeting.sharpa_pinch import PinchPoses

ARM_NAMES = BODY_JOINT_NAMES[NUM_POLICY_JOINTS:]
SETTLE_S = 2.0
KEYFRAME_MOVE_S = 2.0
ARM_RAISE_S = 3.0
KEYFRAME_HOLD_S = 0.5
FINGER_MOVE_S = 1.0
SERVO_ITERS = 2
SERVO_CLAMP_M = 0.02
CARRY_ARM_Q = np.array(
    [-0.1, -0.25, 0.0, 1.5, 0.0, 0.0, 0.0]
)  # hand in front of the chest, above the table edge


def _arm_slice(side: str) -> np.ndarray:
    return np.array([ARM_NAMES.index(n) for n in ARM_JOINTS[side]])


def _smooth(s: float) -> float:
    return s * s * (3 - 2 * s)


def _run(robot: DwbcRobot, secs: float) -> bool:
    m, d = robot.model, robot.data
    for _ in range(int(secs / m.opt.timestep)):
        robot.step()
        if d.qpos[2] < 0.4:
            return False
    return True


def _move_arm(robot: DwbcRobot, idx: np.ndarray, q_goal: np.ndarray, secs: float) -> bool:
    m, d = robot.model, robot.data
    q0 = robot.arm_q_target[idx].copy()
    n = int(secs / m.opt.timestep)
    for i in range(n):
        robot.arm_q_target[idx] = q0 + _smooth((i + 1) / n) * (q_goal - q0)
        robot.step()
        if d.qpos[2] < 0.4:
            return False
    return True


def _object_poses(sc: Scenario):
    obj0 = sc.object_start_xyz()
    goal_obj = obj0.copy()
    goal_obj[:2] = sc.goal_xy()

    def T_obj(phase):
        T = np.eye(4)
        T[:3, 3] = goal_obj if phase in ("place", "release") else obj0
        return T

    return obj0, T_obj


def run_episode(
    sc: Scenario,
    task: ManipulationTask,
    side: str = "right",
    hand: str = "sharpa",
    video: str | None = None,
    poses=None,
    extra_stances=(),
    plan_kw: dict | None = None,
) -> dict:
    """Plan, walk, grasp and place; ``video`` is an optional mp4 path for a third-person recording.

    ``poses`` overrides the finger stages (``carry``/``open``/``touch``/``squeeze``), e.g. with the
    fingers retargeted from a clip; the default is the stored power-wrap blend.
    """
    recorder = VideoRecorder(video) if video else None
    try:
        return _run_episode(sc, task, side, hand, recorder, poses, extra_stances, plan_kw or {})
    finally:
        if recorder:
            recorder.close()


def _run_episode(sc, task: ManipulationTask, side: str, hand: str, recorder, poses, extra, plan_kw) -> dict:
    goal, failure, rejected = plan_stance(sc, task, side, extra, **plan_kw)
    if goal is None:
        return {"ok": False, "failure": failure, "rejected": rejected}
    m, d = sc.build(hand)
    robot = DwbcRobot(m, d)
    if recorder:
        recorder.attach(m, d)
        robot.on_step = recorder.maybe_capture
    if poses is None and hand == "sharpa":
        poses = PinchPoses.load()
    robot.arm_gravity_comp = poses is not None  # the long Sharpa hand sags 5-10 cm under pure PD
    if poses is not None:
        for sd in ("left", "right"):
            robot.set_fingers(poses.pose("carry", sd))
    idx = _arm_slice(side)
    start = tuple(sc.cfg["robot_start_xyyaw"])
    robot.reset_pose(start, pelvis_z=0.79)
    for sd in ("left", "right"):
        robot.arm_q_target[_arm_slice(sd)] = tucked_arm(sd)
    grid = sc.occupancy(EXTENT, RES, plan_kw.get("robot_radius_m", ROBOT_RADIUS_M))
    route = plan_route(grid, EXTENT, RES, start[:2], goal.base_xy_yaw)
    if route is None:
        return {"ok": False, "failure": LocoManipFailure.NO_NAVIGATION_PATH}
    fall = {"ok": False, "failure": LocoManipFailure.LOCOMOTION_FALL}
    if not _run(robot, 1.0):
        return fall

    obj0, T_obj = _object_poses(sc)
    sik = StanceIK(side)
    planned = Stance(*goal.base_xy_yaw, goal.pelvis_height_m, goal.waist_rpy_rad[1])
    plan_res, ok, why = sik.solve(planned, T_obj, task.keyframes)
    if not ok:
        return {"ok": False, "failure": LocoManipFailure.ARM_IK_INFEASIBLE, "reason": why}

    shuffle = bool(plan_kw.get("arrival_slack_m"))  # clip mode: avoid U-turns when the goal is overshot
    rep = goto(robot, route[:-1], timeout_s=90.0, shuffle=shuffle) if len(route) > 1 else None
    if rep is not None and rep.fell:
        return fall
    robot.cmd.vel[:] = 0
    robot.cmd.height_m = goal.pelvis_height_m
    robot.cmd.rpy_rad[:] = goal.waist_rpy_rad
    carry_q = CARRY_ARM_Q * np.array([1, 1 if side == "right" else -1, 1, 1, 1, 1, 1])
    if not (
        _move_arm(robot, idx, carry_q, ARM_RAISE_S)
        and _move_arm(robot, idx, plan_res[0].q_arm, ARM_RAISE_S)
        and _run(robot, SETTLE_S)
    ):
        return fall
    rep = goto(robot, route[-1:], timeout_s=60.0, shuffle=shuffle)
    arrival = ArrivalContract(
        arrived=rep.arrived,
        pos_err_m=rep.pos_err_m,
        yaw_err_rad=rep.yaw_err_rad,
        time_s=rep.time_s,
        failure=None
        if rep.arrived
        else (LocoManipFailure.LOCOMOTION_FALL if rep.fell else LocoManipFailure.LOCOMOTION_TIMEOUT),
    )
    if not rep.arrived:
        return {"ok": False, "failure": arrival.failure, "arrival": arrival}
    robot.cmd.vel[:] = 0
    if not _run(robot, SETTLE_S):
        return fall

    waist = float(d.qpos[m.joint("waist_pitch_joint").qposadr[0]])
    measured = Stance(float(d.qpos[0]), float(d.qpos[1]), base_yaw(d.qpos), float(d.qpos[2]), waist)
    cam = HeadCamera.load()
    vis = visible_fraction(m, d, cam, measured, box_surface_points(obj0, sc.cfg["object"]["half_size"]))
    mujoco.mj_forward(m, d)  # visible_fraction re-poses the model; restore the simulated state
    res, ok, why = sik.solve(measured, T_obj, task.keyframes, limit_margin=0.0)
    if not ok:
        errs = [(k.phase, round(k.pos_err_m, 3), round(k.rot_err_rad, 2)) for k in res]
        return {
            "ok": False,
            "failure": LocoManipFailure.ARM_IK_INFEASIBLE,
            "reason": why,
            "arrival": arrival,
            "measured_stance": measured,
            "planned_stance": planned,
            "ik_errors": errs,
        }
    return _drive_arm(robot, sik, task, T_obj, side, arrival, vis, measured, poses)


def _finger_script(poses: PinchPoses | None, side: str, phase: str, k: int, n: int) -> list[str]:
    """Hand stages to run after arriving at keyframe ``k`` (of ``n``) of a pick-and-place."""
    if poses is None:
        return []
    if phase == "pre_grasp" and k == 0:
        return ["open"]
    return {"grasp": ["touch", "squeeze"], "place": ["open"]}.get(phase, [])


def _move_fingers(robot: DwbcRobot, frm: dict, to: dict, secs: float) -> None:
    n = int(secs / robot.model.opt.timestep)
    for i in range(n):
        s = _smooth((i + 1) / n)
        robot.set_fingers({k: frm[k] + s * (to[k] - frm[k]) for k in to})
        robot.step()


def _servo_keyframe(robot, sik, measured, T_obj, kf, side, corr, kind) -> tuple[np.ndarray, np.ndarray]:
    """Solve a keyframe with the running world-frame correction; returns (q_arm, world wrist target)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(kf.T_object_hand)[:3, :3]
    T[:3, 3] = np.asarray(kf.T_object_hand)[:3, 3] + corr
    shifted = kf.model_copy(update={"T_object_hand": T.tolist()})
    res, _, _ = sik.solve(measured, T_obj, [shifted], limit_margin=0.0)
    target = (T_obj(kf.phase) @ np.asarray(kf.T_object_hand))[:3, 3]
    return res[0].q_arm, target


def _drive_arm(robot, sik, task, T_obj, side, arrival, vis, measured, poses) -> dict:
    m, d = robot.model, robot.data
    idx = _arm_slice(side)
    min_z, max_tilt = float(d.qpos[2]), 0.0
    wrist_id = m.body(WRIST_BODY[side]).id
    obj_adr = m.joint(f"{task.object_name}_free").qposadr[0]
    obj_body = m.body(task.object_name).id
    track_err, corr = [], np.zeros(3)
    z0 = max_obj_z = float(d.qpos[obj_adr + 2])
    cur = poses.pose("carry", side) if poses is not None else None
    log = []
    nk = len(task.keyframes)
    for k, kf in enumerate(task.keyframes):
        q_goal, target = _servo_keyframe(robot, sik, measured, T_obj, kf, side, corr, None)
        for it in range(SERVO_ITERS + 1):
            for dur, hold in ((KEYFRAME_MOVE_S if it == 0 else 0.6, False), (KEYFRAME_HOLD_S, True)):
                q_prev = robot.arm_q_target[idx].copy()
                n = int(dur / m.opt.timestep)
                for i in range(n):
                    s = 1.0 if hold else _smooth((i + 1) / n)
                    robot.arm_q_target[idx] = q_prev + s * (q_goal - q_prev)
                    robot.step()
                    min_z = min(min_z, float(d.qpos[2]))
                    max_obj_z = max(max_obj_z, float(d.qpos[obj_adr + 2]))
                    tilt = np.arccos(np.clip(1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1, 1))
                    max_tilt = max(max_tilt, float(tilt))
                    if d.qpos[2] < 0.4:
                        return {"ok": False, "failure": LocoManipFailure.LOCOMOTION_FALL, "arrival": arrival}
            err = target - d.xpos[wrist_id]
            if it == SERVO_ITERS or np.linalg.norm(err) < 0.004:
                break
            corr = np.clip(corr + err, -SERVO_CLAMP_M, SERVO_CLAMP_M)
            q_goal, _ = _servo_keyframe(robot, sik, measured, T_obj, kf, side, corr, None)
        track_err.append(float(np.linalg.norm(target - d.xpos[wrist_id])))
        for stage in _finger_script(poses, side, kf.phase, k if kf.phase == "pre_grasp" else 1, nk):
            to = poses.pose(stage, side)
            _move_fingers(robot, cur, to, FINGER_MOVE_S)
            cur = to
        obj_xyz = d.qpos[obj_adr : obj_adr + 3]
        touching = sorted(
            {
                m.geom(c.geom1 if m.geom_bodyid[c.geom2] == obj_body else c.geom2).name
                for c in d.contact[: d.ncon]
                if obj_body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
            }
        )
        log.append(
            (
                kf.phase,
                np.round(obj_xyz, 3).tolist(),
                np.round(d.xpos[wrist_id] - obj_xyz, 3).tolist(),
                touching,
            )
        )
    return {
        "ok": True,
        "failure": None,
        "arrival": arrival,
        "visible_fraction_measured": vis,
        "min_pelvis_z": min_z,
        "max_tilt_rad": max_tilt,
        "measured_stance": measured,
        "wrist_track_err_m": [round(e, 4) for e in track_err],
        "object_xyz_after_keyframe": log,
        "object_final_xyz": d.qpos[obj_adr : obj_adr + 3].copy(),
        "object_max_lift_m": max_obj_z - z0,
    }
