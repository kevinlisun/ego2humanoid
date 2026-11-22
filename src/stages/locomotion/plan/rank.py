"""L.12: rank stances (hard constraints first, then a weighted cost) and emit a PoseGoal."""

from __future__ import annotations

import numpy as np

from contracts.locomanip import LocoManipFailure, ManipulationTask, PoseGoal
from robot.head_camera import HeadCamera
from stages.locomotion.plan.candidates import Stance, sample_candidates
from stages.locomotion.plan.ik import StanceIK
from stages.locomotion.plan.validators import forbidden_contacts
from stages.locomotion.plan.visibility import (
    MIN_VISIBLE_FRACTION,
    MIN_VISIBLE_FRACTION_GOAL,
    box_surface_points,
    visible_fraction,
)
from stages.locomotion.route import plan_route
from stages.locomotion.tabletop import Scenario

EXTENT = ((-1.0, 5.0), (-2.0, 2.0))
RES = 0.05
ROBOT_RADIUS_M = 0.22
W_VIS, W_IK, W_NAV, W_WAIST, W_YAW = 2.0, 5.0, 0.2, 0.3, 0.2


def plan_stance(
    sc: Scenario,
    task: ManipulationTask,
    side: str = "right",
    extra_stances=(),
    hand: str = "dex3",
    hand_table_ok=False,
    vis_min=None,
    arrival_slack_m=0.0,
    robot_radius_m=ROBOT_RADIUS_M,
):
    """Return (PoseGoal | None, failure | None). Hard constraints: IK, no forbidden contact, visibility."""
    model, data = sc.build(hand)
    cam, sik = HeadCamera.load(), StanceIK(side)
    grid = sc.occupancy(EXTENT, RES, robot_radius_m)
    obj0, half = sc.object_start_xyz(), sc.cfg["object"]["half_size"]
    goal = obj0.copy()
    goal[:2] = sc.goal_xy()
    pts = {"start": box_surface_points(obj0, half), "goal": box_surface_points(goal, half)}

    def T_obj(phase):
        T = np.eye(4)
        T[:3, 3] = goal if phase in ("place", "release") else obj0
        return T

    cands, rejected = sample_candidates(obj0[:2], grid, EXTENT, RES, focus_xy=0.5 * (obj0[:2] + goal[:2]))
    for st in extra_stances:  # e.g. stances around the one a retarget was solved for; same footprint rule
        row, col = int((st.y - EXTENT[1][0]) / RES), int((st.x - EXTENT[0][0]) / RES)
        if grid[row, col]:
            rejected["footprint_blocked"] += 1
        else:
            cands.append(st)
    vis_start_min, vis_goal_min = vis_min or (MIN_VISIBLE_FRACTION, MIN_VISIBLE_FRACTION_GOAL)
    rejected.update({"ik": 0, "contact": 0, "visibility": 0, "no_path": 0})
    best = None
    start = sc.cfg["robot_start_xyyaw"]
    for st in cands:
        res, ok, _ = sik.solve(st, T_obj, task.keyframes)
        if not ok:
            rejected["ik"] += 1
            continue
        if arrival_slack_m:  # the walking base tends to stop short of its goal; the arm must still reach
            a = np.arctan2(st.y - start[1], st.x - start[0])
            short = Stance(
                st.x - arrival_slack_m * np.cos(a),
                st.y - arrival_slack_m * np.sin(a),
                st.yaw,
                st.height_m,
                st.waist_pitch_rad,
            )
            if not sik.solve(short, T_obj, task.keyframes)[1]:
                rejected["ik"] += 1
                continue
        if any(
            forbidden_contacts(
                model, data, st, side, r.q_arm, task.object_name, k.phase != "pre_grasp", hand_table_ok
            )
            for r, k in zip(res, task.keyframes, strict=True)
        ):
            rejected["contact"] += 1
            continue
        vis_start = visible_fraction(model, data, cam, st, pts["start"])
        vis_goal = visible_fraction(model, data, cam, st, pts["goal"])
        vis = min(vis_start, vis_goal)
        if vis_start < vis_start_min or vis_goal < vis_goal_min:
            rejected["visibility"] += 1
            continue
        route = plan_route(grid, EXTENT, RES, start[:2], (st.x, st.y, st.yaw))
        if route is None:
            rejected["no_path"] += 1
            continue
        length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:], strict=False))
        focus = 0.5 * (obj0[:2] + goal[:2])
        facing = np.arctan2(focus[1] - st.y, focus[0] - st.x)
        cost = (
            W_VIS * (1 - vis)
            + W_IK * sum(r.pos_err_m for r in res)
            + W_NAV * length
            + W_WAIST * st.waist_pitch_rad
            + W_YAW * abs(st.yaw - facing)
        )
        if best is None or cost < best[0]:
            best = (cost, st, vis)
    if best is None:
        return None, LocoManipFailure.NO_FEASIBLE_STATIC_MANIPULATION_POSE, rejected
    cost, st, vis = best
    return (
        PoseGoal(
            base_xy_yaw=(st.x, st.y, st.yaw),
            pelvis_height_m=st.height_m,
            waist_rpy_rad=(0.0, st.waist_pitch_rad, 0.0),
            score=float(cost),
            visible_fraction=float(vis),
            rejected=rejected,
        ),
        None,
        rejected,
    )
