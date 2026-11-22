"""RGB-estimated hands + RGB-estimated gravity -> kinematic retarget -> MuJoCo PD tracking report.

Inputs in the clip dir: hands_fah_f<focal>.npz, intrinsics_est.json, gravity_est.json (no GT used).
Output: ``retarget_rgb.json`` and ``retarget_rgb.npz`` (timestamps_s, q_target_rad).
"""

import dataclasses
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from review_render import render_review

from contracts.arrays import validate_npz
from contracts.sim_params import SceneParams
from robot.collision import ArmCollisionChecker
from skills.params.resolve import params_hash
from skills.params.scene_overrides import apply_scene_overrides
from skills.reachability_check import window_unreachable_fraction
from stages.locomotion.base_placement import PlaceGoal
from stages.manipulation.bundle import save_bundle
from stages.manipulation.events import detect_pick_place
from stages.manipulation.mjcf_scene import (
    D435_IMAGE_HW,
    BoxObject,
    ObjectContact,
    SceneConfig,
    build_scene,
)
from stages.manipulation.rollout import run_rollout
from stages.manipulation.sim_eval import SimCase, start_search
from stages.reconstruction.align.camera_fit import human_uv
from stages.reconstruction.align.table_fit import (
    arm_penetration,
    fit_table_height,
    near_edge_clearance,
    table_penetration,
)
from stages.reconstruction.hands.backend import artefact_names, load_hands_config
from stages.reconstruction.hands.smooth import clean_landmarks
from stages.reconstruction.objects import place_target as place_target_mod
from stages.reconstruction.objects.fuse import hand_depth_scale, select_held_object
from stages.reconstruction.objects.grasp_proxy import detect_grasp_proxy
from stages.reconstruction.objects.place_target import (
    place_target_for,
    rebase,
)
from stages.reconstruction.objects.prompt import object_prompt, place_target_prompt, source_prompt
from stages.reconstruction.scene.inventory import (
    build_scene_inventory,
    depth_ratio,
    disturbance,
    load_inventory,
    scene_extent_xy,
)
from stages.retargeting.clip import retarget_clip
from stages.retargeting.contact import contact_flags, write_alignment_bundle
from stages.retargeting.grip import hold_pinch_stiffness, required_grip_force_n, squeeze_for_force
from stages.retargeting.grip_timing import refine_grip_window
from stages.retargeting.hand_eye import (
    align_pinch,
    blend_corrections,
    limit_carry_acceleration,
    phase_windows,
    pinch_track,
    release_motion,
    settle_approach,
)
from stages.retargeting.held_mesh import build_held_mesh
from stages.retargeting.kinematic import (
    commanded_pinch_axis_xy,
    landmarks_camera_to_sim,
)
from stages.retargeting.script_helpers import (
    object_points,
    object_supports,
    placement_retarget,
    resolve_place_target,
)
from stages.retargeting.script_params import (
    ADJUST_STANCE,
    APPROACH_FRAMES,
    BARE_PLACE_LIFT_M,
    CARRY_ACCEL_M_S2,
    CLOSE_LEAD_FRAMES,
    CLOSE_RAMP_FRAMES,
    EGO_PANEL,
    GRASP_AXIS_FRAMES,
    GRIP_MIN_N,
    GRIP_REACH_TOL_M,
    HELD_CLEARANCE_M,
    HELD_MARGIN_M,
    HOVER_CLEARANCE_M,
    MAX_HOVER_RISE_M,
    MAX_OFFSET_M,
    MIN_PICK_RISE_M,
    MIN_SCORE,
    OBSTACLE_MARGIN_M,
    OFFSETS_M,
    OPEN_MARGIN_M,
    OPEN_PAD_M,
    PAD_REACH_M,
    PASSIVE_MIN_VALID,
    PINCH_AXIS_TOL_DEG,
    PLACE_CLEARANCE_M,
    PLACE_DWELL_FRAMES,
    PLACE_LEAD_FRAMES,
    PLACE_REACH_WEIGHT_M,
    PLACE_RETRY_BUDGET_S,
    PLACE_STANCE_SEARCH,
    PLACE_XY_LEAD_FRAMES,
    REBASE_OUTSIDE_M,
    RECEPTACLE_OPENING_MARGIN_M,
    RELEASE_OPEN_FRAMES,
    RETREAT_BLEND_FRAMES,
    RETREAT_FRAMES,
    RETREAT_HEIGHT_M,
    SCENE_OBJECTS,
    SCENE_TABLE_MARGIN_M,
    SIM_PARAMS,
    SLAB_MAX_WIDTH_M,
    SQUEEZE_M,
    START_SEARCH_M,
    TABLE_INSET_M,
    TABLE_MARGIN_M,
    TABLE_NEAR_EDGE_M,
    TABLE_Z_DEFAULT_M,
    TIP_FLOOR_M,
)
from stages.retargeting.support_clearance import Support, held_object_lift, lift_above_support

_T_START = _T_LAST = time.perf_counter()


def _lap(stage: str) -> None:
    """Wall-clock log of the stage that just finished (``[time]`` lines; the run is long)."""
    global _T_LAST
    now = time.perf_counter()
    print(f"[time] {stage}: {now - _T_LAST:.1f} s (total {now - _T_START:.1f} s)", flush=True)
    _T_LAST = now


clip = Path(sys.argv[1])


# Optional second argument "--tactile=<none|egotac>" overrides configs/hands.yaml and $E2H_TACTILE_BACKEND.
tactile_arg = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--tactile=")), None)
hands_cfg = load_hands_config(tactile_override=tactile_arg)
f = round(json.loads((clip / "intrinsics_est.json").read_text())["focal_px"])
names = artefact_names(hands_cfg, f)
est = np.load(clip / names["pose"])
if os.environ.get("E2H_EGO_POSES") == "1":  # walking clip: express hands in the frame-0 camera frame
    from stages.reconstruction.align.ego_vo import points_to_world, smooth_poses

    est = dict(est)
    cam_poses = smooth_poses(np.load(clip / "camera_poses.npz")["poses"])
    m = min(len(cam_poses), len(est["valid"]))
    est["joints3d"] = np.concatenate(
        [points_to_world(est["joints3d"][:m], cam_poses[:m]), est["joints3d"][m:]]
    )
up = np.array(json.loads((clip / "gravity_est.json").read_text())["up_in_camera"])
n = len(est["valid"])
t = np.arange(n) / 30.0
travel = [
    np.linalg.norm(np.diff(est["joints3d"][est["valid"][:, i], i, 0], axis=0), axis=1).sum()
    if est["valid"][:, i].mean() > 0.5
    else 0.0
    for i in range(2)
]
HAND = os.environ.get("HAND", "dex3")  # dex3 | sharpa (five-finger tactile hand)
# keep the Dex3 baseline artefacts next to the new ones; E2H_OUT_TAG (e.g. "_bt") tags one experiment's outputs  # noqa: E501
OUT_TAG = os.environ.get("E2H_OUT_TAG", "")
SUFFIX = ("" if HAND == "dex3" else f"_{HAND}") + os.environ.get("E2H_OUT_TAG", "")
RENDER_FPS = int(os.environ.get("RENDER_FPS", "10"))
blender_grid = None  # (left, trajectory) frames kept until the report exists; Blender renders after
side_i = int(np.argmax(travel))  # active hand = the one that moves most
side = ("left", "right")[side_i]
ok = est["valid"][:, side_i] & (est["score"][:, side_i] > MIN_SCORE)  # low-score frames are unreliable
first, last = np.where(ok)[0][[0, -1]]  # trim to the span with reliable detections
sl = slice(first, last + 1)
lm, kept = clean_landmarks(est["joints3d"][sl, side_i], ok[sl])
t = t[sl]
other_i = 1 - side_i
ok_p = est["valid"][:, other_i] & (est["score"][:, other_i] > MIN_SCORE)
passive = None  # the other hand is retargeted too, otherwise its arm idles in the workspace and collides
if ok_p[sl].mean() >= PASSIVE_MIN_VALID:
    passive, _ = clean_landmarks(est["joints3d"][sl, other_i], ok_p[sl])
# Where the human pinch sits in the source image: the robot head camera is tilted to reproduce it.
pinch_uv_ref = human_uv(0.5 * (lm[:, 4] + lm[:, 8]), float(f), tuple(est["image_wh"]))
# A clip with its own scene inventory simulates every tabletop object from it; the reconstructed SCENE_DIR
# scene would only add the receptacle (and duplicate objects in the render), so it is ignored then.
if SCENE_OBJECTS and load_inventory(clip) is not None:
    os.environ.pop("SCENE_DIR", None)
# E2H_PLACE_GOAL: unset on the first pass; a JSON PlaceGoal on the second, which re-scores the standing
# spot when the first found the receptacle out of reach; "off" on the last, which restores the first
# pass's placement when the second did not bring the full place correction within reach.
_goal_env = os.environ.get("E2H_PLACE_GOAL")
place_target_mod.OPENING_MARGIN_M = RECEPTACLE_OPENING_MARGIN_M
# E2H_PLACE_RETRY carries the wall-clock start and the per-pass results over the re-execs, so the retry
# budget (trajectory.place_retry_budget_s) and the "never keep a worse pass" check see all passes.
retry_log = json.loads(os.environ.get("E2H_PLACE_RETRY") or "null") or {"t0": time.time(), "passes": []}
retry_outcome = retry_log.get("outcome", "not_needed")
place_goal = PlaceGoal(**json.loads(_goal_env)) if _goal_env not in (None, "off") else None
placement_cache = clip / ".placement_first"
if _goal_env == "off":
    os.environ["E2H_RETARGET_CACHE"] = str(placement_cache)
r, stance_adj = placement_retarget(
    clip, HAND, lm, t, up, side, passive, pinch_uv_ref, adjust=ADJUST_STANCE, place_goal=place_goal
)
r_first = r
_lap("load + placement retarget")
lm_sim = landmarks_camera_to_sim(lm, r.placement.T_sim_camera)
proxy = detect_grasp_proxy(lm_sim, t)
base = np.eye(4)
base[2, 3] = 0.793
# The surface the hand works on. Its height is nominal until a held object is identified, when it is
# raised to the height at which the hand actually grips (see below). Its footprint has to cover the
# whole retargeted reach, otherwise an object that is lifted and carried correctly drops off the edge
# and the episode is scored NOT_PLACED (measured: three clips ended 0.82 m below the table).
hand_xy = lm_sim[:, :, :2].reshape(-1, 2)
lo = np.minimum(hand_xy.min(0) - TABLE_MARGIN_M, (TABLE_NEAR_EDGE_M, -0.5))
hi = np.maximum(hand_xy.max(0) + TABLE_MARGIN_M, (0.8, 0.5))
lo[0] = max(lo[0], TABLE_NEAR_EDGE_M)  # keep the table clear of the robot's own body
table_kw = dict(
    table_top_z_m=TABLE_Z_DEFAULT_M,
    table_center_xy_m=tuple((lo + hi) / 2),
    table_half_size_xy_m=tuple((hi - lo) / 2),
)
res = run_rollout(
    SceneConfig.from_params(SIM_PARAMS, T_sim_robot_base=base, hand=HAND, **table_kw),
    r.timestamps_s,
    r.q_target_rad,
)
a = res.arrays
m = min(len(a["q_actual_rad"]), len(r.q_target_rad))
err = np.abs(a["q_actual_rad"][:m] - a["q_command_rad"][:m])
objects_report = None
if proxy is not None:
    box = BoxObject("proxy", tuple(proxy.half_size_m), tuple(proxy.center_m))
    ores = run_rollout(
        SceneConfig.from_params(SIM_PARAMS, T_sim_robot_base=base, objects=(box,), hand=HAND, **table_kw),
        r.timestamps_s,
        r.q_target_rad,
    )
    ev = detect_pick_place(ores.arrays["timestamps_s"], ores.arrays["object_T_sim"][:, 0])
    objects_report = ev.as_dict()
    objects_report["contact_steps"] = int(ores.contact_steps_with_object)
    w = ores.arrays["wrist_actual_T_sim"][:, 0 if side == "left" else 1, :3, 3]
    objects_report["min_wrist_obj_dist_m"] = float(
        np.linalg.norm(w - ores.arrays["object_T_sim"][:, 0, :3, 3], axis=1).min()
    )
perceived = None
obj_file = clip / "objects_sam3.npz"
if obj_file.exists():
    od = np.load(obj_file)
    fi = od["frame_idx"]
    valid = ok[fi] if fi.max() < len(ok) else np.zeros(len(fi), bool)
    scale = hand_depth_scale(
        od["depth_small"][valid],
        est["joints2d"][fi[valid], side_i, 0],
        est["joints3d"][fi[valid], side_i, 0, 2],
        tuple(est["image_wh"]),
    )
    cand_sim = landmarks_camera_to_sim((od["centroid_cam"] * scale).reshape(-1, 3), r.placement.T_sim_camera)
    cand_sim = cand_sim.reshape(od["centroid_cam"].shape)
    in_span = (fi >= first) & (fi <= last) & valid
    k = np.clip(fi - first, 0, len(t) - 1)
    held = select_held_object(
        cand_sim, od["extent_cam"] * scale, 0.5 * (lm_sim[k, 4] + lm_sim[k, 8]), in_span, 0.80
    )
    if held is not None:
        hold_k = np.arange(max(k[held.first_idx] - CLOSE_LEAD_FRAMES, 0), min(k[held.last_idx] + 2, len(t)))
        # The grip instant: ``select_held_object`` reports the pinch point over the few frames before
        # the object is first seen lifted, which is where the human fingers are physically on it.
        human_pinch = 0.5 * (lm_sim[:, 4] + lm_sim[:, 8])
        tactile_timing, hold_k, held = refine_grip_window(  # fmt: skip
            clip,
            t,
            first,
            side_i,
            human_pinch,
            hold_k,
            held,
            CLOSE_LEAD_FRAMES,
            MIN_PICK_RISE_M,
            GRIP_REACH_TOL_M,
        )
        close_start = hold_k[0]
        half = held.half_size_m
        half, mesh_kw, physics, thickness, open_width, pinch_axis, extruded, tan_alpha, tex, held_asset = (
            build_held_mesh(clip, held, lm_sim[hold_k[:10]], side, HAND)
        )
        # The table does not have to match the video, only carry the object where the robot can grip
        # it. Raising it to the human grip height put it above the G1 elbow on some clips (0.96 m on
        # clip 1, arm links 38-77 mm inside it), so its height is fitted to the robot instead (reach and
        # robot-table clearance) and the whole motion is shifted vertically to match; the object
        # still rests where the hand grips it. See ``stages/align/table_fit.py``.
        release_i = int(hold_k[-1])
        # The grip force is the finger stiffness times the squeeze, so the squeeze is sized from the
        # force this object needs and the stiffness the hand has at the hold pose (see ``grip``).
        mid_k = min(
            int(np.searchsorted(r.timestamps_s, t[hold_k[len(hold_k) // 2]])), len(r.q_target_rad) - 1
        )
        stiffness = hold_pinch_stiffness(r.q_target_rad[mid_k], side, HAND, max(thickness - SQUEEZE_M, 0.0))
        raw_flags: dict = {}
        held_base = (mesh_kw.get("mass_kg", BoxObject.mass_kg), mesh_kw.get("friction", BoxObject.friction))
        held_o = SIM_PARAMS.objects.get("held")
        if held_o is not None:  # RSI: mass / friction override of the carried object
            if held_o.mass_kg is not None:
                mesh_kw["mass_kg"] = held_o.mass_kg
            if held_o.friction is not None:
                mesh_kw["friction"] = (held_o.friction, 0.01, 0.001)
                physics = {**(physics or {}), "friction": held_o.friction}
        obj_mass = float(mesh_kw.get("mass_kg", BoxObject.mass_kg))
        obj_mu = max(float((physics or {}).get("friction", 1.0)), ObjectContact.min_slide_friction)
        grip_n = required_grip_force_n(
            obj_mass, obj_mu, tan_alpha, accel_m_s2=CARRY_ACCEL_M_S2, min_force_n=GRIP_MIN_N
        )
        squeeze = squeeze_for_force(grip_n, stiffness, min_squeeze_m=SQUEEZE_M)
        print(
            f"grip: mass {obj_mass * 1000:.0f} g, mu {obj_mu:.2f}, wedge tan {tan_alpha:.2f} -> "
            f"{grip_n:.1f} N; pinch stiffness {stiffness:.0f} N/m -> squeeze {squeeze * 1000:.1f} mm"
        )
        gaps = np.full(len(t), np.nan)
        gaps[close_start : release_i + 1] = max(thickness - squeeze, 0.0)
        # Approach with the fingers open around the object so the pinch point (not the wrist) is aligned.
        open_gap = max(thickness, open_width) + OPEN_MARGIN_M + OPEN_PAD_M[HAND]
        gaps[max(close_start - APPROACH_FRAMES, 0) : close_start] = open_gap
        # Close gradually: a one-frame step from open to squeezed slammed the Sharpa thumb into the
        # tongs of clip 100, knocking them 2.7 cm under the index, which then closed on the top face.
        n_ramp = min(CLOSE_RAMP_FRAMES, release_i + 1 - close_start)
        gaps[close_start : close_start + n_ramp] = np.linspace(open_gap, gaps[close_start], n_ramp + 1)[1:]
        # Open as deliberately: left to the human curl after the release, the Sharpa fingers stayed
        # 0.7-0.9 closed and the object stuck to a pad (clips 0, 1, 10). They open while the pinch
        # holds still and stay open through the retreat (see ``release_motion``).
        release_end = min(release_i + 1 + PLACE_DWELL_FRAMES + RETREAT_FRAMES + RETREAT_BLEND_FRAMES, len(t))
        gaps[release_i + 1 : release_end] = open_gap
        n_open = min(RELEASE_OPEN_FRAMES, release_end - release_i - 1)
        gaps[release_i + 1 : release_i + 1 + n_open] = np.linspace(gaps[release_i], open_gap, n_open + 1)[1:]
        exempt = np.zeros(len(t), bool)  # the active hand is meant to be at the object here
        exempt[max(hold_k[0] - APPROACH_FRAMES, 0) : release_end] = True
        support_xy = (
            tuple(np.subtract(table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"])),
            tuple(np.add(table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"])),
        )
        _lap("proxy rollouts + held object + grip")
        tfit = fit_table_height(
            lm_sim,
            None if passive is None else landmarks_camera_to_sim(passive, r.placement.T_sim_camera),
            side,
            held.pinch_z_pre_m,
            float(half[2]),
            support_xy,
            HAND,
            r.placement.waist_pitch_rad,
            float(base[2, 3]),
            exempt,
            hold=np.isin(np.arange(len(t)), hold_k),
            pinch_gap_m=gaps,
        )
        _lap("table height fit")
        if SIM_PARAMS.table.top_z_m is not None:  # RSI override: the whole motion follows the table
            dz = SIM_PARAMS.table.top_z_m - tfit.top_z_m
            tfit = dataclasses.replace(tfit, top_z_m=tfit.top_z_m + dz, shift_z_m=tfit.shift_z_m + dz)
        if tfit.object_half_z_m > half[2] + 1e-6:
            # The object reaches further down to the (lower) table; its top and the grip stay put.
            k_z = tfit.object_half_z_m / half[2]
            half = np.array([half[0], half[1], tfit.object_half_z_m])
            if "mesh_vertices" in mesh_kw:
                v = np.array(mesh_kw["mesh_vertices"], float)
                zc = 0.5 * (v[:, 2].max() + v[:, 2].min())
                v[:, 2] = zc + (v[:, 2] - zc) * k_z
                mesh_kw["mesh_vertices"] = v
        T_shift = r.placement.T_sim_camera.copy()
        T_shift[2, 3] += tfit.shift_z_m
        r = dataclasses.replace(r, placement=dataclasses.replace(r.placement, T_sim_camera=T_shift))
        cand_sim = cand_sim + np.array([0.0, 0.0, tfit.shift_z_m])
        grip_z = held.pinch_z_pre_m + tfit.shift_z_m
        table_z = tfit.top_z_m
        table_kw["table_top_z_m"] = table_z
        # The raised table must not swallow the resting hands; both are kept above it from here on.
        support = Support.from_table(table_z, table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"])
        lm_sim = lift_above_support(landmarks_camera_to_sim(lm, r.placement.T_sim_camera), support, exempt)
        human_pinch = 0.5 * (lm_sim[:, 4] + lm_sim[:, 8])
        anchor = np.array([*held.start_xy_m, grip_z])  # where the pinch closes on the resting object
        # The pinch closes where it is lowest just after ``close_start`` (see ``settle_approach``); the
        # human pinch there can sit off the object's rest point (clip 104 Dex3: 1.5 cm, the index came
        # down on the top face). Height is already matched through ``grip_z``.
        lead = human_pinch[close_start : close_start + CLOSE_LEAD_FRAMES + 1, 2]
        grasp_i = close_start + int(np.argmin(lead))
        # The tips must close low enough that the finger pads above them meet the sides, not the top
        # edge: closing at the human pinch height (up to the top face on clip 114) or 1.6 cm under
        # the top of a 4 cm object (clip 100 Dex3) put the distal links on the top edge, squeezing it
        # down into the table. Lower only; never into the table.
        tip_cap = max(table_z + 2 * half[2] - PAD_REACH_M[HAND], table_z + TIP_FLOOR_M)
        drop_z = min(0.0, tip_cap - human_pinch[grasp_i, 2])
        align_raw = np.array([*(anchor[:2] - human_pinch[grasp_i, :2]), drop_z])
        align = np.clip(align_raw, -MAX_OFFSET_M, MAX_OFFSET_M)
        dz = float(align[2])
        # Where the robot's pinch point has to be, frame by frame. The human pinch trajectory is
        # rigidly translated so that at the grasp instant it coincides with the object's graspable
        # centre; the relative lift/carry/place motion is preserved. Without this the fingers close
        # a few centimetres above the object, land on its top face and press it into the table
        # (measured on clip 1: 25 N downwards, zero lift). The correction fades out away from the
        # grasp window so the resting poses at both ends of the clip stay on the table.
        # A place-on-a-receptacle task needs the same accuracy about a *second* object at release,
        # so the correction is a per-phase blend: the grasped object's frame while the fingers
        # close, the receptacle's frame while they open. See docs/notes/target_frame_switching.md.
        task_text = json.loads((clip / "episode.json").read_text())["task_description"]
        place_target = resolve_place_target(
            task_text,
            float(half[2]),
            human_pinch[release_i],
            cand_sim,
            od["extent_cam"] * scale,
            in_span,
            held.start_xy_m,
        )
        scene_inv = None
        inv = load_inventory(clip) if SCENE_OBJECTS else None
        if inv is not None:
            # The inventory frame went through MoGe separately: carry the depth scale over by the
            # ratio of the two depth maps on the nearest object-tracking frame.
            near_f = int(np.argmin(np.abs(od["frame_idx"] - inv.frame)))
            inv_scale = scale * depth_ratio(od["depth_small"][near_f], inv.depth_small)
            scene_inv = build_scene_inventory(
                inv,
                inv_scale,
                r.placement.T_sim_camera,
                table_z,
                np.asarray(held.start_xy_m, float),
                object_prompt(task_text, default=""),
                place_target_prompt(task_text),
                source_prompt(task_text),
                float(table_kw["table_center_xy_m"][0] - table_kw["table_half_size_xy_m"][0]),
                place_hint_xy=None if place_target is None else place_target.drop_point_m[:2],
                held_half_xy=half[:2],
                # The start refine moves the held object by up to START_SEARCH_M; a welded lid within
                # that reach jams it (clip 0: the stapler wedged against the box lid, no lift).
                held_keepout_m=START_SEARCH_M / 2 + 0.01,
                carry_xy=human_pinch[(close_start + release_i) // 2 : release_i + 1, :2],
                source_from=SIM_PARAMS.scene.source_from,
                max_registration_m=(
                    None
                    if SIM_PARAMS.scene.max_registration_m == SceneParams().max_registration_m
                    else SIM_PARAMS.scene.max_registration_m
                ),
            )
            # RSI scene edits (enabled / pose / yaw / scale / static / robot_collision per object)
            rec_name = None if scene_inv.receptacle is None else scene_inv.receptacle.name
            raw_flags.update(
                {
                    o.name: (o.static, o.robot_collision)
                    for o in (*scene_inv.objects, scene_inv.receptacle)
                    if o
                }
            )
            edited = apply_scene_overrides(
                [*scene_inv.objects, *([scene_inv.receptacle] if rec_name else [])],
                SIM_PARAMS,
                {"receptacle": rec_name} if rec_name else {},
            )
            scene_inv.receptacle = next((o for o in edited if o.name == rec_name), None) if rec_name else None
            scene_inv.objects = [o for o in edited if o.name != rec_name]
            if scene_inv.receptacle is not None:
                place_target = place_target_for(
                    scene_inv.receptacle, float(half[2]), scene_inv.receptacle_floor_z
                )
            # Grow the table under the scenery (never towards the robot).
            ext = scene_extent_xy(
                scene_inv.objects + ([scene_inv.receptacle] if scene_inv.receptacle is not None else [])
            )
            if ext is not None:
                lo_t = np.subtract(table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"])
                hi_t = np.add(table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"])
                lo_n = np.minimum(lo_t, ext[0] - SCENE_TABLE_MARGIN_M)
                hi_n = np.maximum(hi_t, ext[1] + SCENE_TABLE_MARGIN_M)
                lo_n[0] = lo_t[0]
                table_kw["table_center_xy_m"] = tuple((lo_n + hi_n) / 2)
                table_kw["table_half_size_xy_m"] = tuple((hi_n - lo_n) / 2)
                support = Support.from_table(
                    table_z, table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"]
                )
            if "warning" in scene_inv.report:
                print("WARNING scene inventory:", scene_inv.report["warning"])
            print(
                f"scene inventory: {sum(o['kept'] for o in scene_inv.report['objects'])} of "
                f"{len(inv.items)} objects simulated; held={scene_inv.report['held_match']}, "
                f"receptacle={scene_inv.report['receptacle']}, source={scene_inv.report['source']}, "
                f"registration {scene_inv.report['registration_xy_m']} m"
            )
        place_k = np.arange(max(release_i - PLACE_LEAD_FRAMES, close_start), release_i + 1)
        # Every clip has a place phase: a bare-table transfer is put down too, not dropped from the
        # height the human let go at (clip 116: 13 cm) -- the receptacle's drop point is replaced
        # by the release point at the object's rest height.
        use_place_phase = PLACE_LEAD_FRAMES > 0
        # The human hand retreats the instant the object is let go; copying that drags the block out
        # of the bowl through the still-closing fingers, so the pinch dwells over the drop point.
        dwell_k = np.arange(release_i + 1, min(release_i + 1 + PLACE_DWELL_FRAMES, len(t)))
        if use_place_phase:
            place_k = np.concatenate([place_k, dwell_k])
        rel_xy = human_pinch[release_i][:2]
        keep_receptacle = False
        if place_target is not None and scene_inv is not None and scene_inv.receptacle is not None:
            # A receptacle from the inventory already stands in the registered layout; it stays there
            # (with everything around it) as long as the human lets go over it, and the drop point
            # follows the release within its rim.
            r_half = np.asarray(place_target.half_size_m[:2], float)
            r_pos = np.asarray(place_target.obj.pos_m[:2], float)
            keep_receptacle = bool((np.abs(rel_xy - r_pos) <= r_half + REBASE_OUTSIDE_M).all())
            # A receptacle found from where the carry went says the detected release is late (the
            # hand already away, clip 101): re-anchoring there would put it onto the held object.
            keep_receptacle |= scene_inv.report.get("receptacle_from") == "carry"
            if keep_receptacle:
                drop = place_target.drop_point_m.copy()
                drop[:2] = scene_inv.drop_xy(rel_xy, half)
                drop[2] = scene_inv.drop_z(drop[:2], half, drop[2], PLACE_CLEARANCE_M)
                place_target = dataclasses.replace(place_target, drop_point_m=drop)
        if place_target is not None and not keep_receptacle:
            # The receptacle is re-anchored onto the synthetic table at the release point, and kept
            # clear of the rim so a welded body can never hang over the floor.
            c_xy = np.asarray(table_kw["table_center_xy_m"], dtype=float)
            h_xy = (
                np.asarray(table_kw["table_half_size_xy_m"], dtype=float)
                - place_target.half_size_m[:2]
                - TABLE_INSET_M
            )
            place_target = rebase(
                place_target,
                np.clip(rel_xy, c_xy - h_xy, c_xy + h_xy),
                table_z,
            )
            if scene_inv is not None and scene_inv.receptacle is not None:
                scene_inv.move_receptacle(place_target.obj)
                drop = place_target.drop_point_m.copy()
                drop[:2] = scene_inv.drop_xy(rel_xy, half)
                drop[2] = scene_inv.drop_z(drop[:2], half, drop[2], PLACE_CLEARANCE_M)
                place_target = dataclasses.replace(place_target, drop_point_m=drop)
        if scene_inv is not None:
            scene_inv.report["receptacle_rebased"] = bool(place_target is not None and not keep_receptacle)
        holds = [np.arange(close_start, release_i + 1)]
        if use_place_phase:
            # The grasp frame hands over to the receptacle's before the release, otherwise both
            # windows are at full weight on the release frame and are halved, undershooting the drop.
            holds = [np.arange(close_start, max(place_k[0], close_start + 1)), place_k]
        windows = phase_windows(len(t), holds)
        window = windows.sum(axis=0)

        # The hands are kept above every other object the robot collides with, like above the table.
        others = [
            o
            for o in (
                *(() if place_target is None or place_target.obj is None else (place_target.obj,)),
                *(() if scene_inv is None else scene_inv.objects),
            )
            if o.robot_collision
        ]
        obstacles = tuple(sup for o in others for sup in object_supports(o, OBSTACLE_MARGIN_M))
        # ... and so is what they carry, piece by piece (it goes down into a pan past its rim).
        held_obstacles = tuple(sup for o in others for sup in object_supports(o, HELD_MARGIN_M))
        # The other hand never reaches into a container, so it stays above each object's whole extent:
        # kept only above the pieces, clip 104's idle right hand rested its fingers against the bowl's
        # rim, and the reaction twisted the waist 0.14 rad and the active hand 7 cm off the block.
        passive_obstacles = tuple(Support.from_points(object_points(o), OBSTACLE_MARGIN_M) for o in others)

        def _retarget(offset, _gaps=gaps, _place=r.placement, _support=support):
            return retarget_clip(
                lm,
                t,
                up,
                side,
                pinch_gap_m=_gaps,
                placement=_place,
                offset_m=offset,
                passive_landmarks_camera=passive,
                hand=HAND,
                support=_support,
                clearance_exempt=exempt,
                obstacles=obstacles,
                passive_obstacles=passive_obstacles,
            )

        # Several clips carry the object past the near edge of the table and drop it on the floor
        # (measured: clip 104 sharpa ends at x = -0.16 with the object 0.82 m below its rest height).
        # The table cannot be grown towards the robot without intersecting its own pelvis, so the
        # commanded pinch is kept over the support surface instead while the fingers are closed.
        tbl_c = np.asarray(table_kw["table_center_xy_m"], dtype=float)
        tbl_h = np.asarray(table_kw["table_half_size_xy_m"], dtype=float) - half[:2] - TABLE_INSET_M
        # The place phase is corrected towards the receptacle's drop point, measured against the
        # human pinch at release rather than at the grasp, so the two phases stay independent.
        aligns = [align]
        if use_place_phase and place_target is not None:
            aligns.append(
                np.clip(place_target.drop_point_m - human_pinch[release_i], -MAX_OFFSET_M, MAX_OFFSET_M)
            )
        elif use_place_phase:
            # Bare table: keep the carry's horizontal offset, come down to the grip height.
            rest_z = human_pinch[grasp_i, 2] + align[2] + BARE_PLACE_LIFT_M
            aligns.append(
                np.clip(
                    np.array([align[0], align[1], rest_z - human_pinch[release_i, 2]]),
                    -MAX_OFFSET_M,
                    MAX_OFFSET_M,
                )
            )
        dt_frame = float(np.median(np.diff(t)))
        # Back off when the shift leaves the arm's workspace: the place correction first, so a
        # receptacle that is hard to reach does not also pull the hand off the object it grasps
        # (clip 0 Sharpa: the whole-path back-off missed the stapler by 3.5 cm).
        back_off = [
            (1.0, 1.0),
            (1.0, 0.75),
            (1.0, 0.5),
            (1.0, 0.25),
            (1.0, 0.0),
            (0.75, 0.75),
            (0.5, 0.5),
            (0.25, 0.25),
            (0.0, 0.0),
        ]
        attempts: dict = {}

        def _attempt(frac, place_frac):
            if (frac, place_frac) in attempts:
                return attempts[frac, place_frac]
            fracs = [frac] + [place_frac] * (len(aligns) - 1)
            target_pinch = human_pinch + blend_corrections(
                windows, [f_ * a_ for f_, a_ in zip(fracs, aligns, strict=True)]
            )
            if place_target is not None and use_place_phase:
                # The receptacle's horizontal offset applies through the whole descent towards it,
                # otherwise the block comes down on the near rim; its height only near the release.
                # It never starts before the fingers have closed: on a short carry (clip 0: 32
                # frames) it would otherwise pull the open hand 3.5 cm off the object it reaches for.
                ramp_start = max(place_k[0] - PLACE_XY_LEAD_FRAMES, close_start + CLOSE_LEAD_FRAMES)
                ramp_len = max(place_k[0] - ramp_start, 1)
                ramp = np.clip((np.arange(len(t)) - ramp_start) / ramp_len, 0, 1)
                ramp[place_k[0] :] = 0.0  # the place window owns these frames
                target_pinch[:, :2] += place_frac * ramp[:, None] * aligns[-1][:2]
            held_xy = np.clip(target_pinch[:, :2], tbl_c - tbl_h, tbl_c + tbl_h)
            target_pinch[:, :2] += window[:, None] * (held_xy - target_pinch[:, :2])
            # The carry, not the place window: the drop point is already clear of the receptacle's
            # collision, and lifting over its rim there released clip 10's block 8 cm up.
            carry_end = place_k[0] if use_place_phase else release_i + 1
            carried = np.zeros(len(t), bool)
            carried[close_start + CLOSE_LEAD_FRAMES : carry_end] = True
            lift = held_object_lift(target_pinch, half, held_obstacles, carried, HELD_CLEARANCE_M)
            lift[carry_end:] = 0.0
            target_pinch[:, 2] += lift
            target_pinch, carry_sigma = limit_carry_acceleration(
                target_pinch, close_start + CLOSE_LEAD_FRAMES, release_i, dt_frame, CARRY_ACCEL_M_S2
            )
            target_pinch, release_w, _ = release_motion(
                target_pinch,
                release_i,
                PLACE_DWELL_FRAMES,
                RETREAT_FRAMES,
                RETREAT_HEIGHT_M,
                RETREAT_BLEND_FRAMES,
            )
            target_pinch, approach_w = settle_approach(
                target_pinch,
                close_start,
                close_frames=CLOSE_LEAD_FRAMES,
                hover_z_m=min(table_z + 2 * half[2] + HOVER_CLEARANCE_M, anchor[2] + MAX_HOVER_RISE_M),
            )
            r_try, hand_eye_offset, pinch_rms_m = align_pinch(
                _retarget,
                lambda res: pinch_track(res.ik.q_target_rad, side, HAND, base),
                target_pinch,
                np.maximum.reduce([window, approach_w, release_w]),
            )
            bad = float((r_try.ik.pos_err_m > 0.03).mean())
            attempts[frac, place_frac] = (bad, r_try, hand_eye_offset, pinch_rms_m, target_pinch, carry_sigma)
            return attempts[frac, place_frac]

        # A correction is backed off only for the frames it takes out of reach. Some clips leave
        # the workspace without any correction (clip 117 Sharpa: 19 % of the frames even uncorrected);
        # all seven back-offs then failed the fixed 5 % bound, the hand kept the human release over
        # the scooper's handle instead of its pan, and the retarget took 67 min.
        bad_max = 0.05
        if _attempt(*back_off[0])[0] > bad_max:
            # The place correction is judged against the grasp-only path, not the uncorrected clip:
            # what the grasp correction already takes out of reach (clip 1: 12 %) is not its fault.
            bad_max = max(bad_max, _attempt(1.0, 0.0)[0] + 0.01)
        frac, place_frac = next((b for b in back_off if _attempt(*b)[0] <= bad_max), back_off[-1])
        _, r_try, hand_eye_offset, pinch_rms_m, target_pinch, carry_sigma = _attempt(frac, place_frac)
        r, dz = r_try, frac * dz
        print(f"path correction: grasp {frac:.2f}, place {place_frac:.2f}")
        retry_log["passes"].append(
            {
                "pass": len(retry_log["passes"]) + 1,
                "frac": frac,
                "place_frac": place_frac,
                "bad": _attempt(frac, place_frac)[0],
                "wall_s": time.time() - retry_log["t0"],
            }
        )
        if use_place_phase and place_frac < 1.0 and _goal_env != "off" and PLACE_STANCE_SEARCH:
            # The standing spot was chosen on the human path alone; the receptacle's drop point is out
            # of reach from it (clips 114, 116: the hand released 5-7 cm above the bowl). Start over with
            # the spot scored on the corrected path, which can move the robot instead of the drop point;
            # if that does not bring the full correction within reach, start over with the first spot.
            passes = retry_log["passes"]
            pass_s = passes[-1]["wall_s"] - (passes[-2]["wall_s"] if len(passes) > 1 else 0.0)
            if place_goal is None and passes[-1]["wall_s"] + pass_s > PLACE_RETRY_BUDGET_S:
                # a retry costs about a full pass; start none that would overrun the budget
                retry_outcome = "budget_exhausted"
            else:
                if place_goal is None:
                    placement_cache.mkdir(exist_ok=True)
                    with open(placement_cache / f"{clip.name}_{HAND}.pkl", "wb") as f_:
                        pickle.dump((r_first, stance_adj), f_)
                    goal = PlaceGoal(
                        tuple(float(v) for v in aligns[-1]),
                        int(place_k[0]),
                        int(release_i),
                        PLACE_REACH_WEIGHT_M,
                    )
                    os.environ["E2H_PLACE_GOAL"] = json.dumps(dataclasses.asdict(goal))
                else:
                    os.environ["E2H_PLACE_GOAL"] = "off"
                    retry_log["outcome"] = "restored_worse"
                os.environ["E2H_PLACE_RETRY"] = json.dumps(retry_log)
                print(f"place correction {place_frac:.2f}: restarting", flush=True)
                os.execv(sys.executable, [sys.executable, *sys.argv])
        elif place_goal is not None:
            # The retry reached the full place correction; keep it only if it did not cost the grasp
            # path (clip 114: the re-scored stance degraded the pass). Otherwise restore the first pass.
            p1, p2 = retry_log["passes"][0], retry_log["passes"][-1]
            if p2["frac"] < p1["frac"] or p2["bad"] > p1["bad"] + 0.01:
                os.environ["E2H_PLACE_GOAL"] = "off"
                retry_log["outcome"] = "restored_worse"
                os.environ["E2H_PLACE_RETRY"] = json.dumps(retry_log)
                print("place retry is worse than the first pass: restoring it", flush=True)
                os.execv(sys.executable, [sys.executable, *sys.argv])
            retry_outcome = "accepted"
        elif _goal_env == "off":
            retry_outcome = retry_log.get("outcome", "restored_first")
        if placement_cache.exists():
            shutil.rmtree(placement_cache)
        place_diag = None
        if use_place_phase:
            # Where the place correction is lost: asked for (aligns[-1]), kept after the back-off,
            # commanded to the IK (target_pinch) and reached (pinch_track) at the release frame.
            got_pinch = pinch_track(r.ik.q_target_rad, side, HAND, base)
            ref = np.asarray(
                place_target.drop_point_m if place_target is not None else target_pinch[release_i]
            )
            place_diag = {
                "drop_point_m": ref.tolist(),
                "asked_offset_m": aligns[-1].tolist(),
                "kept_frac": place_frac,
                "human_release_m": human_pinch[release_i].tolist(),
                "commanded_release_m": target_pinch[release_i].tolist(),
                "reached_release_m": got_pinch[release_i].tolist(),
                "commanded_minus_drop_m": (target_pinch[release_i] - ref).tolist(),
                "reached_minus_commanded_m": (got_pinch[release_i] - target_pinch[release_i]).tolist(),
                "reached_minus_drop_m": (got_pinch[release_i] - ref).tolist(),
                "ik_err_release_m": float(r.ik.pos_err_m[release_i]),
            }
            print("place diag: " + ", ".join(f"{k}={np.round(v, 3).tolist()}" for k, v in place_diag.items()))
        print("  unreachable per try: " + ", ".join(f"{k}: {v[0]:.3f}" for k, v in attempts.items()))
        print(
            f"held path: carry smoothing {carry_sigma:.0f} frames (limit {CARRY_ACCEL_M_S2:.1f} m/s^2); "
            f"release: open {RELEASE_OPEN_FRAMES} f, dwell {PLACE_DWELL_FRAMES} f, "
            f"retreat {RETREAT_HEIGHT_M * 100:.0f} cm/{RETREAT_FRAMES} f, blend {RETREAT_BLEND_FRAMES} f"
        )
        if "mesh_vertices" in mesh_kw:
            # The object was yawed to a pre-solved robot pinch axis, but the twist of the final
            # solution can differ (clip 1 Sharpa: 50 deg, the fingers then met a 7 cm span with a
            # 3.8 cm gap and pressed on top). Re-yaw the narrow side to the axis actually commanded at
            # the grasp; the width across it, and therefore the gaps, stay the same. The axis is read
            # while the pinch is still open and comes down around the object: the Dex3 thumb closes
            # on an arc, turning the tip axis by 70 deg until the tips meet (clip 108: -82 deg open,
            # -153 deg shut), and a sliver yawed to the shut axis lies across the open fingers.
            pre = np.arange(max(close_start - GRASP_AXIS_FRAMES, 0), close_start)
            grasp_k = np.searchsorted(r.timestamps_s, t[pre if len(pre) else hold_k[:1]])
            got = commanded_pinch_axis_xy(r.q_target_rad[np.unique(grasp_k)], side, HAND, base)
            yaw = np.arctan2(got[1], got[0]) - np.arctan2(pinch_axis[1], pinch_axis[0])
            yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # the axis has no sign
            axis_now = pinch_axis
            if abs(yaw) > np.radians(PINCH_AXIS_TOL_DEG):
                c_, s_ = np.cos(yaw), np.sin(yaw)
                v = np.array(mesh_kw["mesh_vertices"], float)
                ctr = 0.5 * (v.max(0) + v.min(0))
                v = (v - ctr) @ np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]]) + ctr
                half = 0.5 * (v.max(0) - v.min(0))
                mesh_kw = {**mesh_kw, "mesh_vertices": v}
                axis_now = got
                print(f"object re-yawed {np.degrees(yaw):+.0f} deg to the commanded pinch axis")
            # The stretched surface keeps its rounded rim, so its hull tapers and the fingers slip up
            # the slanted sides; it collides as a straight prism over its footprint instead. A
            # reconstructed (not stretched) surface tapers the same way -- MoGe sees only the top --
            # so every held mesh does. A thin extruded one also gets parallel pinched faces.
            from stages.reconstruction.objects.mesh_asset import prism_hull, slab_prism

            v = mesh_kw["mesh_vertices"]
            piece = (
                slab_prism(v, axis_now, thickness)
                if extruded and thickness < SLAB_MAX_WIDTH_M
                else prism_hull(v)
            )
            mesh_kw = {**mesh_kw, "collision_pieces": (piece,)}
        # The table edge is fitted to the robot, not to the video: an elbow or upper arm hanging over
        # the near edge (clip 1: upper arm 3 mm inside it) is cleared by moving that edge forward,
        # but never past the resting object, any receptacle or the held pinch path.
        keep_x = [anchor[0] - half[0] - START_SEARCH_M, float(np.min(target_pinch[window > 0.5, 0]))]
        if place_target is not None and place_target.obj is not None:
            keep_x.append(place_target.obj.pos_m[0] - place_target.obj.half_size_m[0])
        lo_x, deficit = near_edge_clearance(
            r.q_target_rad, support, HAND, float(base[2, 3]), min(keep_x) - TABLE_INSET_M
        )
        if lo_x > support.lo_xy_m[0] + 1e-3:
            hi_x = support.hi_xy_m[0]
            table_kw["table_center_xy_m"] = (0.5 * (lo_x + hi_x), table_kw["table_center_xy_m"][1])
            table_kw["table_half_size_xy_m"] = (0.5 * (hi_x - lo_x), table_kw["table_half_size_xy_m"][1])
            support = Support.from_table(
                table_z, table_kw["table_center_xy_m"], table_kw["table_half_size_xy_m"]
            )
            print(
                f"table near edge moved to x={lo_x:.3f} m for arm clearance"
                + (f" ({deficit * 1000:.0f} mm short of the margin)" if deficit > 1e-4 else "")
            )
        lm_sim = lift_above_support(landmarks_camera_to_sim(lm, r.placement.T_sim_camera), support, exempt)
        # The pinch gaps and the z offset change the trajectory, so the self-collision and tracking
        # numbers have to be measured again on the trajectory that is actually executed and saved.
        res = run_rollout(
            SceneConfig.from_params(SIM_PARAMS, T_sim_robot_base=base, hand=HAND, **table_kw),
            r.timestamps_s,
            r.q_target_rad,
        )
        a = res.arrays
        m = min(len(a["q_actual_rad"]), len(r.q_target_rad))
        err = np.abs(a["q_actual_rad"][:m] - a["q_command_rad"][:m])
        # Penetration that is not intended (the active fingers may touch the table while gripping).
        pen = arm_penetration(
            table_penetration(r.q_target_rad, support, HAND, float(base[2, 3])),
            side,
            np.interp(r.timestamps_s, t, exempt.astype(float)) > 0.5,
        )
        table_pen = {
            "init_penetration_m": float(pen[0].max()),
            "max_penetration_m": float(pen.max()),
            "penetrating_fraction": float((pen.max(axis=1) > 0.005).mean()),
            "torso_contact_fraction": float(
                ArmCollisionChecker(build_scene(SceneConfig(T_sim_robot_base=base, hand=HAND)))
                .torso_frames(r.q_target_rad)
                .mean()
            ),
        }
        print("table fit:", tfit, table_pen)
        # Perceived depth is only good to a few cm: refine the resting xy within that uncertainty by physics.
        best = None
        # The receptacle has to be in the physics scene, otherwise "place it on the bowl" lands on
        # the bare table and nothing about the placement can be scored. It is welded in place.
        extra = (place_target.obj,) if place_target is not None and place_target.obj is not None else ()
        if scene_inv is not None:
            extra = (*extra, *scene_inv.objects)
        drop_point = place_target.drop_point_m if place_target is not None else None
        _lap("placement re-retarget + table check")
        # How far the demonstration carries the object: the pinch path from the grip to the release.
        demo_carry = float(np.linalg.norm(target_pinch[release_i, :2] - target_pinch[close_start, :2]))
        case = SimCase(
            hand=HAND,
            T_sim_robot_base=base,
            table_kw=table_kw,
            timestamps_s=r.timestamps_s,
            q_target_rad=r.q_target_rad,
            anchor=anchor,
            half_size_m=np.asarray(half, float),
            table_z=table_z,
            mesh_kw=mesh_kw,
            extra=tuple(extra),
            drop_point_m=drop_point,
            demo_carry_m=demo_carry,
            release_t_s=float(t[release_i] - t[0]),
            base_flags={
                **raw_flags,
                **{o.name: (o.static, o.robot_collision) for o in extra if o.name not in raw_flags},
            },
            held_base=held_base,
            receptacle_name=None
            if place_target is None or place_target.obj is None
            else place_target.obj.name,
        )
        sr = start_search(case, SIM_PARAMS, OFFSETS_M)
        start, pbox, pres, pev = sr.start_m, sr.box, sr.result, sr.events
        _lap(f"start refine ({len(OFFSETS_M)} rollouts)")
        # Real2sim scene for downstream sim stages (locomanipulation): the perceived held object,
        # the scenery and the table, all in the retargeting frame.
        with open(clip / "scene_sim.pkl", "wb") as f_:
            pickle.dump(dict(pbox=pbox, extra=extra, table_kw=table_kw, table_z=table_z, base=base), f_)
        t_rel = t - t[0]

        def _at(i: int) -> float:
            return float(t_rel[min(int(i), len(t_rel) - 1)])

        n_ramp = min(CLOSE_RAMP_FRAMES, release_i + 1 - close_start)
        n_open = min(RELEASE_OPEN_FRAMES, release_end - release_i - 1)
        carry_k = int(place_k[0]) if len(place_k) else release_i
        hold_end = min(close_start + n_ramp + 15, carry_k)  # first 0.3 s after the close: settle in the grip
        phases_t_s = {
            "approach": (_at(close_start - APPROACH_FRAMES), _at(close_start)),
            "close": (_at(close_start), _at(close_start + n_ramp)),
            "hold": (_at(close_start + n_ramp), _at(hold_end)),
            "carry": (_at(hold_end), _at(carry_k)),
            "place": (_at(carry_k), _at(release_i)),
            "release": (_at(release_i), _at(release_i + n_open)),
            "retreat": (_at(release_i + n_open), _at(release_end - 1)),
        }
        refs = {"t_s": t_rel, "human_pinch_m": human_pinch, "target_pinch_m": target_pinch}
        if tactile_timing and tactile_timing.get("grasp_onset_s") is not None:
            refs["grasp_onset_s"] = np.asarray(tactile_timing["grasp_onset_s"], float)
        save_bundle(
            clip / f"real2sim{SUFFIX}",
            case,
            SIM_PARAMS,
            clip=clip.name,
            side=side,
            best_start_xy_m=tuple(start[:2]),
            phases_t_s=phases_t_s,
            refs=refs,
            extra_meta=dict(
                sized_grip_n=float(grip_n),
                held_asset=held_asset,
                tactile_timing=tactile_timing,
                close_start_idx=int(close_start),
                release_idx=int(release_i),
                scene_layout=None
                if scene_inv is None
                else json.loads(
                    json.dumps(scene_inv.report, default=lambda o: getattr(o, "tolist", lambda: float(o))())
                ),
                place_retry={
                    "outcome": retry_outcome,
                    "budget_s": float(PLACE_RETRY_BUDGET_S),
                    "elapsed_s": time.time() - retry_log["t0"],
                    "passes": retry_log["passes"],
                },
                reach=dict(
                    place_unreach_frac=window_unreachable_fraction(
                        r.ik.pos_err_m, t, (t[carry_k], t[release_i])
                    ),
                    grasp_unreach_frac=window_unreachable_fraction(
                        r.ik.pos_err_m, t, (t[max(close_start - APPROACH_FRAMES, 0)], t[close_start])
                    ),
                    stance_scored_on_place=place_goal is not None,
                ),
            ),
        )
        obj_final_xy = pres.arrays["object_T_sim"][-1, 0, :2, 3].tolist()
        if pbox.mesh_vertices is not None:
            obj_pts = np.asarray(pbox.mesh_vertices) + np.asarray(pbox.pos_m)
        else:
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            obj_pts = corners * np.asarray(pbox.half_size_m) + np.asarray(pbox.pos_m)
        flags = contact_flags(lm_sim[hold_k], obj_pts)
        write_alignment_bundle(clip, r.placement.T_sim_camera, table_z, scale, flags)
        if "tactile" in names and (clip / names["tactile"]).exists():
            validate_npz("tactile", clip / names["tactile"])
            shutil.copyfile(clip / names["tactile"], clip / "alignment" / "tactile_raw.npz")
        _lap("alignment bundle")
        touch = None
        if os.environ.get("RENDER") == "1":
            touch, blender_grid = render_review(
                clip=clip,
                SUFFIX=SUFFIX,
                HAND=HAND,
                base=base,
                pbox=pbox,
                extra=extra,
                table_kw=table_kw,
                r=r,
                t=t,
                est=est,
                sl=sl,
                side=side,
                MIN_SCORE=MIN_SCORE,
                pev=pev,
                f=f,
                SIM_PARAMS=SIM_PARAMS,
                D435_IMAGE_HW=D435_IMAGE_HW,
                EGO_PANEL=EGO_PANEL,
                _lap=_lap,
            )
        perceived = {
            "depth_scale": scale,
            "table_shift_m": float(dz),
            "views": [[int(fi[i]), k] for i, k in held.views],
            "n_matches": held.n_matches,
            "hold_s": [float(t[hold_k[0]] - t[0]), float(t[hold_k[-1]] - t[0])],
            "start_m": start.tolist(),
            "start_refine_m": (start[:2] - held.start_xy_m).tolist(),
            "half_size_m": half.tolist(),
            "pinch_rms_m": pinch_rms_m,
            "object_align_m": (frac * align).tolist(),
            "object_align_raw_m": align_raw.tolist(),
            "table_top_z_m": table_z,
            "tactile_timing": tactile_timing,
            "support_touch": touch,
            "table_fit": {**dataclasses.asdict(tfit), **table_pen},
            "table_center_xy_m": list(table_kw["table_center_xy_m"]),
            "table_half_size_xy_m": list(table_kw["table_half_size_xy_m"]),
            "object_final_xy_m": obj_final_xy,
            "release_t_s": float(t[release_i] - t[0]),
            "place_target": None
            if place_target is None
            else {
                "name": place_target.name,
                "drop_point_m": place_target.drop_point_m.tolist(),
                "half_size_m": np.asarray(place_target.half_size_m).tolist(),
                "source": "scene" if place_target.obj is not None else "detections",
            },
            "hand_eye_offset_max_m": float(np.abs(hand_eye_offset).max()),
            "place_diag": place_diag,
            "mesh_asset": physics is not None,
            "held_asset": held_asset,
            "mesh_texture_iou": (
                None if physics is None or tex.texture_iou is None else round(tex.texture_iou, 3)
            ),
            "physics": physics,
            "observed_lift_m": held.observed_lift_m,
            "contact_steps": int(pres.contact_steps_with_object),
            "arm_collision_steps": int(pres.arm_collision_steps),
            "sim_events": pev.as_dict(),
            "scene_objects": None
            if scene_inv is None
            else {
                **scene_inv.report,
                **disturbance(
                    pres.arrays["object_T_sim"],
                    [pbox.name, *(o.name for o in extra)],
                    {pbox.name, *(o.name for o in extra if o.static)},
                ),
            },
            "human_pinch_contact_frames": int(flags["pinch"].sum()),
        }
report = {
    "sim_params_hash": params_hash(SIM_PARAMS),
    "perceived_object": perceived,
    "hand": HAND,
    "object_source": "grasp_proxy",
    "proxy_events": objects_report,
    "side": side,
    "valid_frames": float(ok.mean()),
    "placement_xy_m": list(r.placement.target_xy_m),
    "waist_pitch_rad": r.placement.waist_pitch_rad,
    "stance": None if stance_adj is None else stance_adj.as_dict(),
    "camera_coverage": r.placement.camera_coverage,
    "camera_layout_err": r.placement.camera_layout_err,
    "ik_pos_err_mean_mm": float(r.ik.pos_err_m.mean() * 1e3),
    "ik_pos_err_max_mm": float(r.ik.pos_err_m.max() * 1e3),
    "unreachable_fraction": float((r.ik.pos_err_m > 0.03).mean()),
    "passive_side": None if passive is None else ("left", "right")[other_i],
    "passive_valid_fraction": float(ok_p[sl].mean()),
    "passive_unreachable_fraction": None
    if r.ik_passive is None
    else float((r.ik_passive.pos_err_m > 0.03).mean()),
    "arm_collision_steps": int(res.arm_collision_steps),
    "track_err_mean_rad": float(err.mean()),
    "track_err_max_rad": float(err.max()),
    "proxy": None
    if proxy is None
    else {
        "grasp_t": float(t[proxy.grasp_idx] - t[0]),
        "release_t": float(t[proxy.release_idx] - t[0]),
        "center_m": proxy.center_m.tolist(),
        "half_size_m": proxy.half_size_m.tolist(),
    },
    "finite": bool(np.isfinite(a["q_actual_rad"]).all()),
}
(clip / f"retarget_rgb{SUFFIX}.json").write_text(json.dumps(report, indent=1))
np.savez_compressed(
    clip / f"retarget_rgb{SUFFIX}.npz", timestamps_s=r.timestamps_s, q_target_rad=r.q_target_rad
)
print(json.dumps(report))
_lap("report")
if blender_grid is not None:
    import subprocess

    from stages.qa.grid_video import compose_grid, read_video_rgb

    py = Path(sys.executable)
    export = Path(__file__).with_name("export_twin_scene.py")
    subprocess.run([str(py), str(export), str(clip), OUT_TAG], check=True)
    bpy_py = os.environ.get("BPY_PYTHON", "/tmp/bpy_venv/bin/python")
    subprocess.run(
        [
            bpy_py,
            str(Path(__file__).with_name("twin_render.py")),
            str(clip),
            "--tag",
            OUT_TAG,
            "--views",
            "ego",
            "third",
            "--fps",
            str(RENDER_FPS),
            "--device",
            "gpu",
        ],
        check=True,
        env={**os.environ, "TMPDIR": "/tmp"},
    )
    step = round(50 / RENDER_FPS)
    left_frames, traj_frames = blender_grid
    third = read_video_rgb(clip / "review" / f"third_blender{OUT_TAG}.mp4")
    ego = read_video_rgb(clip / "review" / f"ego_blender{OUT_TAG}.mp4")
    grid_path = compose_grid(
        [
            ("RGB + estimated hands (+tactile)", left_frames[::step]),
            (f"robot in Blender ({HAND})", third),
            (EGO_PANEL + ", Blender", ego),
            ("task-space trajectories", traj_frames[::step]),
        ],
        clip / "review" / f"result_grid{SUFFIX}.mp4",
        fps=RENDER_FPS,
    )
    print("grid video:", grid_path)
    _lap("blender views + grid")
