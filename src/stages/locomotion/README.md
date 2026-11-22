# `src/stages/locomotion/` — stance, walking and approach for loco-manipulation

Loco-manipulation: where the robot stands, how it walks there, and the approach episode (walk -> raise arm ->
align -> drive the arm through object-relative keyframes). Stance planning proper is in
[`plan/`](plan/README.md). Design: [docs/TECH_DESIGN.md §4.1 and §6](../../../docs/TECH_DESIGN.md).
`__init__.py` holds only a docstring.

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `base_placement.py` | pick where the fixed-pelvis robot stands relative to the hand motion (IK reach + head-camera view) | `place_hand_motion(landmarks_camera, R_sim_camera, side, table_top_z_m=0.8, stride=4, passive_landmarks_camera=None, pinch_uv_ref=None, hand="dex3", place_goal=None) -> Placement`; `select_placement(scored)`; `ik_waist(model) -> int`; `Placement`, `PlaceGoal(offset_sim_m, start, release, reach_weight_m=0.1)` (`.apply`); `OTHER_SIDE` |
| `stance_adjust.py` | move the robot only if camera coverage / reach fail at the original stance | `adjust_stance(landmarks_camera, side, placement, window, pinch_uv_ref, hand="sharpa", stride=4) -> StanceAdjustment`; `critical_window(t_s, grasp_t_s, release_t_s)`; `reanchor(T_sim_camera, dx, dy, yaw)` |
| `tabletop.py` | MuJoCo [[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco)] scenario: robot + floor + table + obstacles + object + goal | `Scenario` (`load(path=DEFAULT_CONFIG)`, `build(hand="dex3")`, `build_spec`, `table_top_z`, `object_start_xyz`, `goal_xy`, `reset_object`, `obstacle_boxes`, `occupancy(extent, res=0.05, robot_margin=0.0)`) |
| `route.py` | grid A* to waypoints | `plan_route(grid, extent, res, start_xy, goal_xyyaw, spacing=0.5) -> list[(x, y, heading)] \| None` |
| `executor.py` | closed-loop waypoint pursuit on Decoupled WBC [[NVIDIA et al., 2025](../../../docs/REFERENCES.md#nvidia2025gr00tn1)] with arrival gate | `goto(robot: DwbcRobot, waypoints, timeout_s=60.0, log_every_s=0.1, shuffle=False) -> ArrivalReport`; `wrap(a)`, `base_yaw(qpos)` |
| `episode.py` | full approach episode | `run_episode(sc, task, side="right", hand="sharpa", video=None, poses=None, extra_stances=(), plan_kw=None) -> dict` |
| `clip_task.py` | turn a retargeted EgoDex [[Hoque et al., 2025](../../../docs/REFERENCES.md#hoque2025egodex)] clip into a `ClipCase` | `load_clip(clip: int\|str, derived=DERIVED) -> ClipCase`; `ClipCase(clip, side, scenario, task, poses, native_stances)`; `ClipPoses.pose(stage, side)` |
| `video.py` | third-person H.264 recording (PyAV, EGL) | `VideoRecorder` (`attach(model, data)`, `maybe_capture(robot=None)`, `close()`) |

## Inputs / outputs

- `tabletop.py` reads `configs/scenarios/tabletop_v1.json` (`DEFAULT_CONFIG`); `occupancy` returns a blocked
  grid (inflated by the robot radius) over `extent=((-1,5),(-2,2))` m at 0.05 m.
- `run_episode` returns a dict; on success: `ok=True`, `arrival`, `visible_fraction_measured`,
  `min_pelvis_z`, `max_tilt_rad`, `measured_stance`, `wrist_track_err_m` (per keyframe),
  `object_xyz_after_keyframe`, `object_final_xyz`, `object_max_lift_m`. On failure: `ok=False` and
  `failure` (a `contracts.locomanip.LocoManipFailure`: e.g. `NO_NAVIGATION_PATH`, `LOCOMOTION_FALL`,
  `ARM_IK_INFEASIBLE`).
- `ArrivalReport(arrived, fell, pos_err_m, yaw_err_rad, time_s, reason, log)`.
- `clip_task` reads `data/derived/egodex/<clip>/` retarget outputs; table placed `TABLE_X_M = 3.0` m from the start.
- Frames: sim world z-up; stance = `(x, y, yaw)` plus pelvis height and waist pitch.

## Callers

`scripts/loco_manip_episode.py` (episode, clip_task, tabletop, plan), `scripts/render_scene_preview.py`
(clip_task), `scripts/solve_sharpa_pinch.py` (tabletop), `scripts/retarget_estimated.py` and
`stages/retargeting/{clip,script_helpers}.py` (base_placement, stance_adjust), `skills/reachability_check.py`
(base_placement), `reconstruction/align/table_fit.py` (lazy `ik_waist`). Tests: `tests/unit/test_nav_route.py`,
`test_stance_adjust.py`, `tests/smoke/test_locomotion_goto.py`, `test_approach_episode.py`.

## Constants and env vars

- `base_placement`: `X_GRID_M=(0.2..0.5)`, `Y_GRID_M=(0.0..0.25)`, `CAMERA_PITCHES_RAD`, `IK_SLACK_M=0.01`;
  env `E2H_WORKERS` = processes for the candidate grid (default all cores, 1 = serial).
- `executor`: `MAX_VX=0.5`, `MAX_YAW_RATE=0.4`, `POS_TOL_M=0.08`, `YAW_TOL_RAD=0.12`, `FALL_Z=0.4`.
- `episode`: `SETTLE_S=2.0`, `KEYFRAME_MOVE_S=2.0`, `ARM_RAISE_S=3.0`, `SERVO_ITERS=2`, `SERVO_CLAMP_M=0.02`.
- `stance_adjust`: `PITCHES_RAD`, `LEVELS`, `COVERAGE_OK=0.9`.

## Gotchas

- `video.py` has no importer besides `episode.py`; it needs EGL offscreen rendering.
- The retarget stage keeps the pelvis fixed (`FIXED_PELVIS_Z = 0.793`) at the origin facing +x; locomotion
  shifts the scene instead of the retarget.
- `plan_route` needs a grid already inflated by the base radius.
- `episode.py` imports `robot.*`, `contracts.*` and `stages.retargeting.sharpa_pinch` (outside this package).
- `shuffle=True` in `goto` (clip mode, when `arrival_slack_m` is set) avoids U-turns on overshoot.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-hoque2025egodex"></span>[Hoque et al., 2025](../../../docs/REFERENCES.md#hoque2025egodex). *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.*
- <span id="cite-nvidia2025gr00tn1"></span>[NVIDIA et al., 2025](../../../docs/REFERENCES.md#nvidia2025gr00tn1). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
