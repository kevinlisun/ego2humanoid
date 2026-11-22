# `src/stages/locomotion/plan/` — static stance planning

Static stance planning for loco-manipulation (L.8 - L.12): sample candidate stances around the object,
solve stationary arm IK for each, validate contacts and head-camera visibility, rank, and emit a `PoseGoal`.
Parent: [locomotion](../README.md). Design: [docs/TECH_DESIGN.md §4.1](../../../../docs/TECH_DESIGN.md).
`__init__.py` is a docstring only.

## Modules

| File | Step | Key public API (verified) |
|---|---|---|
| `candidates.py` | L.8 sample stances on an annulus x heading x height x waist pitch | `Stance(x, y, yaw, height_m, waist_pitch_rad)`; `sample_candidates(object_xy, grid, extent, res, focus_xy=None) -> (list[Stance], dict[str, int])` |
| `ik.py` | L.9 stationary arm IK per keyframe | `KeyframeIK(phase, q_arm, pos_err_m, rot_err_rad, feasible)`; `StanceIK(side)` with `world_to_pelvis_frame(stance, T_world)`, `solve(stance, T_world_object, keyframes, limit_margin=0.02)`, `approach_path(stance, T_world_target, retreat_m=(-0.2, 0.0, 0.2), n=8)`, `wrist_world(stance, q_arm)` |
| `visibility.py` | L.10 head-camera visibility | `set_stance(model, data, stance)`; `visible_fraction(model, data, cam, stance, pts_world, margin=0.15) -> float`; `box_surface_points(center, half_size, n_per_axis=3)` |
| `validators.py` | L.11 terminal contact validators | `tucked_arm(side) -> q`; `set_arm(model, data, side, q_arm)`; `forbidden_contacts(model, data, stance, side, q_arm, object_name, allow_object, hand_table_ok=False) -> names` |
| `rank.py` | L.12 hard constraints, weighted cost, `PoseGoal` | `plan_stance(sc, task, side="right", extra_stances=(), hand="dex3", hand_table_ok=False, vis_min=None, arrival_slack_m=0.0, robot_radius_m=0.22) -> (PoseGoal \| None, failure \| None)` |
| `golden.py` | manual keyframes for the golden pick-and-place (spec §15) | `golden_task(sc, hand="right") -> ManipulationTask` |

## Inputs / outputs

- Input: a `Scenario` (`../tabletop.py`), a `contracts.locomanip.ManipulationTask` with object-relative
  `TaskKeyframe`s (phases such as `pre_grasp`), the active `side`, and optional extra candidate stances
  (`clip_task` offers the retarget's native stance).
- Output: `(PoseGoal, None)` on success, otherwise `(None, failure)` with failure a
  `LocoManipFailure`; the planner also reports rejected candidates by reason.
- Units: metres/radians, world z-up. `golden.py`: `PINCH_OFFSET = (-0.07, -0.05, 0.0)`,
  `SIDE_CLEAR_Y = -0.17`, keyframes in `GOLDEN_KEYFRAMES`.

## Constants

- `candidates`: `HEIGHTS_M=(0.6, 0.7, 0.74)`, `WAIST_PITCH_RAD=(0.0, 0.2, 0.4, 0.5)`,
  `RADII_M=(0.3..0.65)`, `BEARING_STEP_DEG=30`, `YAW_OFFSETS_RAD`.
- `ik`: `POS_TOL_M=0.02`, `ROT_TOL_RAD=0.25`, `MAX_KEYFRAME_STEP_RAD=2.0`, `FIXED_PELVIS_Z=0.793`.
- `visibility`: `MIN_VISIBLE_FRACTION=0.8`, `MIN_VISIBLE_FRACTION_GOAL=0.4`, `BORDER_MARGIN=0.15`,
  `OCCLUSION_EPS_M=0.02`.
- `validators`: `PENETRATION_TOL_M=0.005`, `LEG_PARTS`. `rank`: `EXTENT=((-1,5),(-2,2))`, `RES=0.05`,
  `ROBOT_RADIUS_M=0.22`.

## Callers

`../episode.py` imports `plan_stance`, `StanceIK`, `Stance`, `tucked_arm`, `visible_fraction`,
`box_surface_points`; `scripts/loco_manip_episode.py` drives it. Tests: `tests/smoke/test_plan_stance.py`,
`tests/unit/test_plan_ik.py`, `tests/unit/test_plan_visibility.py`.

## Gotchas

- `rank.py` uses `robot.head_camera.HeadCamera` (not the one in `reconstruction/align/camera_fit.py`).
- `set_stance` leaves legs at the model default; the real posture is refined at arrival by the executor.
- IK is stationary: feet, base and waist are fixed per candidate; only arm joints are solved.
- Hard constraints (IK, no forbidden contact, visibility) run before the weighted cost.
