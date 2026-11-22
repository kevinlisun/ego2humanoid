# `src/stages/retargeting/` — human hand motion to robot joint targets

Turns estimated hand landmarks (camera frame, `(N, 21, 3)` per hand, OpenPose [[Simon et al., 2017](../../../docs/REFERENCES.md#simon2017hand)]/MANO order) plus gravity into
50 Hz joint targets of the fixed-pelvis G1 with Dex3 (43 DoF) or Sharpa Wave [[Sharpa](../../../docs/REFERENCES.md#sharpa-wave)] (73 DoF), and decides when the
grip closes and opens. Method, constants and rationale: `docs/TECH_DESIGN.md` §4.1–§4.9 (frame/placement §4.1,
kinematic §4.2, grip timing §4.3, pinch gap §4.5, hand-eye §4.7, outputs §4.9). Robot-side math (IK, pinch
solvers, finger maps, collision) is in `src/robot/` ([README](../../robot/README.md)). For the concepts and the
step-by-step flow, read [`docs/RETARGETING.md`](../../../docs/RETARGETING.md) first.

**Entry points:** `scripts/retarget_estimated.py` (everything except the two items below) and
`scripts/solve_sharpa_pinch.py` (`sharpa_pinch`). `stages/locomotion/{clip_task,episode}.py` import
`sharpa_pinch`. `spider_convert` and `tactile_target` are only used by tests.

## Modules

| File | Responsibility | Key public API |
|---|---|---|
| `kinematic.py` | Per-frame hand → 43/73-DoF targets: wrist IK, finger curls, pinch gap, pinch leveling/roll, branch merge | `retarget_hand(landmarks_sim, side, T_sim_robot_base=None, pinch_gap_m=None, wrist_offset_m=None, hand="dex3", waist_pitch_rad=0.0, pinch_roll_rad=None, cache=None) -> RetargetResult(q_target_rad, pos_err_m, rot_err_rad, ik_converged)`; `hand_frame(lm, side, hand)`; `finger_curls(lm)`; `turn_rotation`, `level_rotation`, `axis_rotation`; `held_frames(gaps)`; `rate_limited_share`; `landmarks_camera_to_sim(lm, T_sim_camera)`; `robot_pinch_axis_xy`, `commanded_pinch_axis_xy`; `SIDE_SLICE = {right: 36:43, left: 22:29}` |
| `clip.py` | Clip level: gravity alignment, placement, support lift, arm-collision avoidance, table-clearance loop, resample to 50 Hz | `retarget_clip(landmarks_camera, timestamps_s, up_in_camera, side, table_top_z_m=0.80, pinch_gap_m=None, placement=None, offset_m=None, passive_landmarks_camera=None, hand="dex3", pinch_uv_ref=None, support=None, clearance_exempt=None, obstacles=(), passive_obstacles=None, place_goal=None) -> ClipRetarget(timestamps_s (M,), q_target_rad (M, DoF), placement, ik, ik_passive)` |
| `hand_eye.py` | Closed-loop correction of the robot's own pinch point onto the object / receptacle | `pinch_track(q_rad, side, hand, T_sim_robot_base)`; `align_pinch(retarget, measure, target_pinch_m, weight, max_iters=4, tol_m=...) -> (result, offset, err)`; `phase_windows`, `grasp_window`, `blend_corrections`; `settle_approach`; `limit_carry_acceleration`; `release_motion` |
| `grip.py` | Size the squeeze of a position-controlled pinch from object mass/friction and finger stiffness | `required_grip_force_n(mass_kg, mu, tan_alpha=0, accel_m_s2=3.0, safety=2.0, min_force_n=4.0)`; `squeeze_for_force(force_n, stiffness_n_m, min_squeeze_m, max_squeeze_m=0.02, allowance_m=0.002)`; `actuated_stiffness(model)`; `pinch_stiffness_n_m(...)`; `hold_pinch_stiffness(q_full, side, hand, gap_m)` |
| `grip_timing.py` | Move the grip window to where the pinch is on the object (EgoTac [[Zhang et al., 2026](../../../docs/REFERENCES.md#zhang2026egotac)] span + lift onset) | `refine_grip_window(clip, t, first, side_i, human_pinch, hold_k, held, close_lead, min_pick_rise_m, reach_tol_m) -> (tactile_timing, hold_k, held)`; reads `<clip>/tactile_egotac.npz` |
| `tactile_grasp.py` | Contact span and lift onset from human tactile | `pinch_contact(contact_prob, side_index)`, `contact_spans`, `contact_span`, `pick_span`, `grasp_onset(pinch_z, lo, hi)`, `tactile_release_index`; `PAD_VERTICES`, `CONTACT_PROB = 0.25`, `SMOOTH_FRAMES = 5` |
| `support_clearance.py` | Keep hands and the held object above table/obstacle surfaces | `Support(top_z_m, lo_xy_m, hi_xy_m)` (`from_table`, `from_points`); `lift_above_support(landmarks_sim, support(s), exempt=None)`; `support_lift`, `exempt_weight`, `held_object_lift`; `CLEARANCE_M = 0.015` |
| `contact.py` | Per-frame hand-object contact flags and the `alignment/` files | `contact_flags(landmarks_sim, object_points_sim, threshold_m=0.02) -> {thumb, index, pinch}` bool arrays (`pinch = thumb & index`); `write_alignment_bundle(out_dir, T_sim_camera, table_top_z_m, scale, flags=None)` → `alignment/alignment.json` + `contact_flags.npz` |
| `held_mesh.py` | Held-object mesh/box asset for the physics scene | `scene_held_mesh(clip)`, `resolve_held_asset(clip, allow_box)`, `build_held_mesh(clip, held, lm_hold, side, hand)`; `HeldMeshMissing` (code `HELD_MESH_MISSING`) |
| `script_helpers.py` | Support/obstacle geometry and place-target resolution for the script | `object_supports(obj, margin_m)`, `object_points(obj)`, `resolve_place_target(...)`, `placement_retarget(clip, hand, lm, t, up, side, passive, pinch_uv_ref, adjust, place_goal=None)` |
| `script_params.py` | Module-level constants of the script, resolved from `SimParams` (import-time side effects) | `SIM_PARAMS`, `TP`, `OFFSETS_M`, ~60 constants such as `SQUEEZE_M`, `APPROACH_FRAMES`, `START_SEARCH_M`, `TABLE_MARGIN_M`, `MIN_SCORE = 0.7` |
| `sharpa_pinch.py` | Stored Sharpa thumb-index pad-pinch joint sets (open/touch/squeeze) | `PinchPoses.load()` (`.object_width_m`, `.pose`); `finger_pose(kind, side="right")`; `solve_pinch(model, object_width_m, side)`; `POSES_PATH = configs/grasp/sharpa_pinch_v1.json`, `PINCH_POINT_WRIST` |
| `tactile_target.py` | Human per-region tactile → Dex3 target; sim-vs-target comparison | `build_tactile_target(region_names, contact_prob, human_force=None, human_force_max_n=None, force_max_n=10.0)`; `compare_tactile(sim, target)`; arrays `(N, 2, R)` |
| `spider_convert.py` | Our bundles → SPIDER [[Pan et al., 2025](../../../docs/REFERENCES.md#pan2025spider)] processed layout | `convert_clip(dataset_dir, task, data_id, ref_dt_s, wrist_T_sim, landmarks_sim, object_T_sim=None, ...)` writes `processed/ego2humanoid/mano/<embodiment>/<task>/<id>/trajectory_keypoints.npz` + `task_info.json`; `mat_to_pose7` = `[x y z qw qx qy qz]` |

## Frames, shapes, units

- `landmarks_camera`: `(N, 21, 3)` m, OpenCV camera frame of the source video. `landmarks_sim`: same, simulation
  frame (z up, table top at `table_top_z_m`, default 0.80). `T_sim_camera` comes from gravity alignment
  (`reconstruction/align/frames.py`) plus table anchoring.
- `q_target_rad`: actuator order; right hand `36:43`, left `22:29` for Dex3. `CONTROL_RATE_HZ = 50`.
- `pinch_gap_m`: `(N,)` commanded thumb–index tip gap; `offset_m`: `(N, 3)` per-frame wrist displacement added by
  hand-eye correction; `PINCH_TOL_M = 0.005`, `MAX_OFFSET_M = 0.3`, `MAX_ITERS = 4` (hand_eye).
- Outputs written by the script (not by this package, except `alignment/`): `retarget_rgb{,_sharpa}.{json,npz}`
  (`timestamps_s`, `q_target_rad (M, 43|73)`), `alignment/*` (`docs/TECH_DESIGN.md` §4.9).

## Important constants and environment variables

- `kinematic.py`: `RESTART_POS_ERR_M = 0.01`, `PINCH_TUCK_CURL = 0.9`, `LEVEL_RAMP_FRAMES = 10`,
  `TURN_FALLBACK_FRACS = (1, 0.5, 0.25, 0)`, `TURN_FRAC_RATE = 0.05`, `PELVIS_HEIGHT_M = 0.793`.
- `clip.py`: `MAX_AVOID_ITERS = 16`, `AVOID_STEP_M = 0.02`, `MAX_CLEARANCE_ITERS = 3`, `ROLL_LEVER_M = 0.06`,
  `MAX_PINCH_ROLL_RAD = 60°`, `HAND_CLEARANCE_M = {dex3: 0.0, sharpa: 0.03}`.
- `HAND` (`dex3` | `sharpa`; read in `script_params` for the accepted-params path, in the script for the model),
  `E2H_SIM_PARAMS` (SimParams overlay, via `skills.params.resolve`), `SCENE_DIR` (scene `objects.json` for
  place-target resolution), `E2H_RETARGET_CACHE=<dir>` (pickle of the placement search per clip and hand),
  `E2H_TACTILE_RELEASE=1` (release at the tactile contact end, `grip_timing`), `E2H_HELD_MUG=1` (hold a procedural
  mug instead of the generated slab, `held_mesh`).

## Gotchas

- `script_params` runs `resolve_params(accepted=accepted_path(sys.argv[1], $HAND))` **at import**: importing it
  from anything but `retarget_estimated.py` reads `sys.argv[1]` as a clip path and the environment.
- `retarget_hand(cache=...)` is a dict owned by the caller; frames before the first changed input are reused.
- `placement_retarget` pickles the cache and skips it when `place_goal` is set (second pass of the two-pass place
  search); older caches stored the bare retarget instead of a tuple.
- `hand_frame` follows `RIGHT_Y_FLIP[hand]` (both `False` now): a wrong sign rolls that hand 180° about its axis.
- While a Sharpa hand pinches, middle/ring/pinky tuck at curl `PINCH_TUCK_CURL = 0.9`.
- `kinematic.py` (490 lines) and `hand_eye.py` (294) are the largest modules; keep new logic in separate files.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-pan2025spider"></span>[Pan et al., 2025](../../../docs/REFERENCES.md#pan2025spider). *SPIDER: Scalable Physics-Informed Dexterous Retargeting.*
- <span id="cite-sharpa-wave"></span>[Sharpa](../../../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
- <span id="cite-simon2017hand"></span>[Simon et al., 2017](../../../docs/REFERENCES.md#simon2017hand). *Hand Keypoint Detection in Single Images using Multiview Bootstrapping.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../../../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
