# `src/stages/manipulation/` — physics replay and pick-place judgement

Physics replay of the retargeted motion in MuJoCo [[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco)] and the success judgement: scene construction, joint-target
control, seeded rollouts, grasp-lift-translate-place events, QA checks, tactile readout and the sim bundle
used by the RSI loop. See [docs/TECH_DESIGN.md §4.8, §4.11 and §6 Simulation model](../../../docs/TECH_DESIGN.md).
`__init__.py` holds only a docstring.

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `mjcf_scene.py` | build the scene: fixed-pelvis G1 + hand at `T_sim_robot_base`, table, rigid objects, D435 [[Intel RealSense](../../../docs/REFERENCES.md#realsense-d435)] | `SceneConfig` (`T_sim_robot_base`, `table_top_z_m`, `table_center_xy_m`, `table_half_size_xy_m`, ..., `from_params`); `BoxObject(name, half_size_m, pos_m, mass_kg=0.1, friction=(1, 0.01, 0.001), rgba, mesh_*...)`; `build_scene_spec(cfg) -> MjSpec`; `build_scene(cfg) -> MjModel` (LRU cache `MODEL_CACHE_SIZE=16`); `add_object(spec, obj, contact=None)`; `mesh_shape`; `ObjectContact`, `PDGains`, `PDConfig` (`from_params`) |
| `controller.py` | linear-interpolated joint target clipped to limits | `PositionController` (`q_target(t_s)`, `command(t_s)`) |
| `rollout.py` | action-controlled rollout: joint targets in, achieved states out | `run_rollout(cfg, timestamps_s, q_target_rad, seed=0, perturbation=None, record_rate_hz=50.0, settle_tail_s=0.5, settle_max_s=1.5, on_record=None, tactile=...) -> RolloutResult`; `Perturbation(object_xy_m=0.005, friction_scale=0.2, mass_scale=0.2)` (`apply(cfg, seed)`); `RolloutResult(arrays, physics_step_count, seed, contact_steps_with_object, state_injection_after_reset, arm_collision_steps, tactile)` |
| `guard.py` | anti-cheat: object state only changes via `mj_step` | `StateInjectionGuard` (`snapshot(data)`, `check(data)`; `.violations`) |
| `events.py` | grasp-lift-translate-place detection on object trajectory | `detect_pick_place(timestamps_s, T_world_object (T,4,4), th=None, place_target_pos_m=None, expected_translation_m=None, hand_contact=None, release_t_s=None) -> PickPlaceEvents`; `EventThresholds(lift_min_m=0.05, translate_min_m=0.1, place_height_tol_m=0.02, ...)` |
| `checks.py` | finite / joint-limit / tracking / speed checks + events | `check_rollout(arrays, joint_limits_rad, thresholds=None, event_thresholds=None, object_index=0) -> CheckResult(passed, error_codes, metrics)`; `CheckThresholds(joint_limit_tol_rad=0.02, max_tracking_rms_rad=0.25, max_joint_speed_rad_s=30, max_object_speed_m_s=5)` |
| `engine.py` | seeded multi-rollout replay, QA, bundle writing | `replay_episode(cfg, timestamps_s, q_target_rad, out_dir, episode_id, n_seeds=5, render=True, perturbation=None, input_source="synthetic") -> ReplayOutcome(bundle, report, results)` |
| `sim_eval.py` | re-simulate a planned trajectory; start-offset search | `SimCase` (`config(params, objects)`, `held_at(xy)`); `offset_grid(start_search_m)` (no-offset + 5x5 grid); `start_search(case, params, offsets_m, extra=None) -> SearchResult`; `pick_place_events`; `sim_tier_extra`; `held_physics_kw` |
| `bundle.py` | frozen plan + refs as npz+json (`sim_bundle/1`) | `save_bundle(out_dir, case, params, *, clip, side, best_start_xy_m, phases_t_s, refs=None, extra_meta=None)`; `load_bundle(out_dir) -> SimBundle` (`.params()`, `.best_offset_m()`) |
| `probes.py` | per-frame contact/motion probes via `on_record` | `ContactProbe` (penetration, contact force, object speed; `.arrays()`) |
| `sim_tactile.py` | per-region sim tactile in the EgoTac [[Zhang et al., 2026](../../../docs/REFERENCES.md#zhang2026egotac)] region schema | `detect_hand(model) -> "dex3"\|"sharpa"`; `SimTactile` (`sample`, `update`, `arrays`, `save`); `support_touch(arrays, timestamps_s, side) -> dict\|None` |
| `render.py` | headless MP4 rendering | `VideoRecorder` (`frames()`, `close()`); `select_gl_backend()` |
| `scripted.py` | scripted right-hand top-down pinch pick-and-place (synthetic input) | `scripted_pick_place(cfg, cube_center_m, delta_xy_m=(0.0, -0.12), hover_m=0.12, clearance_m=0.003) -> (timestamps_s (M,), q_target_rad (M, 43))` |

## Inputs / outputs

- `q_target_rad` is `(M, 43)` in the authoritative joint/actuator order (legs, waist, arms, then hands;
  right-hand finger slice `36:43`), timestamps in seconds; replay runs at 50 Hz (`CONTROL_RATE_HZ`).
- `RolloutResult.arrays`: `timestamps_s`, `q_command_rad`, `q_actual_rad`, `dq_actual_rad_s`,
  `wrist_actual_T_sim`, `object_T_sim (T, n_obj, 4, 4)`, `object_robot_contact (T, n_obj)` bool.
- `replay_episode` writes into `out_dir`: `actual_states.npz`, `manifest.json`, `qa_report.json`, optional `render.mp4`.
- Held object in sim-eval is named `HELD_NAME = "perceived"`.

## Callers

`scripts/retarget_estimated.py` (mjcf_scene, rollout, events, sim_eval, bundle); `scripts/review_render.py`
(mjcf_scene, rollout, render, sim_tactile); `scripts/record_sim_state.py`, `preview_object_mesh.py`;
`src/orchestration/local_runner.py` (engine, scripted); `src/rsi/backends/mujoco.py` (bundle, sim_eval, probes,
render, rollout); `src/cli/main.py`; `src/robot/head_camera.py`; `skills/body_state.py`; `stages/qa/traj_panel.py`.
`checks.py`, `controller.py` and `guard.py` are used internally (rollout/engine).

## Constants and env vars

`CONTACT_IMPRATIO=10.0`; `HEAVY_HAND_WRIST_SCALE = {"dex3": 1.0, "sharpa": 2.5}`; D435:
`D435_PARENT_BODY="torso_link"`, `D435_POS_IN_TORSO_M=(0.0576, 0.0175, 0.4299)`, `D435_PITCH_RAD≈0.8308`,
`D435_IMAGE_HW=(360, 640)`; `CONTACT_FORCE_THRESHOLD_N=0.05` (sim_tactile); `BUNDLE_VERSION="sim_bundle/1"`.
Env: `MUJOCO_GL` (EGL default, set `osmesa` before import to override).

## Gotchas

- Sim tactile hand axis order is (left, right); Dex3 has no ring/pinky, so those regions are always zero
  (`has_region=False`). `sharpa` is detected by its palm bodies.
- Bundles are saved with `allow_pickle=False`; they replace the old `E2H_DUMP` pickle.
- `state_injection_after_reset=True` marks a violated anti-cheat guard; treat the rollout as invalid.
- Object contact uses geom priority and contact bits (`_OBJECT_BITS`, `_ROBOT_EXEMPT_BITS`).
- `build_scene` compiles the robot (~0.7 s) once per config and caches it; each caller gets a deep copy, so
  mutating the returned model is safe.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-realsense-d435"></span>[Intel RealSense](../../../docs/REFERENCES.md#realsense-d435). *Intel RealSense Depth Camera D435.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../../../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
