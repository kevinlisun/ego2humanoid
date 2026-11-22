# `src/skills/` — reusable real-to-sim feedback skills

Pure-numpy detectors and checks that turn one rollout into `Real2SimFeedback` (events, residuals, match score,
hints, `feedback.md`). Most modules avoid importing MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)] (the package `__init__` only sets
`MUJOCO_GL=egl` by default). Caller: [`../rsi/runner.py`](../rsi/README.md) via `real2sim_report.build_feedback`.
Parameter handling: [`params/`](params/README.md). Registry/docs: `library.py`, `docs/notes/skills_library.md`.

## Modules

| File | Responsibility | Key API |
|---|---|---|
| `real2sim_report.py` | Top-level builder of `feedback.json` + `feedback.md` | `FeedbackInputs` (dataclass: clip, hand, iter, params, arrays, probe, pick_place, phases, refs, receptacle, reach, ...); `build_feedback(inp) -> Real2SimFeedback`; `write_feedback(out_dir, fb, changed_vs_parent=None) -> (json_path, md_path)` |
| `feedback.py` | Event detectors on rollout + `ContactProbe` arrays (thresholds are module constants, not `SimParams`) | `EventInputs` dataclass; `detect_events(x) -> list[FeedbackEvent]` (merged, sorted, capped at `MAX_EVENTS=40`); `phase_at`, `contact_runs`, `touch_times`, `grip_force_stats`, `merge_and_cap` |
| `residuals.py` | Sim-vs-video residuals and score | `SCALES`, `WEIGHTS`, `SEVERITY` dicts; `carry_path_m`, `place_err_m`, `held_track_m`, `tactile_residual`, `lift_ratio`, `timing_residual`; `match_score(residuals, success, failure_reasons) -> float` (fail: `1 + worst severity`; success: weighted mean of `min(r/scale, 1)`) |
| `attribution.py` | Failure pattern → suggested `SimParams` knobs (rule table `RULES`) | `fired_patterns(events, qa, residuals, phases, *, grip_ratio, grasp_sign_s, has_receptacle)`, `build_hints(params, events, qa, residuals, phases, ...) -> list[FeedbackHint]`, `param_exists`, `param_affects`, `param_range`, `get_param`; rules for missing paths are skipped |
| `feedback_md.py` | Renders `feedback.md` (≤ `MAX_LINES=60`) | `render_feedback_md(fb, changed_vs_parent=None) -> str` |
| `final_pose.py` | Final tilt/drift/height of the placed object | `final_pose(t, object_T_sim, bare_table) -> FinalPose \| None`; defects `FINAL_TILTED` (>15°), `FINAL_NOT_AT_REST` (>5 mm in last 0.5 s), `FINAL_HEIGHT` (>1 cm, bare table only) |
| `place_check.py` | Is the held mesh hidden in the receptacle mesh (ray along +z) | `buried_fraction(held_vertices, T_held, receptacle_vertices, receptacle_faces, T_receptacle)`, `buried_fraction_of(held, receptacle, T_held, T_receptacle)`, `object_mesh`, `box_mesh`; `BURIED_FRAC=0.3` fires `PLACE_BURIED` |
| `receptacle_check.py` | Footprint test of the rest position | `ReceptacleGeometry`, `check_place(end_pos_m, release_pos_m, held_half_z_m, g) -> PlaceCheck` (`PLACE_OUTSIDE_RECEPTACLE`, `RIM_BLOCKED`), `check_retry(record) -> RetryCheck \| None` (`RETRY_WORSE`, `RETRY_BUDGET_EXHAUSTED`) |
| `reachability_check.py` | Place target inside arm workspace | `window_unreachable_fraction(pos_err_m, timestamps_s, window_s)`, `reach_events(reach, t_s, phase, object_name)`; bars `UNREACHABLE_POS_M=0.03`, `PLACE_UNREACH_FRAC=0.2` |
| `scene_layout.py` | Registered layout vs demonstration | `layout_events(report) -> list[FeedbackEvent]` (`SCENE_ROLE_SWAP`, `SCENE_RECEPTACLE_REBASED`, `SCENE_GRIP_GAP`; `GRIP_GAP_M=0.08`, `REBASE_M=0.10`) |
| `asset_provenance.py` | Held-mesh source events | `provenance_events(held_asset, held_name)` → `HELD_MESH_FALLBACK_BOX` / `HELD_MESH_FROM_SCENE` |
| `library.py` | Data-only registry of skills (events, knobs, bug class) | `Skill`, `SKILLS`, `module_path(skill)`, `available(skill)` |
| `body_state.py` | Blender-replay state (needs `mujoco`) | `BodyStateRecorder(cameras=("d435","ego"))` (`on_record` hook; `.arrays()`, `.save(path)`), `export_sim_model(model, out_dir)` → `sim_model.json` + `sim_model_meshes.npz`; `STATE_VERSION="sim_state/2"` |
| `physics_fidelity.py` | Fidelity checks on a recorded `sim_state` | `fidelity_report(sim_state, objects, table_top_z_m, half_heights_m, held="perceived", max_sink_m=0.005, max_drift_m=0.01, max_rest_speed_mps=0.02, quat_tol=1e-3) -> dict`, `format_markdown`, `main`; `PAD_FORCE_MAX_N=8.2` |
| `twin_compare.py` | Real RGB vs rendered twin (0 identical, 1 different; needs `cv2`) | `hand_mask(joints2d, image_wh, dilate_px=25)`, `compare(real, twin, ignore=None, rois=None) -> dict`, `diff_image`, `side_by_side` |

## Used by

- `rsi/runner.py` → `real2sim_report`; `scripts/retarget_estimated.py` → `reachability_check`, `params.*`;
  `scripts/record_sim_state.py`, `scripts/review_render.py` → `body_state`; `scripts/twin_appearance_rsi.py` → `twin_compare`;
  `rsi/backends/mujoco.py` → `place_check`.
- `physics_fidelity` CLI: `python -m skills.physics_fidelity <clip_dir> [--tag _bt]` reads
  `<clip>/review/sim_state_sharpa<tag>.npz` and `review/sim_model/sim_model.json`, prints Markdown.

## Gotchas

- Residual weights/scales/severities and detector thresholds are code constants, not agent-tunable.
- Frames: `T_sim_*` z-up, metres, seconds. `FeedbackInputs.hand` indexes axis 1 of `wrist_actual_T_sim` (left = 0).
- `attribution.HELD_NAME="perceived"` is duplicated on purpose (importing the sim module needs MuJoCo).
- Tests: `tests/unit/test_real2sim_{report,feedback,attribution}.py`, `test_final_pose.py`, `test_place_check.py`,
  `test_receptacle_check.py`, `test_reachability_check.py`, `test_scene_layout.py`, `test_physics_fidelity.py`,
  `test_twin_compare.py`, `test_body_state.py`, `test_skills_library.py`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
