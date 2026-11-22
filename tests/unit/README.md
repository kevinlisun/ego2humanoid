# `tests/unit` — unit tests (64 files)

Fast, mostly synthetic-input tests of one module each. Run: `python -m pytest tests/unit -q` (setup, markers, skip rules:
[`../README.md`](../README.md)). No shared fixtures beyond pytest built-ins (`tmp_path`, `parametrize`) and
`../conftest.py` (`MUJOCO_GL=egl`). No test here needs a GPU. Real data is needed only by two files (marked below).

Legend: **M** = needs `mujoco` (`importorskip` or direct import); **A** = needs external assets/data (skips when absent).

## Real2sim loop (Track RSI) → [`src/rsi`](../../src/rsi/README.md), [`src/skills`](../../src/skills/README.md)

| File | Covers |
|---|---|
| `test_real2sim_loop_parts.py` | relay submit/read, `history` roundtrip + best entry, `validate_answer` (merge, fenced/dotted, invalid, repeat-of-history) |
| `test_real2sim_backend.py` | loop modules do not import a simulator (AST), unknown backend rejected, MuJoCo registered + `SimBackend` protocol, `robust_check` on a fake backend |
| `test_real2sim_report.py` | `build_feedback`, `render_feedback_md` |
| `test_real2sim_feedback.py` | event detectors (`detect_events`, `contact_runs`, ...) |
| `test_real2sim_attribution.py` | attribution rules / hints, residual match score |
| `test_scene_layout.py` | `layout_events` and its hints |
| `test_final_pose.py`, `test_place_check.py`, `test_receptacle_check.py`, `test_reachability_check.py` | final-pose, buried-in-receptacle, rim/outside + retry, unreachable-place checks |
| `test_held_asset.py` | `asset_provenance` events, held-mesh resolution (`HeldMeshMissing`) |
| `test_skills_library.py` | `SKILLS` registry consistency with `RULES` / existing modules |
| `test_physics_fidelity.py` | `fidelity_report` / `format_markdown` |
| `test_sim_params.py` | `SimParams` validation, `configs/sim_params/default.yaml` parity, resolve/diff/hash/`needs_plan`, scene config wiring (M) |
| `test_body_state.py` | `BodyStateRecorder`, `export_sim_model` (M) |
| `test_twin_compare.py` | real-vs-twin score, hand mask, diff/side-by-side |

## Contracts and common utilities

| File | Covers |
|---|---|
| `test_contracts.py`, `test_contracts_locomanip.py`, `test_twin_appearance.py`, `test_twin_schema.py` | pydantic contracts (bundles, QA report, loco-manip, twin appearance/asset schema) |
| `test_common_arrays.py` | `common.io` npz, `timebase` resampling, SE(3)/quaternion/Umeyama transforms |
| `test_synthetic_events.py` | synthetic arrays satisfy contracts; `detect_pick_place` events and failure reasons |
| `test_review_paths.py` | `stages.qa.review_paths` |

## Perception, alignment, reconstruction

| File | Covers |
|---|---|
| `test_egodex_reader.py` | EgoDex clip reader/export (`h5py`; real-sample tests need `data/raw/egodex/...` **A**) |
| `test_ego_vo.py`, `test_camera_fit.py` (M), `test_table_fit.py`, `test_retarget_align.py` (M), `test_twin_pose_parity.py` | ego visual odometry, head-camera fit, table fit, frame alignment/retarget kinematics, camera/pose math parity |
| `test_hand_eye.py`, `test_hand_metrics.py`, `test_hand_overlay.py`, `test_tactile.py`, `test_tactile_grasp.py`, `test_contact.py`, `test_grip.py` (M) | hand-eye pinch alignment, hand metrics, overlay/tactile colouring, tactile bundle, grasp timing from tactile, contact flags, grip sizing |
| `test_object_fuse.py`, `test_crop_select.py`, `test_grasp_proxy.py`, `test_place_target.py`, `test_scene_inventory.py`, `test_scene_state.py`, `test_removal_quality.py` | object fusion/prompts/mesh prep, crop selection, grasp proxy, place target, scene inventory + roles, scene state stability, background removal quality |
| `test_bg_splat.py`, `test_export_mjcf.py`, `test_twin_asset.py`, `test_scene_viz.py` | background splat compositing, scene → MJCF export, twin asset import, slideshow frames |
| `test_vlm_relay.py`, `test_relay_hook.py` | file-based VLM relay round trip / pending exit 75; SimFoundry VLM hook (needs `simfoundry` importable) |
| `test_spider_convert.py` | SPIDER clip conversion |

## Robot, locomotion, simulation (mostly M)

| File | Covers |
|---|---|
| `test_head_camera.py`, `test_collision.py`, `test_floating_hand.py`, `test_finger_map_checks.py` | head camera model, arm collision checker, floating-hand MJCF, finger mapping + rollout checks |
| `test_sharpa_hand.py` | Sharpa hand model (**A**: `scripts/fetch_sharpa_assets.sh`) |
| `test_sim_tactile.py` | simulated tactile regions |
| `test_scenario_tabletop.py`, `test_plan_ik.py`, `test_plan_visibility.py`, `test_stance_adjust.py`, `test_nav_route.py` | tabletop scenario, stance IK, visibility, stance adjust, route planning |
| `test_sim_bundle.py` | `load_bundle` roundtrip (**A**: `data/derived/egodex/basic_pick_place_104/real2sim_sharpa/sim_bundle.json`) |
| `test_review_video.py` | review grid video, trajectory panel, ego camera of a scene (M) |

## Evaluation

`test_hand_metrics.py` (see above) covers `evaluation.hand_metrics` (MPJPE, wrist position).

## Gotchas

- Keep new unit tests free of real data and GPU; use `tmp_path` and synthetic arrays.
- Tests that import `mujoco` directly (e.g. `test_body_state`, `test_export_mjcf`, `test_floating_hand`, `test_twin_asset`)
  need the `sim` extra; `importorskip` ones skip cleanly.
