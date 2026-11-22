# `src/stages/` — stage logic

Everything that turns pipeline data into pipeline data. The real-clip scripts (`scripts/*.py`) import these
packages; the contract path uses `adapters.py`. Pipeline overview: `docs/TECH_DESIGN.md` §2; layout: §7.
Per-subpackage READMEs: [retargeting](retargeting/README.md), [reconstruction](reconstruction/README.md), [manipulation](manipulation/README.md), [locomotion](locomotion/README.md), [qa](qa/README.md); short summaries below.

## Top-level modules

| File | Responsibility |
|---|---|
| `__init__.py` | Sets `os.environ.setdefault("MUJOCO_GL", "egl")` so MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)] renders headless (same snippet as `robot/__init__.py`) |
| `adapters.py` | Third-party seams of the contract path: `FAHAdapter.estimate_hands(episode, video_localpath, output_dir) -> HandTrackBundle`, `SimFoundryAdapter.reconstruct_scene`, `ObjectMotionAdapter.track_objects`, `RegistrationAdapter.align`, `SPIDERAdapter.retarget`, `OmniGibsonReplayAdapter.simulate`, `PhysicsQAAdapter.validate`. **All raise `NotImplementedError`** (messages cite M1–M4); only `cli/main.py` (`hand`, `scene`) calls two of them |

## Subpackages

| Package | Role | Main callers |
|---|---|---|
| `reconstruction/` | Perception post-processing and scene: `align/`, `hands/`, `objects/`, `scene/` | `scripts/{estimate_camera_poses,perceive_objects_sam3,perceive_scene_objects,crop_object_sam3,build_bg_splat,run_simfoundry_clip,twin_import,twin_appearance_rsi,preview_object_mesh,make_review_grid,review_render,retarget_estimated}.py` |
| `retargeting/` | Hand landmarks → 43/73-DoF joint targets, grip timing, hand-eye correction | `scripts/retarget_estimated.py`, `scripts/solve_sharpa_pinch.py` |
| `manipulation/` | MuJoCo scene build, action-controlled rollout, event detection, re-simulation | `scripts/{retarget_estimated,record_sim_state,review_render,preview_object_mesh}.py`, `cli`, `orchestration` |
| `locomotion/` | Stance planning, nav route, DWBC walking executor, tabletop scenario (Track L) | `scripts/{loco_manip_episode,render_scene_preview,retarget_estimated,solve_sharpa_pinch}.py` |
| `qa/` | Review rendering helpers (2×2 grid, trajectory panel, hand overlay, review paths) | `scripts/{make_review_grid,render_side_by_side,review_render,retarget_estimated}.py` |

### `reconstruction/`

- `align/` — `frames.py` (camera → simulation frame: gravity + table anchoring), `camera_fit.py` (robot
  head-camera tilt so hand and manipulation stay in view), `table_fit.py` (synthetic table fitted to the robot;
  `table_penetration`), `ego_vo.py` (monocular visual odometry for walking clips).
- `hands/` — `backend.py` (backend choice from `configs/hands.yaml`), `smooth.py` (gap fill, outlier rejection,
  smoothing of landmarks), `tactile.py` (MANO-vertex tactile → per-region force/contact, smoothing, fusion with
  geometry).
- `objects/` — `prompt.py` (SAM 3 [[Carion et al., 2026](../../docs/REFERENCES.md#carion2026sam3)] text prompts from the task sentence), `crop_select.py`, `fuse.py` (object
  candidates in the MoGe [[Wang et al., 2025b](../../docs/REFERENCES.md#wang2025moge2)] frame fused with FAH [[Cao, 2026](../../docs/REFERENCES.md#cao2026fah)] hands), `grasp_proxy.py` (grasp detection + proxy box),
  `mesh_asset.py` / `mesh_texture.py` / `mug_mesh.py` (generated mesh → table-resting sim asset, texture,
  procedural mug), `physics_params.py` (mass/friction), `place_target.py` (receptacle and place pose),
  `hf_weights.py` (HF cache preflight), `twin_schema.py` / `twin_import.py` / `twin_asset.py` (Blender [[Blender, 2024](../../docs/REFERENCES.md#blender)]
  digital-twin asset package, Track BT).
- `scene/` — `inventory.py` (every tabletop object as a sim body in the retargeting frame; 967 lines, near the
  1000-line limit), `roles.py` (held / source / receptacle), `export_mjcf.py`, `collision.py` (CoACD [[Wei et al., 2022](../../docs/REFERENCES.md#wei2022coacd)]),
  `scene_state.py` (settle report), `bg_splat.py` / `neural_ego.py` / `composite.py` (Gaussian background and
  ego-view rendering), `removal_quality.py`, `simfoundry_runner.py` + `vlm_relay.py` + `relay_hook/` (Copilot
  relay standing in for the Gemini VLM client; `docs/COPILOT_RELAY.md`), `viz.py`.

### `manipulation/`

`mjcf_scene.py` (`SceneConfig`, `BoxObject`, `build_scene`; fixed-pelvis robot at `T_sim_robot_base`, table,
objects, D435 [[Intel RealSense](../../docs/REFERENCES.md#realsense-d435)] camera), `rollout.py` (`run_rollout`, `Perturbation`, `RolloutResult`), `engine.py`
(`replay_episode(cfg, timestamps_s, q_target_rad, out_dir, episode_id, n_seeds=5, render=True, ...)` writes the
`SimRunBundle`), `controller.py` (`PositionController`), `guard.py` (`StateInjectionGuard`: object state changes
only through physics), `events.py` (`detect_pick_place`, `EventThresholds`), `checks.py` (sanity checks of
`actual.npz`), `sim_eval.py` (`SimCase`, `start_search`, `offset_grid`, `pick_place_events`), `bundle.py` (frozen
plan + references for Track RSI), `probes.py` (contact/motion probes via `on_record`), `sim_tactile.py` (per-region
pad force), `render.py` (`VideoRecorder`, `select_gl_backend`: EGL, OSMesa fallback), `scripted.py`
(`scripted_pick_place`).

### `locomotion/`

`base_placement.py`, `stance_adjust.py` (move the robot only when camera/arm cannot do the job), `route.py` (grid
A*), `executor.py` (waypoint pursuit on DWBC), `episode.py` (approach episode), `clip_task.py` (retargeted clip →
loco-manip task), `tabletop.py` (scenario), `video.py`; `plan/` = `candidates`, `ik`, `visibility`, `validators`,
`rank` (→ `PoseGoal`), `golden`.

### `qa/`

`grid_video.py` (2×2: perception, simulation, scene render, task-space paths), `traj_panel.py`, `hand_overlay.py`
(estimated hands + tactile on a frame), `review_paths.py` (where review artefacts live, stale ones removed).

## Cross-package imports (not a strict layering)

Main flow `reconstruction → retargeting → manipulation → qa`, but `retargeting` imports
`locomotion.{base_placement,stance_adjust}`, `manipulation.{mjcf_scene,sim_eval}`, `reconstruction.{align,objects,scene}`
and `skills.params.resolve`; `locomotion` and `manipulation` import `robot`, `contracts`, `common`.
`rsi/` and `skills/` drive `manipulation.sim_eval` through `contracts.sim_backend`.

## Gotchas

- In the real-clip path stages exchange per-clip files (`objects_sam3.npz`, `tactile_egotac.npz`,
  `camera_poses.npz`, `scene/inventory.json`, ...), not bundle manifests.
- Some docstrings/configs still say `stages/replay/`; the package is now `stages/manipulation/`.
- Hard limit 1000 lines per file; `scene/inventory.py` is at 967.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-blender"></span>[Blender, 2024](../../docs/REFERENCES.md#blender). *Blender — a 3D modelling and rendering package.*
- <span id="cite-cao2026fah"></span>[Cao, 2026](../../docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](../../docs/REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-realsense-d435"></span>[Intel RealSense](../../docs/REFERENCES.md#realsense-d435). *Intel RealSense Depth Camera D435.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](../../docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-wei2022coacd"></span>[Wei et al., 2022](../../docs/REFERENCES.md#wei2022coacd). *Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search (CoACD).*
