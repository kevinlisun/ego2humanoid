# `src/stages/reconstruction/scene/` — tabletop inventory and scene export

Everything about the *tabletop around the manipulated object*: the per-clip object inventory registered to
the sim frame, task roles (held / source / receptacle), MJCF-ready object export, the background Gaussian
splat, ego-view compositing, scene-state recording, and the SimFoundry [[NVIDIA GEAR, 2026](../../../../docs/REFERENCES.md#simfoundry2026)] runner with its file-based VLM relay.
See [docs/TECH_DESIGN.md §4.10](../../../../docs/TECH_DESIGN.md) and §5 (review rendering).

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `inventory.py` | inventory of all tabletop objects -> `BoxObject`s in the sim frame (scale, registration, footprint, heightmaps, collision) | `load_inventory(clip) -> Inventory \| None`; `build_scene_inventory(inv, scale, T_sim_camera, table_z, held_xy, held_phrase, place_phrase, source_phrase, table_near_x, place_hint_xy=None, held_half_xy=None, ...) -> SceneInventory`; `footprint(xy, height) -> Footprint`; `fit_vertices`, `orient_quarter`, `body_heightmap`, `separate`, `scene_extent_xy(objects)`, `disturbance(object_T_sim, names, skip)`, `depth_ratio` |
| `roles.py` | which inventory object is held / source / receptacle | `PhraseMatcher`, `find_held(...) -> int\|None`, `pick_source(...)`, `pick_receptacle(...)`, `source_shift(fp, held_xy, held_half_xy)`, `colour_name(rgb)`, `is_container(label)`, `inside`, `clamp_inside` |
| `export_mjcf.py` | `<scene_dir>/objects.json` (+ GLBs) -> sim objects | `load_scene_objects(scene_dir, decompose=False) -> tuple[BoxObject, ...]` (body-frame meshes, bbox-centre origin) |
| `collision.py` | CoACD [[Wei et al., 2022](../../../../docs/REFERENCES.md#wei2022coacd)] convex decomposition | `convex_pieces(vertices, faces, threshold=0.05, max_pieces=16, seed=0)`; falls back to the hull for non-watertight meshes |
| `scene_state.py` | per-frame object pose/velocity/contact record + settle gate | `SceneState` (`.save`), `load_scene_state(path)`, `record_scene_state(cfg, duration_s=1.0, fps=30.0)`, `settle_report(state, max_drift_m=0.01, max_speed_mps=0.02, max_penetration_m=0.005)` |
| `bg_splat.py` | build (not train) a background splat in the sim frame from DA3 + FLUX [[Black Forest Labs, 2025](../../../../docs/REFERENCES.md#batifol2025flux)] clean plate | `build_background_splat(scene_dir, clip_dir, upsample=3, disc_thickness=0.2, radius_scale=0.75, foreground_thresh=35.0) -> (BackgroundSplat, report)`; `BackgroundSplat.save/load`; `removal_mask`, `match_colors`, `surface_labels`, `fill_smooth` |
| `neural_ego.py` | MuJoCo [[Todorov et al., 2012](../../../../docs/REFERENCES.md#todorov2012mujoco)] foreground over rasterised splat, by depth | `EgoNeuralRecorder` (`on_record`, `.save`, `.close`); `composite_neural_ego(ego_npz, bg_npz, ghost_alpha=0.45, hole_fill_px=6) -> list[frames]` |
| `composite.py` | render the robot from an ego camera over a clean plate | `free_camera(T_sim_camera, distance=1.0)`, `CompositeRecorder`, `write_side_by_side(video_path, frames, out_path, t0_s, fps=50.0, src_fps=30.0, labels=...)` |
| `removal_quality.py` | SimFoundry stage-5 removal quality gate | `removal_metrics(before, after, mask, dilate_px=15)`, `passes(m)`, `build_report(s5)` |
| `viz.py` | captioned slideshow MP4s for reviewing sub-modules | `slideshow_frames(...)`, `write_slideshow(path, slides, panel_hw=(480, 640), fps=10, hold_s=1.5)` |
| `simfoundry_runner.py` | run SimFoundry pipeline A on a clip, VLM via relay | `build_command(simfoundry_dir, scene_name, root_dir, video, stages=STAGES, extra_overrides=DEFAULT_OVERRIDES, stream=True)`; `run_round(...) -> (exit_code, pending_count)` |
| `vlm_relay.py` | file-based Copilot relay replacing the Gemini client | `RelayClient` (`factory`, `answer`), `RelayPending` (BaseException), `request_id`, `relay_status`, `validate_inbox`, `submit_request`, `read_answer`; `PENDING_RELAY_EXIT_CODE = 75` |
| `relay_hook/` | PYTHONPATH hook that installs the relay | see [relay_hook](relay_hook/README.md) |

## Inputs / outputs

- Inventory (`<clip>/scene/`): `inventory.json` + `points.npz` (camera-frame, unscaled MoGe [[Wang et al., 2025b](../../../../docs/REFERENCES.md#wang2025moge2)] points per item),
  crops and optional meshes; produced by `scripts/perceive_scene_objects.py` and `generate_object_mesh.py --scene`.
- `objects.json`: `{"objects": [{"name", "glb", "T_sim_object" (4x4, metres, z-up), "mass_kg", "friction"}]}`.
- Background splat: `BackgroundSplat` fields `means (N,3)`, `quats (N,4)` wxyz, `scales (N,3)`, `opacities`,
  `colors`; reads `s2_da/da/exports/npz/results.npz` and `<clip>/alignment/alignment.json`.
- Relay layout: `<root>/<scene>/relay/outbox/<id>/{prompt.md,request.json,images/}` and `inbox/<id>.json`
  (`{"answered_by","text"}`).

## Callers

`scripts/retarget_estimated.py`, `generate_object_mesh.py` (inventory); `build_bg_splat.py`; `review_render.py`,
`make_review_grid.py` (composite/neural_ego/export_mjcf); `run_simfoundry_clip.py` (runner);
`viz_stage5_removal.py`, `viz_stage6_upsample.py` (viz); `twin_appearance_rsi.py`, `src/rsi/loop.py`,
`src/cli/main.py` (relay, scene_state, export_mjcf); `src/contracts/sim_params.py` (roles).

## Constants and env vars

- `inventory.py`: `HELD_MATCH_M = MAX_REGISTRATION_M = 0.15` (overridden by `$E2H_HELD_MATCH_M`),
  `WORKSPACE_RADIUS_M=0.55`, `MIN_SIZE_M=0.012`, `DISTURBED_M=0.03`, `NEAR_EDGE_MARGIN_M=0.02`, `HMAP_CELL_M=0.015`.
- `composite.py`/`neural_ego.py`: geom groups `HIDDEN_GEOM_GROUP=5`, `COLLISION_GEOM_GROUP=3`;
  `GHOST_BODIES = (pelvis, hip, knee, ankle, waist, torso)`.
- `simfoundry_runner.py`: `STAGES = ('1b','2','3','4','5','6','7','8','10','11','12')`; sets `E2H_RELAY_DIR`,
  prepends `relay_hook/` to `PYTHONPATH`, `CACHE_MODE=1`, optional `CONDA_ENVS_DIRS`, `HF_HOME`.

## Gotchas

- Exit code 75 from `run_round` means VLM requests await answers: answer into `inbox/`, then re-run.
- `RelayPending` derives from `BaseException` so retry loops catching `Exception` cannot swallow it.
- gsplat rasterisation needs CUDA and another environment than the main one (`neural_ego.py` docs).
- Monocular depth scale comes from the median depth ratio against `objects_sam3.npz`, not metric truth.
- The whole layout is translated so the inventory's held object sits at the grasp point; relative layout kept.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../../docs/REFERENCES.md).

- <span id="cite-batifol2025flux"></span>[Black Forest Labs, 2025](../../../../docs/REFERENCES.md#batifol2025flux). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing.*
- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](../../../../docs/REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](../../../../docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-wei2022coacd"></span>[Wei et al., 2022](../../../../docs/REFERENCES.md#wei2022coacd). *Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search (CoACD).*
