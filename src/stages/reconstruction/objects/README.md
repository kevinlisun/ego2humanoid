# `src/stages/reconstruction/objects/` — object selection, assets and grasp geometry

Object-side reconstruction: prompts for the detectors, choosing the held object among per-frame candidates,
turning a generated mesh into a table-resting asset, grasp/place geometry, physics priors, and the
digital-twin asset package (schema, import, loader). See
[docs/TECH_DESIGN.md §3, §4.3, §4.4, §4.10](../../../../docs/TECH_DESIGN.md).

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `prompt.py` | SAM 3 [[Carion et al., 2026](../../../../docs/REFERENCES.md#carion2026sam3)] text prompts from the task description (regexes) | `object_prompt(task_description, default="object")`, `place_target_prompt(task_description) -> str\|None`, `source_prompt(task_description) -> str\|None` |
| `hf_weights.py` | preflight for cached Hugging Face weights | `hub_dir(env=None)`, `cached_model(repo_id, env=None) -> bool`, `require_model(repo_id, label, env=None)` (SystemExit `ERROR WEIGHTS_NOT_FOUND`); `python -m ...hf_weights <repo>` |
| `crop_select.py` | cleanest mask among candidate views | `is_fragmented(mask) -> bool`, `select_crop_mask(masks, inventory_area_px=None) -> int\|None` |
| `fuse.py` | MoGe-frame candidates + FAH [[Cao, 2026](../../../../docs/REFERENCES.md#cao2026fah)] hand estimate | `hand_depth_scale(depth_small, wrist_px, wrist_z_m, image_wh) -> float`; `select_held_object(cand_sim, extent_m, pinch_mid_sim, usable, table_top_z_m, max_dist_m=0.12, lift_m=0.03, max_extent_m=0.4) -> HeldObject\|None` |
| `grasp_proxy.py` | grasp span from landmarks + proxy box | `pinch_aperture(lm_sim)`, `detect_grasp_proxy(lm_sim, timestamps_s, table_top_z_m=0.8, max_height_m=0.1, min_hold_s=0.3) -> GraspProxy\|None` |
| `mesh_asset.py` | generated unit-scale mesh -> table-resting sim asset | `rest_transform(vertices, faces, max_extent_m, pinch_axis_xy=None, min_height_m=0.015) -> (4,4)`; `prepare_mesh(...)`; `build_mesh_asset(vertices, faces, max_extent_m, pinch_axis_xy, crop_rgba) -> MeshAsset`; `prism_hull`, `slab_prism`, `footprint_chord_width`, `antipodal_axis_xy(vertices, axis_xy, max_turn_deg=45, ...)` |
| `mesh_texture.py` | colour a shape-only mesh from its RGBA crop | `fit_view(points, rgba) -> (R, iou)`, `bake_texture(vertices, faces, rgba, size=512) -> MeshTexture` |
| `mug_mesh.py` | procedural mug (handle on +x) | `procedural_mug(radius_m=0.037, height_m=0.09, wall_m=0.004)` -> `(V,3)`, `(F,3)` |
| `physics_params.py` | mass/friction priors (SimFoundry [[NVIDIA GEAR, 2026](../../../../docs/REFERENCES.md#simfoundry2026)] stage 11) | `vlm_prompt(phrase, bbox_cm, volume_cm3)`, `parse_physics_json(text)`, `heuristic_physics(phrase, volume_m3) -> dict` |
| `place_target.py` | receptacle and release/drop point | `match_object_name(names, phrase)`, `resolve_from_scene(scene_objects, phrase, held_half_z_m) -> PlaceTarget\|None`, `resolve_from_detections(...)` (no-reconstruction fallback), `place_target_for`, `container_profile`, `drop_height`, `tub_profile`, `container_pieces`, `rebase(target, centre_xy_m, table_top_z_m)` |
| `twin_schema.py` | `asset.yaml` pydantic contract | `AssetSpec`, `RestPose`, `PhysicsPrior`, `Hollow`; `load_asset_spec(path)`, `sha256_file`, `validate_asset_dir(root) -> list[str]` |
| `twin_import.py` | authored mesh -> asset package | `load_mesh`, `normalise_mesh(mesh, size_m=None)`, `mesh_report`, `collision_pieces(mesh, threshold=0.05, max_pieces=16)`, `import_twin(src, out_dir, *, asset_id, category, size_m, colour_hsv, physics, licence, author, ...)` |
| `twin_asset.py` | package -> simulator object | `asset_mass_kg(spec, mesh)`, `load_twin_asset(asset_dir, name, pos_m, *, friction=None, mass_kg=None, static=False, robot_collision=True) -> BoxObject` |

Dataclasses: `HeldObject(start_xy_m, half_size_m, n_matches, observed_lift_m, first_idx, last_idx,
pinch_z_pre_m, views, size_m)`; `GraspProxy(grasp_idx, release_idx, center_m, half_size_m)`;
`MeshAsset(vertices, faces, uv, texture, texture_iou)`; `PlaceTarget(name, drop_point_m, half_size_m, obj)`.

## Inputs / outputs

- Candidates are in the sim frame (metres, z up); `cand_sim`, `extent_m`, `pinch_mid_sim` come from
  `scripts/retarget_estimated.py` after `align.frames` anchoring.
- Twin package under `data/assets/twin/<id>/`: `asset.yaml` (`version: twin_asset/1`), `visual.glb`
  (Z-up, metres, origin at bbox bottom centre), `collision/piece_NN.obj`, `preview.png`; the importer also
  writes `import_report.json`.
- Mesh generation outputs themselves come from `scripts/generate_object_mesh.py`.

## Callers

`scripts/crop_object_sam3.py`, `perceive_objects_sam3.py` (`hf_weights`, `prompt`, `crop_select`);
`perceive_objects_owl.py`, `perceive_scene_objects.py`, `make_clean_plate.py`, `twin_appearance_rsi.py`
(`prompt`); `retarget_estimated.py` (`fuse`, `grasp_proxy`, `mesh_asset`, `place_target`, `prompt`);
`preview_object_mesh.py`, `generate_object_mesh.py`, `export_twin_scene.py` (`mesh_asset`/`mesh_texture`);
`scripts/twin_import.py` (`twin_import`); `scripts/gen_schema.py` (`twin_schema`);
`stages/retargeting/held_mesh.py`, `stages/locomotion/clip_task.py` (`mug_mesh`, `physics_params`).

## Constants and env vars

`MIN_HEIGHT_M=0.015` (mesh_asset), `MAX_FRAME_FRACTION=0.1`, `MIN_COMPONENT_FRACTION=0.05` (crop_select),
`MIN_MASS_KG=0.03` (physics_params), `TWIN_MIN_MASS_KG=0.005`, `MAX_COLLISION_PIECES=32`,
`TWIN_ASSET_VERSION="twin_asset/1"`, `ID_PATTERN=^[a-z][a-z0-9_]*$`, `MIN_WALL_M=0.0005`,
place_target `CLEARANCE_M=0.01`, `TUB_FLOOR_FRAC=0.6`, `TUB_RIM_FRAC=0.82`. Env: `HF_HUB_CACHE`, `HF_HOME`.

## Gotchas

- `.glb/.gltf` are treated as Y-up on import; `.obj/.stl/.ply` keep the author's axes (assumed Z-up).
- `load_twin_asset` `pos_m` is the *foot* (bbox bottom centre), not the centre.
- Watertight meshes get CoACD [[Wei et al., 2022](../../../../docs/REFERENCES.md#wei2022coacd)] pieces, others a convex hull (`mesh_report`).
- `RelayPending`-style VLM answers are not used here; `physics_params.vlm_prompt` only builds the prompt.
- Depth scales of hand (FAH) and MoGe [[Wang et al., 2025b](../../../../docs/REFERENCES.md#wang2025moge2)] differ; `hand_depth_scale` reconciles them.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../../docs/REFERENCES.md).

- <span id="cite-cao2026fah"></span>[Cao, 2026](../../../../docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](../../../../docs/REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](../../../../docs/REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](../../../../docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-wei2022coacd"></span>[Wei et al., 2022](../../../../docs/REFERENCES.md#wei2022coacd). *Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search (CoACD).*
