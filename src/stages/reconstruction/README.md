# `src/stages/reconstruction/` — hands, objects, scene and camera alignment

Egocentric video -> hands, objects, scene and camera alignment. This directory holds no logic of its own:
`__init__.py` is a one-line docstring and all code lives in four sub-packages. Design context:
[docs/TECH_DESIGN.md §3 Perception](../../../docs/TECH_DESIGN.md), §4 (alignment, object asset, table),
§4.10 (tabletop scene objects).

## Sub-packages

| Package | Responsibility | README |
|---|---|---|
| `align/` | camera -> simulation frame, head-camera tilt, synthetic table fit, monocular VO | [align](align/README.md) |
| `hands/` | hands backend config, landmark cleanup, EgoTac [[Zhang et al., 2026](../../../docs/REFERENCES.md#zhang2026egotac)] tactile regions | [hands](hands/README.md) |
| `objects/` | held-object selection, mesh asset, grasp proxy, place target, twin asset packages | [objects](objects/README.md) |
| `scene/` | tabletop inventory, background splat, roles, MJCF export, SimFoundry [[NVIDIA GEAR, 2026](../../../docs/REFERENCES.md#simfoundry2026)] runner | [scene](scene/README.md) |

## Data flow (as wired by the scripts)

1. `scripts/estimate_hands_fah.py` -> `hands_fah_f<focal>.npz` (read through `hands.backend.artefact_names`).
2. `scripts/estimate_depth_moge.py` -> `depth_moge.npz`; `scripts/estimate_camera_poses.py` runs
   `align.ego_vo.estimate_poses` -> `camera_poses.npz`.
3. `scripts/estimate_tactile_egotac.py` -> `tactile_egotac.npz` (raw per-vertex), reduced by `hands.tactile`.
4. `scripts/crop_object_sam3.py` / `perceive_objects_sam3.py` / `perceive_scene_objects.py` use
   `objects.prompt` and `objects.hf_weights`; scene inventory builds `objects.json` (`scene.inventory`).
5. `scripts/retarget_estimated.py` is the main consumer: `align.*`, `hands.smooth`, `hands.backend`,
   `objects.fuse`, `grasp_proxy`, `mesh_asset`, `place_target`.

## Used outside this package

- `stages/locomotion/base_placement.py`, `stance_adjust.py`: `align.camera_fit` and `align.frames`.
- `stages/retargeting/clip.py`: `align.frames.gravity_aligned_rotation`, `align.table_fit`.
- `skills/body_state.py`, `stages/manipulation/sim_tactile.py`, `stages/qa/hand_overlay.py`:
  `hands.tactile.REGION_NAMES` / `N_REGIONS`.
- `stages/retargeting/held_mesh.py`, `stages/locomotion/clip_task.py`: `objects.mug_mesh.procedural_mug`,
  `objects.physics_params.heuristic_physics`.

## Environment variables

| Variable | Where | Meaning |
|---|---|---|
| `E2H_TACTILE_BACKEND` | `hands/backend.py` | `none` or `egotac`; overrides `configs/hands.yaml` |
| `HF_HUB_CACHE`, `HF_HOME` | `objects/hf_weights.py` | where cached Hugging Face weights are looked up |
| `E2H_HELD_MATCH_M` | `scene/inventory.py` | overrides held-object match / registration radius (m) |
| `E2H_RELAY_DIR` | `scene/relay_hook/sitecustomize.py` | directory of the file-based VLM relay |

## Conventions and gotchas

- Frames: perception arrays are in the OpenCV camera frame (x right, y down, z forward); the simulation
  frame is z-up with the table top at `table_top_z_m` and the pelvis near the origin (`align/frames.py`).
- Monocular metric depth is biased, so positions are anchored to the table/workspace, not trusted absolutely.
- Heavy dependencies (mujoco, trimesh, cv2, torch) are imported lazily or per module; `hands/backend.py` is
  deliberately pure Python.
- Some modules need different Python environments (e.g. gsplat needs CUDA, see `scene/neural_ego.py`).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](../../../docs/REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../../../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
