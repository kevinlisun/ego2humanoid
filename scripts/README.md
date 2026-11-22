# `scripts/` — command-line entry points

Command-line entry points of the real2sim pipeline (62 files). `<clip>` is a clip directory such as
`data/derived/egodex/basic_pick_place_0/`. Stage names (`export`, `intrinsics`, `gravity`, `hands`, `tactile`,
`objects`, `scene`, `scene_mesh`, `held_mesh`, `retarget`) are those of `scripts/run_pipeline.sh`; the reference is
`run_pipeline.md` and the design is `docs/TECH_DESIGN.md`. Each Python script's `--help` or docstring is authoritative.

Most perception scripts run in a model venv (`PERC_PY`, `FAH_PY`, `WILOR_PY`, `EGOTAC_PY`) or, with `DOCKER=1`, in a
node image through `nodes/bin/<node>-python`. Only the RGB video is used; EgoDex [[Hoque et al., 2025](../docs/REFERENCES.md#hoque2025egodex)] ground truth is used only by the
`eval_*` scripts.

## 1. Pipeline drivers

| Script | Purpose / key args and env | Reads → writes |
|---|---|---|
| `run_pipeline.sh` | One-command chain `<clip_id>...`. Env: `HAND`, `RENDER`, `JOBS`, `STAGES`, `FORCE`, `DRY_RUN`, `DOCKER`, `TASK`, `RAW_DIR`, `DERIVED_DIR`, `LOG_DIR`, `PERC_PY`, `FAH_PY`, `WILOR_PY`, `EGOTAC_PY`, `OBJ_DETECTOR`, `HF_HOME`. Skips finished stages. | raw EgoDex hdf5+mp4 → all per-clip outputs, `$LOG_DIR/batch_results.md` (all stages) |
| `pipeline_progress.sh` | Progress bar for a running batch: `<clip>...`; env `LOG_DIR`, `HAND`. Use with `watch`. | `$LOG_DIR` logs → terminal |
| `setup_usecase0.sh` | Idempotent one-shot setup for the clip-0 tutorial (submodules, clips, robot models, weights, node images); `--no-build` skips Docker. Needs the SAM 3 [[Carion et al., 2026](../docs/REFERENCES.md#carion2026sam3)] licence, `hf auth login` and `MANO_RIGHT.pkl`. | network → `third_party/`, `data/`, images |
| `build_node.sh` | Build a node image from the repo root: `<node> [tag]`. | `nodes/<node>/Dockerfile` → image `ego2humanoid/<node>` |
| `node_python.sh` | Runs `python <args>` inside a node image; invoked via the `nodes/bin/<node>-python` symlinks. Env: `IMAGE_TAG`, `HF_HOME`, `FAH_SRC`, `WILOR_SRC`, `EGOTAC_SRC`, `MANO_SRC`, `FAH_WEIGHTS_SRC`, `DOCKER_ARGS`. | host paths mounted as-is |
| `smoke_node.sh` | Smoke test of a built image: `<preflight\|replay> [tag]`. | image → pass/fail |
| `trim_clip.py` | Trim `video.mp4` of your own long clip to a window: `clip_dir start_s end_s`. Keeps the original in `orig/video_full.mp4`, refreshes the sha256 in `episode.json`; delete perception outputs afterwards. | `<clip>/video.mp4` → same + `orig/` |
| `rebuild_hot_disk.sh` | Recreate the ephemeral NVMe hot disk and rebuild the SimFoundry [[NVIDIA GEAR, 2026](../docs/REFERENCES.md#simfoundry2026)] conda envs on it (default `simfoundry hunyuan any6d da3`). | host → `/mnt/nvme_hot` |

## 2. Perception stage scripts (RGB only)

| Script | Stage | Purpose / args | Output |
|---|---|---|---|
| `estimate_intrinsics_moge.py` | `intrinsics` | Median MoGe-2 [[Wang et al., 2025b](../docs/REFERENCES.md#wang2025moge2)] focal over sampled frames; `<clip>` | `intrinsics_est.json` |
| `estimate_gravity_moge.py` | `gravity` | Gravity-up (OpenCV camera frame) from MoGe-2 normals, RANSAC-style vote on the lower image half; `<clip>` | `gravity_est.json` |
| `estimate_hands_fah.py` | `hands` | FAH [[Cao, 2026](../docs/REFERENCES.md#cao2026fah)] video-mode hands; `clip_dir --model --focal --max-frames`, `--use-gt-k` is an ablation. Env `FAH_HOME` (FAH venv) | `hands_fah.npz` (`joints3d (N,2,21,3)`, `valid`) |
| `estimate_hands_wilor.py` | `hands` | WiLoR [[Potamias et al., 2025](../docs/REFERENCES.md#potamias2025wilor)] fallback when FAH is unavailable; `clip_dir --focal --repo --max-frames`. Env `WILOR_REPO` | `hands_fah_f<focal>.npz` (FAH-compatible keys) |
| `estimate_tactile_egotac.py` | `tactile` | EgoTac [[Zhang et al., 2026](../docs/REFERENCES.md#zhang2026egotac)] per-vertex force and contact (EgoTac venv); `clip_dir --checkpoint --max-frames --stride --device` | `tactile_egotac.npz` (`force`, `contact_prob (N,2,778)`) |
| `perceive_objects_sam3.py` | `objects` | SAM 3 text-prompted masks + MoGe-2 point map → object 3D track; `<clip> "<prompt>" [stride] [threshold]` | `objects_sam3.npz` |
| `perceive_objects_owl.py` | `objects` | Public-weights fallback (OWLv2 [[Minderer et al., 2023](../docs/REFERENCES.md#minderer2023owlv2)] boxes → SAM masks + MoGe-2); same args | `objects_sam3.npz` (same arrays + `detector`) |
| `perceive_scene_objects.py` | `scene` | Inventory of every tabletop object from early keyframes; `<clip> [n_keyframes=4] [span_s=1.0]` | `scene/inventory.json`, `points.npz`, `crop_*.png` |
| `generate_object_mesh.py` | `scene_mesh`, `held_mesh` | Hunyuan3D-2mini [[Hunyuan3D Team, 2025](../docs/REFERENCES.md#hunyuan2025hunyuan3d2)] image → mesh (best seed by silhouette IoU); `<clip> [--scene]`. Weights `data/weights/hunyuan3d/Hunyuan3D-2mini` | `object_mesh_raw.glb`, or `scene/mesh_<id>.glb` with `--scene` |
| `crop_object_sam3.py` | `held_mesh` | Background-free RGBA crop of the held object; `<clip> "<prompt>"`; reads `perceived_object.views` of `retarget_rgb*.json` | `object_crop.png` |
| `estimate_depth_moge.py` | (walking clips) | Per-frame metric MoGe-2 depth at the clip focal; `<clip>` | `depth_moge.npz` |
| `estimate_camera_poses.py` | (walking clips) | Lightweight visual odometry with hands masked; needs `depth_moge.npz`, `intrinsics_est.json`, `hands_fah_f645.npz` | `camera_poses.npz`, plot in `review/` |
| `preview_object_mesh.py` | check | Crop next to four orbit views of the mesh; `<clip>`, `MUJOCO_GL=egl` | `object_preview.png` |

## 3. Retarget and simulation

| Script | Purpose / args and env | Output |
|---|---|---|
| `retarget_estimated.py` | The core stage `retarget`: `<clip> [--tactile=none\|egotac]`. Hands + gravity → placement, IK, pinch, grip timing, table fit, closed-loop pinch correction, MuJoCo [[Todorov et al., 2012](../docs/REFERENCES.md#todorov2012mujoco)] PD rollout, physics verdict, optional review video. Env: `HAND` (`dex3` default here, `sharpa`), `RENDER`, `RENDER_FPS`, `E2H_OUT_TAG`, `E2H_RETARGET_CACHE`, `E2H_SIM_PARAMS`, `E2H_EGO_POSES`, `E2H_SCENE_OBJECTS`, `SCENE_DIR`, `BPY_PYTHON`; `E2H_PLACE_GOAL` / `E2H_PLACE_RETRY` are internal (self re-exec). | `retarget_rgb{,_sharpa}.{json,npz}`, `scene_sim.pkl`, `review/*` |
| `review_render.py` | Module used by the retarget script when `RENDER=1` (sim, scene, ego, trajectory panels, grid). Env `RENDER_BACKEND` (`blender` default), `SCENE_DIR`. Not a CLI. | `review/*` |
| `record_sim_state.py` | Replay the cached retarget through MuJoCo without perception or render (~1 min): `<clip> [tag]`. | `review/sim_state_sharpa<tag>.npz`, `review/sim_model/` |
| `loco_manip_episode.py` | Locomanipulation episode: `--list`, `--clip N`, `--case NAME`, `--jobs`, `--no-video`. | `loco_manip_result.json`, `review/loco_manip.mp4`, or `loco_manip_<case>/` |
| `real2sim_eval.py` | Agentic real2sim CLI (Mode A): `init \| schema \| eval \| history \| show \| robust \| accept`; options `--hand --backend --params --set --freeze-plan --offsets --render --parent --rationale --iter`. | iteration bundles and `feedback.md` |
| `real2sim_loop.py` | Mode B: evaluate, ask the Copilot relay for a parameter patch, repeat; `clip_dir --max-iters --patience --relay-dir --hand --backend`. Exit 0 = done, 75 = relay answer pending. | iteration bundles, relay outbox |
| `sim_params.py` | `SimParams` CLI: `schema \| resolve \| diff \| dump-defaults`; `--params`, `--set`, `--write`. | stdout or file |
| `solve_sharpa_pinch.py` | Solve the Sharpa pinch joint sets; `--width` (m, default 0.03). | `configs/grasp/sharpa_pinch_v1.json` |
| `profile_dwbc.py` | Measure the Decoupled WBC [[NVIDIA et al., 2025](../docs/REFERENCES.md#nvidia2025gr00tn1)] capability envelope on the G1 asset. | `configs/dwbc_profile.json` |
| `gen_schema.py` | Regenerate the JSON Schemas from the Pydantic models (run with `PYTHONPATH=src`). | `contracts/*.schema.json` |

## 4. Evaluation and batch

| Script | Purpose / args | Output |
|---|---|---|
| `summarize_batch.py` | Metric table over all clips: `[root] [out.md]`; reads `retarget_rgb.json` and `retarget_rgb_sharpa.json`. | markdown table (`$LOG_DIR/batch_results.md`) |
| `eval_hands_vs_egodex.py` | Hand estimates vs EgoDex ARKit GT; `<clip>`; evaluation only. | `hand_eval.json` |
| `eval_tactile_vs_hold.py` | EgoTac contact vs the perceived hold window: `<derived_dir> [out.json]`. | JSON report |
| `run_hand_eval_batch.sh` | Pure-RGB hand evaluation over fetched clips (export → focal → FAH → metrics); env `FAH_PY`, `FAH_HOME`, `FAH_SAM3D`, `MANO_DIR`. | per-clip `hand_eval*.json` |
| `run_object_eval_batch.sh` | Per clip: SAM 3 + MoGe objects → perceived-object retarget; `<clip_dir>...`; needs `HF_TOKEN`, `HF_HOME`. | `objects_sam3.npz`, retarget outputs |
| `run_object_mesh_batch.sh` | Per clip: crop → Hunyuan3D mesh → mesh-object retarget and render; needs outputs of the previous script. | `object_crop.png`, `object_mesh_raw.glb`, retarget outputs |

## 5. Asset fetch

| Script | Purpose | Output |
|---|---|---|
| `fetch_egodex.py` | Remote-zip fetch of EgoDex test clips: `task -n N --out`; keeps clips accepted by the selector (CC-BY-NC-ND, never redistribute). | `data/raw/egodex/test/<task>/<id>.{hdf5,mp4}` |
| `fetch_g1_assets.sh` | Pinned Unitree G1 [[Unitree Robotics](../docs/REFERENCES.md#unitree-g1)] + Dex3 meshes (unitree_ros, BSD-3-Clause). | `assets/robots/g1_dex3/meshes/` |
| `fetch_sharpa_assets.sh` | Pinned Sharpa Wave [[Sharpa](../docs/REFERENCES.md#sharpa-wave)] MJCF and meshes (MuJoCo Menagerie [[Zakka et al., 2022](../docs/REFERENCES.md#zakka2022menagerie)], Apache-2.0). | `assets/robots/sharpa_wave/` |
| `twin_import.py` | Import an authored mesh as a twin asset: `src --id --category --size --hsv --density --licence --author ...`. | `data/assets/twin/<id>/` |

## 6. Render and scene twin (Track BT) and SimFoundry

| Script | Purpose / args | Output |
|---|---|---|
| `render_side_by_side.py` | Library of helpers (no CLI entry point): source RGB with hand estimates and tactile next to the MuJoCo replay. | `review/side_by_side<suffix>.mp4` |
| `make_review_grid.py` | Rebuild `result_grid<suffix>.mp4` from cached panels, redoing only the ego panel: `<clip> [suffix]`. | `review/result_grid*.mp4` |
| `export_twin_scene.py` | Export the simulated scene for Blender [[Blender, 2024](../docs/REFERENCES.md#blender)]: `<clip> [tag]` (default `_bt`). | `review/twin_scene/` |
| `build_twin_scene_cam.py` | Metric twin scene in the video camera frame: `<clip>`. | `review/twin_scene_cam/scene_cam.json`, `hands.npz` |
| `render_twin_blender.py` | bpy renderer of `twin_scene`: `scene --view ego\|overview --out --stride --fps --samples --width`. | mp4 |
| `render_twin_cam.py` | bpy render of the camera-frame twin: `clip --video-frames --stride --samples --width --out`. | PNGs or mp4 |
| `render_twin_sim.py` | bpy replay of a MuJoCo rollout from the head camera: `clip --tag --frames --fps --samples --width --device --out --view --appearance`. | `review/twin_sim<tag>.mp4` or PNGs |
| `twin_render.py` | All Blender views in one bpy session: `clip --tag --state-tag --views ego third human --fps --samples --width --device --frame --appearance`; resumable via cached frames. | `review/<view>_blender<tag>.mp4` |
| `twin_appearance_blender.py` | Blender-side `TwinAppearance` used by `render_twin_sim.py`. Module, not a CLI. | - |
| `twin_appearance_rsi.py` | LLM+RSI loop tuning the twin's look; one resumable round per call: `clip --rounds --frame --target --width --samples --device`. Exit 75 = relay answer pending. | `review/twin_appearance/` |
| `render_blender_clips.sh` | Batch Blender render: `<clip_id>...`; env `TAG`, `DERIVED_DIR`, `TASK`, `FPS`, `BPY_PY`, `WAIT_S`. | `review/twin_sim$TAG.mp4` |
| `render_scene_preview.py` | Orbit video of a locomanipulation scene: `<clip>`. | mp4 |
| `build_bg_splat.py` | Background Gaussian splat [[Kerbl et al., 2023](../docs/REFERENCES.md#kerbl20233dgs)] of a reconstructed scene: `<scene_dir> <clip_dir> [out_npz]`. | `bg_splat.npz` |
| `render_bg_splat.py` | gsplat raster from the head camera (CUDA, `simfoundry` conda env): `<ego_npz> <splat_npz> <out_npz>`. | `bg_rgbd*.npz` |
| `run_simfoundry_clip.py` | Resumable SimFoundry pass: `<scene_name> <video.mp4> [--stages 5,6] [--no-stream]`; exit 75 = relay answer pending. | `data/scenes/<scene>/` |
| `make_clean_plate.py` | Remove people and objects from one frame with FLUX [[Black Forest Labs, 2025](../docs/REFERENCES.md#batifol2025flux)].1-Kontext-dev: `src dst [seed]`. | clean plate PNG |
| `viz_stage5_removal.py`, `viz_stage6_upsample.py`, `viz_stage7_meshes.py`, `viz_stage12_sim.py` | Review videos for SimFoundry stages 5, 6, 7 and 12; `<scene_dir> [out.mp4]` (stage 7 and 12 default to `data/scenes/egodex_104`). | mp4 |

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../docs/REFERENCES.md).

- <span id="cite-batifol2025flux"></span>[Black Forest Labs, 2025](../docs/REFERENCES.md#batifol2025flux). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing.*
- <span id="cite-blender"></span>[Blender, 2024](../docs/REFERENCES.md#blender). *Blender — a 3D modelling and rendering package.*
- <span id="cite-cao2026fah"></span>[Cao, 2026](../docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](../docs/REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-hoque2025egodex"></span>[Hoque et al., 2025](../docs/REFERENCES.md#hoque2025egodex). *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.*
- <span id="cite-hunyuan2025hunyuan3d2"></span>[Hunyuan3D Team, 2025](../docs/REFERENCES.md#hunyuan2025hunyuan3d2). *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation.*
- <span id="cite-kerbl20233dgs"></span>[Kerbl et al., 2023](../docs/REFERENCES.md#kerbl20233dgs). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
- <span id="cite-minderer2023owlv2"></span>[Minderer et al., 2023](../docs/REFERENCES.md#minderer2023owlv2). *Scaling Open-Vocabulary Object Detection (OWLv2).*
- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](../docs/REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
- <span id="cite-nvidia2025gr00tn1"></span>[NVIDIA et al., 2025](../docs/REFERENCES.md#nvidia2025gr00tn1). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*
- <span id="cite-potamias2025wilor"></span>[Potamias et al., 2025](../docs/REFERENCES.md#potamias2025wilor). *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.*
- <span id="cite-sharpa-wave"></span>[Sharpa](../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-unitree-g1"></span>[Unitree Robotics](../docs/REFERENCES.md#unitree-g1). *G1 humanoid robot and Dex3-1 dexterous hand.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](../docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-zakka2022menagerie"></span>[Zakka et al., 2022](../docs/REFERENCES.md#zakka2022menagerie). *MuJoCo Menagerie.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
