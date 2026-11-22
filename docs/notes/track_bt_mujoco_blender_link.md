# Track BT — MuJoCo ↔ Blender link (technical design and feasibility)

**Status:** design + feasibility analysis (owner request 2026-11-19). Extends
`docs/notes/track_bt_blender_twin.md` (§6 decision: Blender twin replaces SimFoundry). Tasks: `TASKS.md`
BT.15–BT.31 (§7 renderer rule and §8 appearance loop implemented in BT.23–BT.31).

Paths in this note are relative to `src/` unless prefixed `scripts/`, `configs/` or `docs/`. Module layout after the 2026-11 refactor: `stages/{reconstruction,retargeting,locomotion,manipulation,qa}`, `skills/`, `rsi/`.

## 0. Production pipeline (baseline `0025c3c`, the one currently run)

This is the pipeline that produces the deliverable videos. §1–§6 are the design background, §7 the renderer
rule, §8 an **optional** appearance loop that is *not* part of the default run, §9 the known failures.

**Scheme:** Hunyuan3D reconstructs every object as a mesh; MuJoCo simulates headless (physics only); Blender
renders all views. No primitive-shape appearance override is applied: the Blender scene shows the
reconstructed meshes with their own textures (a stale `review/twin_appearance.json` must not be present,
otherwise §8 primitives replace the meshes).

```
EgoDex clip
  export → intrinsics → gravity → hands (FAH/WiLoR) → tactile (optional, EgoTac)
  → objects (SAM 3 / OWLv2 held-object crop)
  → scene (inventory.json: labels, roles, metric footprints; MoGe point map, table plane)
  → scene_mesh (Hunyuan3D per scene object → scene/mesh_*.glb)
  → held_mesh (Hunyuan3D for the held object)
  → retarget (scripts/retarget_estimated.py, stages/retargeting + stages/manipulation)
        ├─ hand/arm retargeting to Sharpa, headless MuJoCo rollout @50 Hz
        ├─ writes trajectory_*.npz, sim_state_*.npz (v2: poses, camera, pad force), retarget report JSON
        ├─ export_twin_scene.py → review/twin_scene/ (neutral npz/json)
        ├─ twin_render.py (bpy, GPU Cycles, one session): ego (D435) + third-person views
        └─ stages/qa/grid_video.compose_grid → review/result_grid_sharpa<tag>.mp4
```

| Item | Value |
|---|---|
| Entry point | `scripts/run_pipeline.sh <clip>...` (`STAGES`, `FORCE`, `JOBS`, `HAND=sharpa`, `RENDER=1`, `DOCKER=1` for per-stage node images) |
| Experiment tag | `E2H_OUT_TAG` (e.g. `_bt_20261120_2242`, date + minute); file suffix `_sharpa$E2H_OUT_TAG` |
| Videos per clip | `review/result_grid_sharpa<tag>.mp4`, `review/third_blender<tag>.mp4`, `review/ego_blender<tag>.mp4` |
| Render rate | `RENDER_FPS` (default 10); PNG cache makes the render resumable |
| Environments | host: main uv venv (physics), `/tmp/bpy_venv` (Blender 5, Python 3.11; `BPY_PYTHON`), perception venvs (`PERC_PY`, `FAH_PY`). `DOCKER=1`: node images `nodes/<node>` via `scripts/node_python.sh` (the retarget image carries bpy at `/opt/bpy`) |
| QA | `python -m skills.physics_fidelity <clip_dir> --tag <tag>` (sink, drift, timing, pad range) |
| Run results | honest per-clip table in `docs/notes/batch_results.md` |

Typical cost per clip when 3 clips run in parallel on one T4: ~20–30 min for `retarget` + Blender render
(mesh stages are cached in `scene/`).

**Known limitations of this baseline** (diagnosis in §9; fixes are BT.32–34, not implemented here):
duplicate detections interpenetrate at spawn (114), source/receptacle roles can swap when the hand-eye offset
is large (108), and single-view Hunyuan meshes of occluded objects can be flat or fragmented (100, 114).
Video look is whatever the meshes give; look tuning (§8) is opt-in per clip.

**Relation to other branches:** `E2H_SCHEME` is not in this tree. `feature/blender` carries later work (`E2H_SCHEME` blender|hunyuan switch,
multi-view mesh validation and agent RSI of meshes, BT.35–42). Its recent commits regressed, so this
document describes the `0025c3c` baseline; port those features back one at a time with a rerun of all
ten usecases as the gate.

## 1. Goal and principle

Blender reconstructs and renders; MuJoCo is the **only physics engine**.

```
video ─► perception ─► TwinSpec (one scene graph, metric, video-camera frame)
                          │                         │
                 compile_mjcf()               compile_blender()
                          ▼                         ▼
                 MuJoCo scene  ──rollout──►  sim_state.npz  ──replay──►  Blender render (Cycles/EEVEE)
                     ▲                                                        │
                     └───────────── RSI: tune SimParams ◄── compare(video, render, state)
```

1. **One source of truth** (`TwinSpec`): both MJCF and the `.blend` scene are *compiled* from it, never
   hand-edited, so shape, scale, origin and name of every object are identical on both sides.
2. **State, not geometry, crosses the link**: MuJoCo writes per-frame body poses (`sim_state.npz`);
   Blender only sets transforms and renders. Blender never simulates and never feeds back physics.
3. **Calibration happens in MuJoCo** (mass, friction, solref/solimp, pose offsets, release timing). A
   render is an *observation* of the result, used to judge it against the video.
4. Same constraints as Track BT: Sharpa + MuJoCo only; no GT in the pipeline (GT poses only in labelled
   evaluation); no large binaries in git.

## 2. Components

### 2.1 `TwinSpec` (`contracts`, new, versioned `twin_spec/1`)

Per clip `review/twin_spec.json`:

| Field | Content |
|---|---|
| `frame` | `world = video camera frame at the keyframe` (metric, MoGe × `depth_scale`), `up` from the table normal |
| `T_world_sim` | **one rigid transform** (robot placement + table fit). The only link between the twin world and the robot-centric sim frame. No per-object non-rigid fix |
| `objects[]` | `name`, `role` (`static`/`dynamic`/`held`/`surface`), `asset` (id, version, sha256 or `procedural`), `pose0` (world), `support` (name, top_m), `physics` (prior, from `asset.yaml`), `body_name` (shared MJCF/Blender name) |
| `room` | table plane, wall planes, colours (render only; walls are MuJoCo static planes only if reachable) |
| `camera` | intrinsics, per-frame `T_world_cam` (source recorded: `estimated` or `gt_eval_only`) |
| `render` | lights, exposure, samples (render only) |

At the time of writing these live in `scene_cam.json` (render) and `scene_sim.pkl` (sim, written by `scripts/retarget_estimated.py`) separately, which is the gap this
design closes. `build_twin_scene_cam.py` becomes the `TwinSpec` writer.

### 2.2 `compile_mjcf` (extends `twin_asset.py`, BT.8)

* `surface`/`static`: box or mesh geoms, `contype=1`, no mass.
* `dynamic`/`held`: free joint, visual mesh `contype=0` + CoACD pieces (≤ 32), mass from `density×volume`,
  inertia from the pieces, `SimParams.objects.<name>` overrides.
* Body names equal `TwinSpec.body_name`. Geometry centre = Blender object origin (the asset origin is the
  bbox bottom-centre; the compiler writes the offset once, in one place).

### 2.3 `sim_state.npz` (the link format)

Recorder generalises `stages/qa/traj_panel.py::TrajectoryRecorder` (implemented as `skills/body_state.BodyStateRecorder`, §7.6), which stores only `wrist_m`,
`fingertip_m` and the **held object** pose. New arrays, per physics step subsampled to the render rate:

| Key | Shape | Content |
|---|---|---|
| `t` | (N,) | sim time |
| `body_names` | (B,) | scene bodies (objects + robot links that have visual meshes) |
| `xpos`, `xquat` | (N,B,3), (N,B,4) | world pose of each body in the **sim frame** (`mujoco` `data.xpos/xquat`) |
| `contact` | (N,B) | any contact with the hand (for overlays and QA) |
| `meta` | json | model hash, `SimParams` hash, `TwinSpec` hash, `T_world_sim` |

Replay needs only these arrays and the asset files; no MuJoCo import in Blender. Size: 593 frames × ~80
bodies × 7 floats ≈ 1.3 MB in float32.

### 2.4 Blender replay (`scripts/render_twin_cam.py` extension)

1. Load `TwinSpec`, build objects once (stands, pen, textured meshes); robot links are imported once from
   the existing MuJoCo visual meshes (`assets/robots/...`, 15 MB Sharpa, 111 MB G1) as Blender objects
   named after bodies (**rigid-body replay, no armature**: a MuJoCo kinematic chain is already a set of
   rigid link poses).
2. Per frame: for each body, `matrix_world = T_world_sim⁻¹-mapped(xpos, xquat)`, then the OpenCV→Blender
   conversion `D` (already implemented); camera from `T_world_cam[i]` (video frame) or the robot head
   camera (sim) per `--view`.
3. Render with Cycles (final) or EEVEE (preview), write PNG/video. Resample sim time → video time by
   interpolation (sim 50 Hz, render 10 fps).

Views: `video-view` (twin world, video camera: compare with the human video) and `robot-view` (D435 head
camera: the deliverable).

### 2.5 RSI loop (BT.12/13 on top of this link)

```
SimParams ─► MuJoCo rollout ─► sim_state.npz ─┬─► QA: timing, lift, success (cheap, every trial)
                                              └─► Blender keyframe render ─► compare vs video (BT.11)
```

* Inner loop is MuJoCo only (seconds/trial). Blender is invoked **only for accepted candidates and
  keyframes** (3–6 frames, ~30 s/frame on 8 CPU cores) — never per trial.
* Tuned: per-object `mass_kg`, `friction`, `solref/solimp`, `pose_offset_m`, `yaw_rad`; release/hold
  timing. Measured against `tactile_timing`, `hold_s`, observed lift and carry (monocular evidence, so
  reported as a plausible range, not identification).

## 3. Feasibility assessment

Evidence is from the repo as of `a042012` and the clip 108 render runs.

| # | Question | Finding | Verdict |
|---|---|---|---|
| F1 | Can Blender render a scene from MuJoCo state? | `render_twin_blender.py` already replays `object_pos_m/object_rot` + `wrist_m` per frame from `trajectory_sharpa_bt.npz`; `render_twin_cam.py` renders the camera-frame twin. Pose-setting is trivial in bpy | **Feasible, done for 1 object + hand skeleton** |
| F2 | Do MuJoCo and Blender see identical shapes? | `twin_asset.py` maps `visual.glb` → MuJoCo mesh; UV v-flip is handled and unit-tested (5 tests). Collision = CoACD pieces from the same mesh | **Feasible**; need the shared-origin convention in one compiler function |
| F3 | Is state recorded for all bodies? | **No.** `TrajectoryRecorder` records only the held object, wrists and fingertips; scene objects and robot links are not saved (`stages/qa/traj_panel.py`, `TrajectoryRecorder.arrays`; later closed by `skills/body_state.py`). `run_rollout(on_record=...)` gives `(model, data)` each step, so a body recorder is a ~40-line addition | **Feasible, gap (BT.15)** |
| F4 | Same coordinates on both sides? | **Main risk.** The sim frame is robot-centric with per-object registration offsets and a table fitted to the robot (`table_z=0.78`, §4.6), so it is *not* metrically comparable to the video (hands were ~200 px off in the first sim-frame render). Twin world (camera frame, metric) is correct | **Feasible only with F4-fix below** |
| F5 | Robot meshes in Blender? | MuJoCo visual meshes (STL/OBJ in `assets/robots`) load with `bpy.ops.wm.*_import` or trimesh→bpy; 126 MB total, ~100–150 mesh geoms; per-frame cost is transform updates only | **Feasible**; check mesh count/units (mm vs m) in BT.17 |
| F6 | Render cost? | Clip 108 camera-frame render: 114 frames, 960×540, Cycles CPU, 3 bounces, **≈ 31 s/frame (≈ 60 min / video, 8 cores)**; sim-frame open scene ≈ 7–10 s/frame. A robot with ~120 meshes adds little (BVH build), but the closed room is the cost | Video renders are batch jobs; **keyframes (3–6) for RSI ≈ 2–3 min**. Use EEVEE / fewer samples / denoise for previews. Is GPU available? A100 devbox: Cycles CUDA ≈ 10× faster, not yet tried |
| F7 | Physics fidelity vs. Blender geometry? | Convex decomposition of the visual mesh is an approximation (≤ 32 pieces); thin/hollow objects (pen, cup) need care. Pen as cylinder: exact. Meshes from Blender are *manifold-checked* in `twin_import.py` | **Feasible**; hollow containers need explicit inner-floor collision (asset field `hollow`) |
| F8 | Can MuJoCo tune what we need? | `SimParams.objects.<name>` has `mass_kg`, `friction` and the scene edits (`enabled`, `static`, `robot_collision`, `pos_offset_m`, `yaw_offset_deg`, `size_scale`; `contracts/sim_params.py` `ObjectPhysics`); `solref/solimp/pose_offset/yaw/scale` are BT.12 work. Rollout takes seconds-minutes per trial (one rollout per start offset, 25 offsets today) | **Feasible**; budget ≤ 6 params/object, coordinate descent |
| F9 | Does the loop converge on clip 108? | Unknown. 108 currently fails `NO_LIFT` because sim stands overlap and pen z follows grip height. Both are fixed by compiling from `TwinSpec` (pen on stand top) | **Must be shown first** (milestone M2) |
| F10 | Head/camera motion in render | EgoDex GT poses work (≤ 3 cm static-hand check). A no-GT source needs `estimate_camera_poses.py` (needs `depth_moge.npz`, `hands_fah_f645.npz`, not present for 108) | Feasible for evaluation; no-GT path pending |
| F11 | Determinism / provenance | Hash of `TwinSpec`, `SimParams`, model and asset sha256 in `sim_state.meta`; render reads and re-checks | Feasible; required by HARNESS reproducibility |

### F4-fix: one world frame, one rigid link

1. Define `world` = video camera frame (keyframe), metric.
2. Robot placement already yields `T_sim_camera`. Make the sim scene by **rigidly transforming the twin**
   with `T_sim_world = T_sim_camera⁻¹`-composed placement; no per-object correction.
3. The table fit is a *translation of the whole twin world along `up`* (objects, stands, table
   together), recorded in `T_world_sim`. Relative heights (pen 10 cm above the table, etc.) stay exact.
4. Remaining mismatches (hand-eye error, depth bias) are explicit RSI `pose_offset_m` values, applied in
   MuJoCo and **inverted at render**, so the render shows what the video shows and the offset is reported.

Risk: the G1/Sharpa reach envelope may not allow the real table height. Mitigation as today (fit the
world, not the objects); the robot-view render then shows a lowered/raised room, flagged in `meta`.

## 4. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Frame mismatch (F4) | Render ≠ sim, misleading RSI | Single `T_world_sim`; unit test: body pose round trip `sim→world→sim` < 1e-6; keyframe overlay test against the twin |
| Name/origin drift between MJCF and Blender | Objects offset or rotated | Compiler generates both; test that bbox of the MuJoCo geom equals the Blender bbox |
| Render time dominates RSI | Slow iteration | Blender only on accepted candidates + keyframes; EEVEE preview; GPU Cycles trial |
| Robot mesh import (units, normals, duplicates) | Broken robot in render | Import check script; compare silhouette of Blender robot vs MuJoCo EGL render |
| Convex hull ≠ visual shape | Wrong contact | Report `collision_vs_visual` IoU in `import_report.json`; hollow flag |
| Over-fitting parameters to monocular evidence | False "calibrated" claims | Report ranges, not points; QA with ≥ 3/5 perturbed rollouts (RSI.18) |
| bpy deployment | Needs Python 3.11 venv, 125 GB off-root disk | Keep `/tmp/bpy_venv`; document a `uv` recipe; never import bpy in the main env |

## 5. Plan (tasks added to `TASKS.md`)

| Milestone | Tasks | Exit criterion |
|---|---|---|
| M1 link format | BT.15 `TwinSpec` + `build_twin_scene_cam` writes it; BT.16 `compile_mjcf` + `BodyRecorder` → `sim_state.npz` | Round-trip test; clip 108 scene compiled, rollout produces `sim_state.npz` for all bodies |
| M2 108 physics works | BT.17 Blender replay of `sim_state.npz` incl. robot links; BT.18 `T_world_sim` rigid link + pen on support (BT.4) | 108 passes lift/place; twin render and MuJoCo EGL overlay agree within 1 px/1 cm on keyframes |
| M3 tuning | BT.19 object RSI params (`solref/solimp/pose_offset/yaw`) + body-state QA; BT.20 keyframe comparator in the loop (BT.11) | Timing error < 0.3 s on 108; parameter ranges reported |
| M4 scale-out | BT.21 asset catalogue (authored assets), binder (BT.6/7); BT.22 batch 10/100/101/104/108/114 + docs | Honest table in `docs/notes/batch_results.md` |

## 6. Owner decisions and open points

* **Q22 (decided 2026-11-19): simulation renders use the robot camera frame.** The deliverable is the Blender
  render from the robot head (D435) camera, i.e. `robot-view`. The twin-world `video-view` stays as a
  comparison/RSI view only. `compile_blender` therefore takes the camera from `sim_state.npz`/`T_world_sim`
  (robot head body pose per frame), not from `camera_poses.npz`.
* **Q23 (decided 2026-11-19): simulation renders use GPU acceleration.** Cycles with CUDA/OptiX on the A100
  devbox (`devbox-a100`); CPU stays as the fallback. Needs a bpy GPU smoke test (BT.17): device enumeration,
  frame time versus the 31 s/frame CPU baseline, and the T4 devbox behaviour.
* **Q24 (decided 2026-11-19, owner agreed): table-height policy.** Human table and robot differ: the real table may be
  at a height the humanoid cannot reach well. Today the *table* is moved to the robot (§4.6), which
  detaches it from the objects on it. Decision: instead move the *whole scene* (table, stands, objects)
  up/down together by one offset, keeping relative heights exact. Rendered room height may then differ from
  the video by that offset, recorded in `meta`.

## 7. Renderer = Blender only, MuJoCo = headless physics (owner decision 2026-11-20)

Finding that triggers this (clip 0): `result_grid_sharpa_bt.mp4` is composed from MuJoCo EGL frames while
`twin_sim_bt.mp4` is Blender. Poses, timing and camera agree (relative positions within 1 %), but the look
does not (no table slab in Blender, different light, tactile-coloured hands, 640 vs 960 px). Two renderers
for one deliverable cannot be kept consistent, so there is exactly one image source.

### 7.1 Rules

1. **MuJoCo never renders in the production path.** No `mujoco.Renderer`/EGL recorders (`CompositeRecorder`,
   `EgoNeuralRecorder`, MuJoCo side-by-side). MuJoCo only steps physics and writes `sim_state.npz`.
2. **Blender produces every image** of the deliverable: robot head (D435) view, third-person view and the
   human-camera comparison view. All views come from one scene build (one `bpy` session per clip).
3. **Physics truth stays in MuJoCo**: anything the picture shows (pose, contact force, tactile heat) is data
   in `sim_state.npz`, never recomputed in Blender. Blender-only knobs are visual (`TwinAppearance`).
4. The composed grid keeps its four panels, but panels 2 and 3 come from Blender and panel 1 overlays the
   estimated hands on the real video as before; panel 4 (trajectories) is unchanged (matplotlib-style, not a
   physics renderer).

### 7.2 Data flow

```
retarget_estimated.py  ──MuJoCo rollout (headless, 50 Hz)──►  sim_state.npz (v2)
                                                                │  xpos/xquat, cam_pos/cam_mat,
                                                                │  pad_force_N per finger pad, object contacts
sim_model/ + twin_scene/ + twin_appearance.json ────────────────┤
                                                                ▼
                    twin_render.py (bpy venv, GPU Cycles, one session): views = ego, third, human
                                                                ▼
        review/{ego_d435,third_person}_bt.mp4 ──► compose_grid (real+hands | third | ego | trajectories)
```

`sim_state.npz` v2 adds the per-frame finger-pad force so Blender can colour pads (heat map) exactly as the
old MuJoCo panel did, without MuJoCo.

### 7.3 Scene parity (what the old MuJoCo scene had and Blender must also have)

* Table slab from `sim_model` (`table_top` geom, size/pose) with the appearance colour, robot floor, and
  optional room walls; lights from `TwinAppearance.light` (RSI-tuned toward the real video).
* Robot links and Sharpa hands: meshes from `sim_model`, tactile pad heat from `pad_force_N`.
* Cameras: D435 pose per frame from `sim_state` (fovy from the recorded model); third-person camera fixed
  (same lookat/azimuth as the old panel); human camera from `scene.json` for comparison and the appearance loop.

### 7.4 Cost and feasibility

* Cycles OptiX on the T4: about 4 s per 960x540 frame at 16–32 samples. 186 records x 2 views would be
  about 25 min per clip, so Blender renders every 5th record (10 fps, 37 frames, about 5 min for both views);
  `RENDER_FPS` overrides. Rendering runs after, and independently of, the physics so it never blocks a
  re-tune; it resumes from existing PNGs.
* MuJoCo gets faster when its renderers go away (three recorders per step today).
* Risks: `bpy` needs its own Python 3.11 venv (kept out of the main env, called through a subprocess);
  frame subsampling hides fast contact events (physics QA runs on the full 50 Hz `sim_state`, not on pixels).

### 7.5 Verification gates

* **Pose parity:** for keyframes, the Blender object/camera matrices equal MuJoCo `xpos/xquat` (unit test on
  the exported matrices) and the projected object centre agrees with the old EGL frame within 2 px.
* **Physics fidelity (MuJoCo side, independent of looks):** no deep penetration, objects at rest do not drift,
  lift/place events consistent with the video timing, pad force within the sensor range; reported per clip.
* **Look fidelity:** appearance score (BT.25) of the human view against the real frame, per clip.

### 7.6 Implementation status (2026-11-20)

| Piece | Where | Notes |
|---|---|---|
| `sim_state/2` | `skills/body_state.py` | adds `pad_force_N` (T,P), `pad_geom_names` (P,), `object_contact` (T,); `sim_model.json` gets `pad_geoms [{name, body}]`; v1 files still load |
| Multi-view renderer | `scripts/twin_render.py` (uses `render_twin_sim.py` helpers) | one bpy session, views `ego` (D435), `third` (lookat 0.25,-0.05,0.9, 1.5 m, az 150, el -25), `human`; outputs `review/{ego,third}_blender{tag}.mp4`, PNG cache `review/.twin_frames/` (resumable); table slab from `table_top` geom, else `TwinAppearance.table`, else `scene.json`; pad heat 0–8.2 N |
| Pipeline | `scripts/retarget_estimated.py` | with `RENDER=1` MuJoCo only steps physics and records `sim_state`/trajectory; after the report exists the script runs `export_twin_scene.py`, `twin_render.py` (interpreter from `BPY_PYTHON`) and composes `result_grid{tag}.mp4` at `RENDER_FPS` (default 10). `RENDER_BACKEND` (read by `scripts/review_render.py`, default `blender`) selects the Blender panels; `RENDER_BACKEND=mujoco` keeps the legacy MuJoCo panels |
| Fidelity | `skills/physics_fidelity.py` | sink, rest drift, lift/release timing, pad range, finiteness; CLI prints a `batch_results.md` row |

**Bug found and fixed:** `twin_scene/*.npz` meshes are in the body frame. The compiled MuJoCo geom pose
(`pos/quat` in `sim_model.json`) only re-aligns MuJoCo's own inertia-aligned mesh copy; applying it to the
`twin_scene` vertices stood the tray and other objects on edge. Scene objects now use an identity local pose;
robot links keep their geom pose.

**Known limits:** the pad heat colours the whole link mesh of a pad body, not only the pad; a residual
vertical offset of about 5 cm between the twin and the real camera view remains (camera pitch/table height
estimate, BT.18); heat colouring was tested on synthetic forces until a v2 rollout exists.

## 8. Appearance loop: LLM + RSI (BT.23–BT.27) — optional, off in the production run

Enabled per clip only by running `twin_appearance_rsi.py`, which writes `review/twin_appearance.json`; its
primitive shapes then replace the Hunyuan meshes in the render. The baseline batch runs without it.

The look of the twin is tuned against the real video without touching physics.

* **Contract** `contracts/twin_appearance.py` (`twin_appearance/1`): per object `shape`
  (`mesh|box|tray|cylinder`), `size_m`, `wall_m`, `texture` (`atlas|flat|projected`), `color_rgb`,
  `roughness`, visual `offset_m`/`yaw_deg`/`scale`; `table`, `light` (sun, world) and `room` (walls).
  `apply_edits` accepts only whitelisted paths, clamps numbers to `RANGES` and reports rejected edits.
* **Renderer** reads `review/twin_appearance.json` (default look when absent). `--view human` renders the real
  camera pose from `scene.json`; `projected` textures sample frame 0 of the video.
* **Comparator** `skills/twin_compare.py`: hand-masked L1 + SSIM + colour histogram (0 = identical) and
  per-object ROI scores; `hand_mask` from projected hand joints.
* **Loop** `scripts/twin_appearance_rsi.py <clip>`: render → score → relay request (real | twin | diff image,
  params, schema, history) → bounded edits → keep only if the score improves (else roll back). Exit code 75
  while a relay answer is missing; state in `review/twin_appearance/{state.json,best.json,round_NN.png}`.
  The LLM is the Copilot relay (D8: no external API), see `docs/COPILOT_RELAY.md`.
* **Order matters:** tune the appearance only after the layout is right (a first run on the wrongly rotated
  scene was discarded). Remaining layout error is a BT.18 topic, not an appearance one.

## 9. Scene-object modelling failures seen in the 2026-11-20 batch (clips 100, 108, 114)

Diagnosed from `scene/inventory.json`, `review/twin_scene/scene.json`, the pipeline result JSON and static Blender renders (`twin_render.py --views human --frame 0`).

**Clip 100, tongs (held object) are two flat slabs.** The Hunyuan3D input `object_crop.png` is the SAM mask of the tongs seen through the clear tupperware lid: two separate dark strips plus a dotted tip, no pivot, no colour (the tongs are blue). A single-view image-to-3D model reproduces that silhouette as two thin slabs. The tupperware is also textured from its own lid reflection, so it renders as an opaque wooden tray instead of a clear box. The tray sits ~15 cm from the tongs instead of under them (the layout offset of BT.18).

**Clip 108, the pen's stand is a lump on top of the stool.**
1. The detector names are right (the round pedestal table carries the pen). The held-object start from the hand/object track is ~19-23 cm from the round table (`hand_eye_offset_max_m` 0.18 m), so the rule in `inventory.build_scene_inventory` "geometry under the grip wins over the name" re-assigns the source to the square stool.
2. Source = stool, so the place phrase "white square table" is excluded and the receptacle falls back to the round table: roles are swapped.
3. The source container is welded and sunk to the table (`sunk_m` 0.046), and the receptacle is rebased onto the grip (`receptacle_shift_m` -0.336 m in x), so the two tables interpenetrate (overlap 7 x 13 cm) and Blender shows one merged blob.
4. The meshes themselves are rough: the round-table mesh includes the pen as a horn, the stool legs are blocks, and both are tinted warm because the crop colour is taken under warm light.

**Clip 114, fragments and ejected objects.**
- Low-score crops (`toy` score 0.28, mesh IoU 0.52) give meshes with 13 components, vertical sliver walls and floating shards; `FloaterRemover` only removes disconnected pieces, not attached needles. Most of these meshes have a volume of 1-10 % of their bounding box (thin shells).
- The same physical object is detected under several labels, so scenery boxes interpenetrate at spawn: `toy_5` and `dice` (same xy, 30 mm deep overlap), `toy` with `cube` and `toy_3`, and the tray with three toys. MuJoCo resolves the overlap by ejecting them (`physics_fidelity`: `scene_cube` drifts 835 mm and sinks 20 mm).

Next steps (TASKS.md BT.32-34): drop or merge duplicate detections and reject spawn overlaps; fix the source/receptacle role rule (trust the name when the hand-eye offset is large); gate meshes by mesh IoU, component count and volume fill, falling back to a fitted box/tray primitive with the observed colour; use a multi-frame or amodal mask for occluded held objects.
