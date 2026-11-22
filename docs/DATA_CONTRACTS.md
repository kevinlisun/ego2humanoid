# Artifact / Array Contract v1.2

> **As-built note.** These contracts (`src/contracts/`, validated by
> `src/contracts/arrays.py`) govern the stage CLI / synthetic path. The real-clip pipeline
> (`scripts/retarget_estimated.py`) writes per-clip files instead of stage bundles; see
> `docs/TECH_DESIGN.md` §4.9 and `docs/ARCHITECTURE.md` §5.
> Of the bundles below it validates and uses only `tactile` (EgoTac input) and writes an `alignment/` folder (`alignment.json` with
> `T_sim_camera`, `table_top_z_m`, `scale`, plus `contact_flags.npz`). Its joint arrays have 43
> (Dex3) or 73 (Sharpa Wave) columns in `src/robot/layout.py` actuator order, not a fixed 43. It also writes the RSI sim bundle (`real2sim_<hand>/sim_bundle.{json,npz}`, format in the
> docstring of `src/stages/manipulation/bundle.py`).

The Pydantic models in `src/contracts/` (`episode.py`, `bundles.py`, `robot.py`, `qa.py`, ...) define the authoritative JSON structures. Stage output is `manifest.json` plus named NPZ arrays (`numpy.load(..., allow_pickle=False)`). **All timestamps are source-video PTS seconds**, except explicitly declared resampled control times.

| Bundle | Required NPZ keys | Shape | Unit / notes |
|---|---|---|---|
| `hands` | `timestamps_s`, `frame_ids`, `joints2d_px`, `joints3d_est`, `valid`, `confidence` | `(N,)`, `(N,)`, `(N,2,21,2)`, `(N,2,21,3)`, `(N,2)`, `(N,2)` | `joints3d_est` may be camera-relative or world; `hand_3d_frame` MUST declare which; `metric_scale_confirmed=false` for RGB |
| `hands` | `T_world_camera`, `contact_prob` | `(N,4,4)`, `(N,2,V)` | Optional keys; `V` matches MANO topology when provided; contact is probability, NOT force |
| `tactile` | `force`, `contact_prob` | `(N,2,778)`, `(N,2,778)` | EgoTac [Zhang et al., 2026](REFERENCES.md#zhang2026egotac) per-vertex output on MANO [Romero et al., 2017](REFERENCES.md#romero2017mano) topology, hand order `[left, right]`; `contact_prob` in `[0,1]`; optional `timestamps_s`, `region_of_vertex (778,)`; `TactileBundle` carries `source` and `region_names` |
| `tactile_regions` | `force_region`, `contact_prob_region`, `tactile_contact_flags` | `(N,2,R)`, `(N,2,R)`, `(N,2,R)` bool | Per-region aggregation of `tactile` (`src/stages/reconstruction/hands/tactile.py`); region order = `TactileBundle.region_names` |
| `scene` | JSON scene descriptor and rigid/collision assets | files | Store simulation object IDs, scene frame and physics-prior confidence |
| `objects` | `timestamps_s`, `T_world_object`, `valid`, `visibility` | `(N,)`, `(N,O,4,4)`, `(N,O)`, `(N,O)` | 6D pose from monocular fit with residual/confidence, never ground-truth force |
| `alignment` | `T_sim_world_Rt`, `scale_sim_per_world`, `T_sim_robot_base`, `contact_flags` | `(4,4)`, scalar, `(4,4)`, `(N,2)` | `scale` separate from rigid transform; must record source and uncertainty |
| `retarget` | `timestamps_s`, `q_target_rad`, `wrist_T_sim`, `base_T_sim`, `qvel_target_rad_s` | `(M,)`, `(M,43)`, `(M,2,4,4)`, `(M,4,4)`, `(M,43)` | q order = pinned `robot_joint_order.json`, arm/body/fingers together |
| `simulation` | `timestamps_s`, `q_command_rad`, `q_actual_rad`, `dq_actual_rad_s`, `object_T_sim`, `wrist_actual_T_sim` | `(K,)`, `(K,43)`, `(K,43)`, `(K,43)`, `(K,O,4,4)`, `(K,2,4,4)` | Actual from physics step, not reference; contacts/forces optional additional arrays |

### Validity and semantics

- `T_A_B`: homogeneous rigid transform from B to A. In every 3×3 rotation block, `R.T @ R ≈ I`, `det(R)≈1`, last row `[0,0,0,1]`; use float64 for transforms where possible.
- `joints3d_est` must not be called *metric ground truth*. Include `world_scale_observability`, `scale_source`, `scale_sim_per_world`, and uncertainty interval in manifests.
- `q_target_rad` = retargeted desired joint pose; `q_command_rad` = controller command actually sent; `q_actual_rad` = post-simulation measured state.
- `object_reference_T_sim` belongs to trajectory/alignment; `object_T_sim` in simulation is achieved state, never used as a direct control action.
- `RobotAssetSpec` (`src/contracts/robot.py`) fixes `robot_id=unitree_g1_dex3_v1`, `joint_count=43`, but actual `joint_names` comes from robot asset probe and must be recorded in joint order manifest (not fabricated in input JSON).
- Joint angular arrays assume joints are revolute. If pinned model contains non-revolute DOFs, amend contract/mapping and tests before production, do not silently label meters as radians.

### Required `SimRunBundle` evidence (`src/contracts/qa.py`)

- `rollout_mode=ACTION_CONTROLLED`
- `physics_step_count>0`
- `object_driven_by_robot_contacts=true`
- `state_injection_after_reset=false`
- `rollouts_attempted>=5`
- `rollouts_succeeded>=3` (research MVP threshold)
- Each rollout records grasp/lift/translate/place temporal events and failure reasons.
- `QAReport.replay_validity` is set by QA, not by model output alone; physical success is scoped to the **estimated scene**.

### v1.2 changes

- `StageBundle.input_source` (`rgb_estimated` | `oracle_gt`) on every bundle.
- `QAReport.object_source` (`perceived` | `grasp_proxy` | `synthetic`); only `perceived` may be `accepted_for_dataset`.
- `world_scale_observability` also allows `METRIC_GT`; `scale_method` also allows `oracle_gt`.
- `SceneBundle.scene_format` also allows `mjcf`.
- `RobotAssetSpec`: OmniGibson (OG) fields optional; new `sim_backend` (`mujoco` | `omnigibson`).
- `SimRunBundle.sim_backend`; `RobotTrajectoryBundle.ctrl_npz_key` (optional actuator targets `ctrl(M,U)`).
- Array specs are machine-checked by `contracts.arrays` (`validate_arrays`, `validate_npz`).

## Loco-manipulation contracts (Track L)

Defined in `src/contracts/locomanip.py`; JSON schema in the repo-root `contracts/locomanip.schema.json` (regenerate with `uv run python scripts/gen_schema.py`).

- `ManipulationTask` / `TaskKeyframe`: object-relative hand keyframes (`T_object_hand`, 4x4) per phase (`pre_grasp`, `grasp`, `lift`, `place`, `release`).
- `PoseGoal`: chosen static stance (base x, y, yaw; pelvis height; waist rpy), score, visible fraction and reject-reason histogram.
- `ArrivalReport`: arrival gate result; `failure` is set exactly when `arrived` is false.
- `LocoManipFailure`: failure taxonomy (`NO_FEASIBLE_STATIC_MANIPULATION_POSE`, `FOV_OCCLUSION`, `NO_NAVIGATION_PATH`, `LOCOMOTION_FALL`, `ARRIVAL_MISALIGNMENT`, ...).

## Digital-twin asset package (Track BT)

Defined in `src/stages/reconstruction/objects/twin_schema.py` (`AssetSpec`); JSON schema in the repo-root `contracts/twin_asset.schema.json` (regenerate with `uv run python scripts/gen_schema.py`).

- Layout `data/assets/twin/<id>/` (git-ignored; only the `asset.yaml` index and tiny test assets are committed): `asset.yaml`, `visual.glb` (Z-up, origin at the bbox bottom centre, metres), `collision/*.obj` (<= 32 CoACD pieces), `preview.png`.
- `asset.yaml` (`twin_asset/1`): `id` (= directory name), `category`, `synonyms`, `colour_hsv` (hue deg, sat, val), `size_m` (authoritative bbox scale), `rest_poses` (probabilities sum to 1), `physics` (exactly one of `density_kg_m3` / `mass_kg`, `friction`, `restitution`, `source`), optional `hollow.inner_floor_z_m`, `licence`, `author`, `sha256` (file name -> digest).
- `validate_asset_dir(path)` returns the list of problems (missing files, id/directory mismatch, piece count, digest mismatch).


## Simulation parameters (Track RSI)

Defined in `src/contracts/sim_params.py` (`SimParams`, `sim_params/1`); JSON schema in `contracts/sim_params.schema.json`, defaults in `configs/sim_params/default.yaml` (regenerate with `uv run python scripts/gen_schema.py` and `scripts/sim_params.py dump-defaults --write`).

- `scene.source_from` (`grip` | `name`, default `grip`, affects `sim`+`plan`): which evidence picks the source stand/container of the inventory when the task name and the object under the grip disagree (BF.6).
- `scene.max_registration_m` (m, default 0.15, range 0.02..0.5, affects `sim`+`plan`): largest grip-to-source shift the registration trusts; beyond it the inventory is dropped. Both default to the previous hard-coded behaviour.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](REFERENCES.md).

- <span id="cite-romero2017mano"></span>[Romero et al., 2017](REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
