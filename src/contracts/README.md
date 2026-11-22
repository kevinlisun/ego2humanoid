# `src/contracts/` — boundary contracts (Pydantic models, NPZ array specs, Protocols)

The single source of truth for what crosses stage boundaries. `schema_version = "1.2"`
(`enums.SCHEMA_VERSION`); the robot is `unitree_g1_dex3_v1` with 43 commanded DoF
(`ROBOT_ID`, `EXPECTED_G1_DEX3_DOF`). Human-readable counterpart: `docs/DATA_CONTRACTS.md`; generated JSON
schemas live in the repo-level `contracts/` (not here). Related: `docs/TECH_DESIGN.md` §1, §4.11 (SimParams,
SimBackend). **Change rules** (`HARNESS.md` §2): only via a dedicated task that also runs
`scripts/gen_schema.py`, updates `docs/DATA_CONTRACTS.md` and adds tests.

`src/contracts/__init__.py` re-exports the stage bundles, enums, `EpisodeInput` family, `QAReport`,
`SimRunBundle`, protocols and `validate_physics_acceptance`. Not re-exported (import from the module):
`arrays`, `locomanip`, `real2sim`, `sim_backend`, `sim_params`, `twin_appearance`, `robot`.

## Modules

| File | Responsibility | Key public names |
|---|---|---|
| `enums.py` | Shared constants/enums | `SCHEMA_VERSION`, `ROBOT_ID`, `EXPECTED_G1_DEX3_DOF`; `StageStatus` (PASS/WARN/REVIEW/FAIL/BLOCKED_LICENSE), `ReplayValidity` (KINEMATIC_ONLY/PHYSICS_VALIDATED/PHYSICS_FAILED), `RolloutMode` (ACTION_CONTROLLED/STATE_REPLAY), `RealityCorrespondence` (UNVERIFIED_MONOCULAR/METRIC_EXTERNALLY_VERIFIED) |
| `episode.py` | Pipeline input | `ArtifactRef(uri, sha256)`, `VideoSpec`, `RobotRequest`, `EpisodeInput` (`episode_id`, `video`, `task_description`, `robot`, `usage_class="noncommercial_research"`, `physics_required=True`, `scale_policy`) |
| `bundles.py` | Per-stage manifests, all extend `StageBundle` (`schema_version`, `episode_id`, `stage`, `input_source`, `manifest_uri`, `data_uri`, `frame_count`, `status`, `warnings`, `error_codes`, `source_commit`, `image_digest`) | `HandTrackBundle`, `TactileBundle`, `SceneBundle`, `ObjectMotionBundle`, `AlignedInteractionBundle`, `RobotTrajectoryBundle` |
| `arrays.py` | Machine-checked NPZ specs | `ArraySpec(shape, kind, se3_axes)`, `HANDS_/TACTILE_/TACTILE_REGIONS_/OBJECTS_/ALIGNMENT_/RETARGET_/SIMULATION_` `REQUIRED`/`OPTIONAL` dicts, `BUNDLE_SPECS`, `validate_arrays(bundle, arrays) -> dict[str, int]`, `validate_npz(bundle, path)`, `check_se3(T, atol=1e-5)`, `ArrayContractError`, `MANO_VERTEX_COUNT = 778` |
| `qa.py` | Simulation and QA outputs | `SimRunBundle`, `QAReport` (validator `enforce_acceptance`), `validate_physics_acceptance(run) -> QAReport` |
| `robot.py` | Robot asset record | `RobotAssetSpec` |
| `protocols.py` | Stage interfaces (adapters implement them) | `HandEstimator.estimate_hands`, `SceneReconstructor.reconstruct_scene`, `ObjectMotionEstimator.track_objects`, `WorldAligner.align`, `Retargeter.retarget`, `Simulator.simulate`, `QualityGate.validate` |
| `locomanip.py` | Track L loco-manipulation | `LocoManipFailure` (StrEnum), `TaskKeyframe`, `ManipulationTask`, `PoseGoal`, `ArrivalReport` |
| `real2sim.py` | Track RSI feedback/history/answers | `REAL2SIM_FEEDBACK_VERSION = "real2sim_feedback/1"`, `HISTORY_VERSION = "real2sim_history/1"`, `PhaseSpan`, `FeedbackEvent`, `FeedbackHint`, `QaSummary`, `Real2SimFeedback`, `HistoryEntry`, `RelayParamAnswer` |
| `sim_backend.py` | Simulator-neutral loop interface | `ClipReference`, `SimSession`, `RolloutRecord`, `SimBackend` protocol: `load(root)`, `plan(clip, hand, params_file)`, `rollout(session, params, start_xy_m, *, seed=0, perturb=False, video_path=None)`, `search_start(session, params)`, `start_grid(session, params)` |
| `sim_params.py` | Versioned simulation parameters | `SIM_PARAMS_VERSION = "sim_params/1"`, `SimParams` with `objects: dict[Selector, ObjectPhysics]`, `table`, `scene`, `contact`, `pd` (`PDParams`/`PDGroups`/`PDGroup`), `trajectory` (`TrajectoryParams`) |
| `twin_appearance.py` | Blender-twin look parameters (Track BT) | `TWIN_APPEARANCE_VERSION = "twin_appearance/1"`, `TwinAppearance`, `ObjectAppearance`, `TableAppearance`, `LightAppearance`, `RoomAppearance`, `apply_edits(app, edits) -> (TwinAppearance, list[str])` |

## Array shapes (`arrays.py`; `N` frames, `M` control steps, `K` sim steps, `O` objects, `V`/`R` vertices/regions)

| Bundle | Required arrays |
|---|---|
| hands | `timestamps_s (N)`, `frame_ids (N)`, `joints2d_px (N,2,21,2)`, `joints3d_est (N,2,21,3)`, `valid (N,2)` bool, `confidence (N,2)`; optional `T_world_camera (N,4,4)`, `contact_prob (N,2,V)` |
| tactile | `force`, `contact_prob` `(N,2,778)`; optional `timestamps_s`, `region_of_vertex (778)` |
| tactile_regions | `force_region`, `contact_prob_region`, `tactile_contact_flags` `(N,2,R)` |
| objects | `timestamps_s`, `T_world_object (N,O,4,4)`, `valid (N,O)`, `visibility (N,O)` |
| alignment | `T_sim_world_Rt (4,4)`, `scale_sim_per_world ()`, `T_sim_robot_base (4,4)`, `contact_flags (N,2)` bool |
| retarget | `timestamps_s (M)`, `q_target_rad (M,43)`, `qvel_target_rad_s (M,43)`, `wrist_T_sim (M,2,4,4)`, `base_T_sim (M,4,4)`; optional `ctrl (M,U)` |
| simulation | `timestamps_s (K)`, `q_command_rad`, `q_actual_rad`, `dq_actual_rad_s` `(K,43)`, `wrist_actual_T_sim (K,2,4,4)`, `object_T_sim (K,O,4,4)` |

Named dims must agree across arrays of one bundle; `se3` arrays are checked for orthonormal rotation,
`det = 1` and last row `[0,0,0,1]`. Hand axis order is `(left, right)`.

## Used by

`scripts/gen_schema.py` (episode, locomanip, sim_params, twin_asset schemas; the last from `stages/reconstruction/objects/twin_schema.AssetSpec`), `scripts/retarget_estimated.py`,
`scripts/sim_params.py`, `scripts/twin_appearance_rsi.py`, `scripts/render_twin_sim.py`,
`scripts/loco_manip_episode.py`, `cli/main.py`, `common/io.py`, `egodex/export.py`, `synthetic/tracks.py`,
`rsi/*`, `skills/*`, `stages/{adapters,locomotion,manipulation,reconstruction}`.

## Gotchas

- `QAReport.accepted_for_dataset` only validates if status is PASS, replay is PHYSICS_VALIDATED, task
  succeeded, no error codes, `input_source == "rgb_estimated"` and `object_source == "perceived"`.
- Contract models are strict about `Literal` fields (`robot_id`, `commanded_joint_count = 43`); the Sharpa
  (73-DoF) trajectories are **not** `RobotTrajectoryBundle`s — they live in `retarget_rgb_sharpa.npz`.
- `SimParams` leaves carry `unit`, `range` and `affects` (`sim` = physics only, `plan` = also changes
  retargeting and invalidates caches) in their field metadata; the models are frozen.
- `cli/main.py` writes preflight manifests with `schema_version` `"1.1"` (literal in the CLI), not 1.2.
