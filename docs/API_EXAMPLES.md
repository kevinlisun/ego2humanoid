# Integration API Examples and Failure Contracts

> **Status: archival.** The following is the **target architecture** of the archived v0.2 plan (OmniGibson
> [[Li et al., 2022](REFERENCES.md#li2022behavior1k)], SPIDER [[Pan et al., 2025](REFERENCES.md#pan2025spider)]), kept for reference. The stubs live in `src/stages/adapters.py` and raise `NotImplementedError`; the real pipeline does not use them (`docs/TECH_DESIGN.md`, `run_pipeline.md`). The OmniGibson parts were dropped; MuJoCo is the only engine. The M0 asset probe that exists is `src/robot/asset.py::probe_robot`. Actual source entry points and parameters must be pinned and examined in M0 before filling the methods.

## Robot asset bootstrap (M0)

```python
# Logical pseudocode; real robotics library usage is to be implemented.
asset = probe_g1_dex3_mjcf("assets/g1_dex3.xml")
assert asset.n_actuated_joint_dof == 43
assert len(set(asset.joint_names)) == 43
og_asset = import_g1_dex3_to_omnigibson(asset)
assert run_fk_parity_tests(asset, og_asset, sampled_legal_poses=10)
freeze_asset_mapping(asset, og_asset, "robot_joint_order.json")
```

## Stage adapter call pattern

```python
from pathlib import Path
from contracts import EpisodeInput
from stages.adapters import FAHAdapter, SimFoundryAdapter

episode = EpisodeInput.model_validate_json(Path("episode.json").read_text())
hands = FAHAdapter().estimate_hands(episode, Path("video.mp4"), Path("runs/001/hands"))
scene = SimFoundryAdapter().reconstruct_scene(episode, Path("video.mp4"), Path("runs/001/scene"))
# NOTE: these methods deliberately raise NotImplementedError in the v0.2 package.
```

## Adapter requirements

- Every adapter **reads** a versioned JSON bundle manifest, **loads** named NPZ arrays using `allow_pickle=False`, validates frames/units/time, and writes a new immutable bundle.
- `ObjectMotionAdapter` must emit `ObjectMotionBundle`, not a static pose from scene reconstruction.
- `RegistrationAdapter` writes scale separately from `T_sim_world_Rt` and fails when reaching/scale assumptions are invalid.
- `SPIDERAdapter` consumes hands **and objects + alignment**, outputs **43-column** target joint array with authoritative joint order.
- `OmniGibsonReplayAdapter` consumes retargeted **commands**, not state-replay frames, outputs actual robot and object states.
- `PhysicsQAAdapter` independently derives grasp/lift/translate/settle metrics from achieved states and records 5 rollout outcomes. Its accepted dataset result is an explicit `QAReport`.

## Error codes and operator action

| Code | Origin | Action |
|---|---|---|
| `BLOCKED_LICENSE` | Preflight/weights | Do not dispatch; review permitted use |
| `HAND_UNTRACKED` | FAH | Select shorter/clearer egocentric clip; retain diagnostic overlay |
| `SCENE_MESH_INVALID` | SimFoundry | Review object collision proxy and scene scale |
| `OBJECT_POSE_AMBIGUOUS` | Object tracker | Improve/curate visible textured object; do not substitute fake pose |
| `SCALE_UNCERTAIN` | World alignment | Record prior and review; no metric-real claim |
| `ARM_IK_UNREACHABLE` | Reachability | Reject or adjust robot initial placement under logged policy |
| `ROBOT_ASSET_MISMATCH` | SPIDER↔OG | Re-run robot M0 mapping tests |
| `RETARGET_FAILED` | SPIDER | Review contact mapping/IK/object geometry |
| `PHYSICS_REPLAY_FAILED` | OmniGibson/QA | Retain attempted rollouts, controllers, contacts, failure evidence |

**Important:** A `PASS` from FAH or scene reconstruction is not final acceptance. Only physics QA of an action-controlled rollout can mark the integrated dataset episode accepted.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](REFERENCES.md).

- <span id="cite-li2022behavior1k"></span>[Li et al., 2022](REFERENCES.md#li2022behavior1k). *BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities (OmniGibson).*
- <span id="cite-pan2025spider"></span>[Pan et al., 2025](REFERENCES.md#pan2025spider). *SPIDER: Scalable Physics-Informed Dexterous Retargeting.*
