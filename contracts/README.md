# `contracts/` — generated JSON Schema of the episode contract (v1.2)

Authority: `src/contracts/`; generated JSON Schema: `contracts/episode.schema.json`. Human-readable NPZ array shapes and transforms in `docs/DATA_CONTRACTS.md`.

The current contract version is 1.2 (`src/contracts/enums.py::SCHEMA_VERSION`); `docs/DATA_CONTRACTS.md` lists the v1.2 changes.

**Breaking change introduced in v1.1:** a pure egocentric MP4 is the sole required sensor input. Video width/fps/intrinsics, target object size and robot joint list are not required in the request. Robot profile is fixed to `unitree_g1_dex3_v1` / 43 DoF, but the **authoritative actual joint names** only come from parsing a pinned G1+Dex3 robot asset in M0. No placeholder 1-DoF robot is accepted.

```
PYTHONPATH=src python scripts/gen_schema.py
PYTHONPATH=src python -m cli.main validate examples/episode.template.json
```

Do not confuse structural schema validity with the missing real GPU upstream adapters, asset import, or physics rollout validation.

## Files

| File | Generated from | Content |
|---|---|---|
| `episode.schema.json` | `EpisodeInput` | Episode request: video, task description, robot profile, usage class, scale policy |
| `locomanip.schema.json` | `ManipulationTask`, `PoseGoal`, `ArrivalReport` (`src/contracts/locomanip.py`) | Locomanipulation task, pose goal and arrival report, keyed by model name |
| `sim_params.schema.json` | `SimParams` (`src/contracts/sim_params.py`) | Tunable physics and plan constants (`configs/sim_params/default.yaml`) |
| `twin_asset.schema.json` | `AssetSpec` (`src/stages/reconstruction/objects/twin_schema.py`) | Blender [[Blender, 2024](../docs/REFERENCES.md#blender)] twin asset description (`scripts/twin_import.py`) |

All four are written by `scripts/gen_schema.py`; do not edit them by hand. `examples/episode.template.json` is a sample
episode.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../docs/REFERENCES.md).

- <span id="cite-blender"></span>[Blender, 2024](../docs/REFERENCES.md#blender). *Blender — a 3D modelling and rendering package.*
