# `src/` — package map

Importable code of Ego2Humanoid. Design context: `docs/TECH_DESIGN.md` §7 (source layout and
environments); rules: `HARNESS.md` §3 (≤ 1000 lines per file, typed public functions).

## Import convention

`src/` itself is the import root (`[tool.setuptools.packages.find] where = ["src"]`), so packages are
imported **without a `src.` prefix**: `from contracts import EpisodeInput`,
`from stages.retargeting.kinematic import retarget_hand`, `from robot.layout import RobotLayout`.
Scripts get the root from `PYTHONPATH=$ROOT/src` (`scripts/run_pipeline.sh`, `scripts/node_python.sh`,
`nodes/*/Dockerfile`) or from the editable install (`uv sync`). The EgoTac [[Zhang et al., 2026](../docs/REFERENCES.md#zhang2026egotac)] node deliberately keeps
`PYTHONPATH` empty because `src/common` would shadow EgoTac's own `common` package
(`scripts/node_python.sh`).

## Packages

| Package | Role | README |
|---|---|---|
| `contracts/` | Pydantic stage bundles, enums, NPZ array specs, SimParams / SimBackend / twin-appearance contracts | [contracts](contracts/README.md) |
| `common/` | NPZ+manifest IO, SHA256, timebase resampling, SE3/Sim3/quaternion math | [common](common/README.md) |
| `synthetic/` | Deterministic generators for unit/smoke tests | [synthetic](synthetic/README.md) |
| `egodex/` | EgoDex [[Hoque et al., 2025](../docs/REFERENCES.md#hoque2025egodex)] HDF5 reader, exporter to the pipeline layout, clip selector | [egodex](egodex/README.md) |
| `evaluation/` | Hand-pose metrics against EgoDex ground truth | [evaluation](evaluation/README.md) |
| `robot/` | G1 + Dex3 / Sharpa Wave [[Sharpa](../docs/REFERENCES.md#sharpa-wave)] model helpers: probe, layout, IK, pinch, collision, head camera, DWBC | [robot](robot/README.md) |
| `stages/` | Stage logic: reconstruction, retargeting, locomotion, manipulation, qa, adapters | [stages](stages/README.md), [retargeting](stages/retargeting/README.md) |
| `orchestration/` | Minimal stage DAG + in-process runner (synthetic/contract path) | [orchestration](orchestration/README.md) |
| `cli/` | `ego2humanoid` typer CLI (contract path) | [cli](cli/README.md) |
| `rsi/` | Track RSI loop (real2sim parameter tuning): runner, robust, history, loop, answer, prompt, `backends/` | [rsi](rsi/README.md) |
| `skills/` | RSI skill library (feedback, attribution, residuals, place/reach checks, `params/` SimParams resolve) | [skills](skills/README.md) |

## How they relate

```text
contracts  <- common (io validates arrays against contracts.arrays)
contracts, common <- synthetic, egodex, robot, stages, rsi, skills, cli
robot  <- stages.retargeting, stages.manipulation, stages.locomotion
stages.reconstruction -> stages.retargeting -> stages.manipulation -> stages.qa
rsi / skills -> stages.manipulation.sim_eval + contracts.sim_backend (simulator-neutral loop)
orchestration, cli -> stages.adapters, stages.manipulation (synthetic replay)
evaluation, egodex: ground truth only for clip selection and evaluation, never read by stages
```

`contracts` imports nothing from the other packages; `common.io` imports `contracts.arrays`.

## Two execution paths

- **Real clips** (the pipeline as built): per-clip scripts in `scripts/` (perception, then
  `scripts/retarget_estimated.py`, review rendering) import `stages.*`, `robot.*`, `contracts` and
  `egodex`. They exchange per-clip files (`*.npz`, `*.json` in the clip directory), not stage bundles.
- **Contract / synthetic path**: `cli/main.py` (`ego2humanoid synthetic-run`) and
  `orchestration/local_runner.py` use bundles (`manifest.json` + NPZ, `allow_pickle=False`) and
  `stages.manipulation` on a scripted trajectory. Perception adapters in `stages/adapters.py` are stubs.

## Conventions shared by all packages

- `T_A_B` maps coordinates in frame B into frame A; quaternions are `wxyz`; names carry units
  (`timestamps_s`, `q_target_rad`, `*_m`).
- Joint arrays follow actuator order of the compiled MJCF (`robot/layout.py`): 43 for Dex3, 73 for
  Sharpa Wave. `contracts.EXPECTED_G1_DEX3_DOF = 43`.
- Importing `robot` sets `MUJOCO_GL=egl` by default (`robot/__init__.py`).
- Tests mirror the layout under `tests/{unit,smoke,module,system}` (`HARNESS.md` §4).

## Gotchas

- `egodex` ground truth under `<clip>/gt/` must not be read by pipeline stages.
- `src/ego2humanoid.egg-info/` is a build artefact, not source.
- `contracts/` changes only via a dedicated task (regenerate `contracts/*.schema.json`, update
  `docs/DATA_CONTRACTS.md`, add tests).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../docs/REFERENCES.md).

- <span id="cite-hoque2025egodex"></span>[Hoque et al., 2025](../docs/REFERENCES.md#hoque2025egodex). *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.*
- <span id="cite-sharpa-wave"></span>[Sharpa](../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
