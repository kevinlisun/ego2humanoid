# Development Harness — Ego2Humanoid

Binding rules for every human or AI agent working in this repository. Read this file, then
`TASKS.md`, then `docs/TECH_DESIGN.md` before writing code.

## 1. Language

- Conversation with the project owner may be in Chinese.
- **Everything committed is English:** code, identifiers, comments, docstrings, log messages,
  commit messages, configs and documentation.

## 2. Change discipline

- **Minimal change:** implement the task with the smallest correct diff. No drive-by
  refactors, renames or reformatting of unrelated code.
- **No speculative abstractions:** no plugin systems, base classes or config knobs without a
  current caller. Add them when the second real use appears.
- Do not fix unrelated pre-existing issues; record them in `TASKS.md` instead.
- Contracts (`src/contracts/`) change only through a dedicated task that also
  regenerates `contracts/*.schema.json`, updates `docs/DATA_CONTRACTS.md` and adds tests.

## 3. Code style and layout

- Clean, typed Python (type hints on public functions). `ruff check` and `ruff format` clean.
- **Hard limit: 1000 lines per source file** (target ≤ 400). Split by responsibility before
  the limit is reached.
- Source is grouped by function into sub-packages (see `docs/TECH_DESIGN.md` §7). Stage logic
  lives in `src/stages/<stage>/`. Real clips run through the per-clip perception
  scripts and `scripts/retarget_estimated.py`, each run in its own image from `nodes/<node>/`
  (`DOCKER=1 scripts/run_pipeline.sh`; nodes: `perception`, `hands`, `tactile`, `retarget`, plus the
  contract-path `preflight`, `replay`).
- Comments only where the code needs clarification (frames, units, non-obvious math).
- Units and frames are explicit in names: `T_world_camera`, `q_target_rad`, `timestamps_s`.
- Stage boundaries exchange only `manifest.json` + NPZ (`allow_pickle=False`) + standard media.

## 4. Testing ladder (mandatory)

Every module climbs the ladder; a task is not done until its rung passes.

| Rung | Scope | Data | Location / marker | Runs where |
|---|---|---|---|---|
| Unit test | Every non-trivial function (math, transforms, mapping, event detection, parsing) | Hand-made arrays | `tests/unit/` (no marker) | Any machine, CPU, < 1 s each |
| Smoke test | **Every module / stage** end-to-end on synthetic input, output validated against contracts | `synthetic` generators | `tests/smoke/`, `@pytest.mark.smoke` | Any machine, CPU, < 30 s each |
| Module test | One stage on real data, metrics vs. ground truth with thresholds | EgoDex clips in `data/` (GT under `<clip>/gt/`) | `tests/module/`, `@pytest.mark.realdata` | Devbox; auto-skip if data missing |
| System test | Full pipeline on one episode | EgoDex clip; today: the 10-clip × 2-hand `retarget_estimated.py` batch (`docs/TECH_DESIGN.md` §8) | `tests/system/`, `@pytest.mark.system` | Devbox (local) |

- Synthetic data is generated at test time and deterministic (fixed seeds). Never commit
  generated binaries.
- Tests that need a GPU additionally use `@pytest.mark.gpu`.
- Default command (no data, no GPU): `uv run pytest -m "not realdata and not system and not gpu"`.

## 5. Data and artifacts

- **Large data files are never uploaded to the cloud via Git** (GitHub or any remote):
  datasets, videos, images, NPZ, meshes larger than 1 MB, model weights, run outputs.
  They live under `data/` and `runs/` (git-ignored). Only `data/README.md` is tracked.
- Uploads of episode inputs to any remote store happen only for the specific inputs a task needs, after
  owner approval.
- Every downloaded dataset/weight is recorded with source URL, version/commit and SHA256
  in `data/README.md` or `configs/submodules.lock.json`.
- Never commit secrets, tokens, API keys or credentials.
- **No external API calls from code:** pipeline code never sends frames, videos or derived
  data to third-party APIs (Gemini, Vertex AI, OpenAI, ...); no API keys in the pipeline.
  The only approved channel is the **Copilot relay** (TECH_DESIGN §6): code writes request
  bundles, a GitHub Copilot session answers them, code reads the answers. Downloading
  approved datasets/weights is allowed.
- On the devbox keep `data/` and `runs/` on local disk, not on a network share.

## 6. Dependencies and environments

- **uv** is the only Python package manager. Commit `pyproject.toml` and `uv.lock`; never
  `pip install` ad hoc. Use `uv add`, `uv sync --frozen`, `uv run`.
- Every pipeline node that is containerised has `nodes/<stage>/{Dockerfile,pyproject.toml,uv.lock,.python-version}`
  (today `perception`, `hands`, `tactile`, `retarget`, `preflight`, `replay`; lock files are
  per node, or `requirements-*.txt` / the root lock where noted in the node README). Build with
  `scripts/build_node.sh <node>`; run through `nodes/bin/<node>-python` (`scripts/node_python.sh`).
  Weights, restricted code (FAH, WiLoR, EgoTac, MANO) and data are mounted at runtime, never baked in.
- Pin versions: base images by digest/tag, uv version, upstream submodule SHAs.
- Documented exception: upstream SimFoundry installs its own conda environments; our adapter
  code in that image is still uv-managed (see `docs/TECH_DESIGN.md` §7).
- Restricted weights (FAH, SAM, MANO, ...) are mounted at runtime, never baked into images.

## 7. Physics and claims (invariants from PRD)

- Never mark `PHYSICS_VALIDATED` without an action-controlled, contact-driven rollout that
  passed independent QA. No state injection after reset; no object teleportation.
- Monocular RGB results carry `correspondence_to_real=UNVERIFIED_MONOCULAR`.
- Failures are first-class outputs: emit a report with error codes, never fake success.

## 8. Task tracking and hand-over

- `TASKS.md` is the single source of truth for progress. Before starting a task set it to
  🔄 with your agent/session tag; after finishing set ✅ and append a Session Log line.
- Leave the repo in a resumable state at all times: tests green or the failing test named
  in the task notes; partial work described in the task's `Notes` column.
- Decisions that need the owner are recorded as ❓ in `TASKS.md` "Open Questions" and asked;
  do not guess on design-level choices.

## 9. Git

- Small commits, one task per commit where possible: `G4: fade the pinch leveling after release`.
- Include the trailer `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
  for agent-authored commits.
- Never rewrite published history.

## 10. Definition of Done (per task)

1. Code + tests at the required ladder rungs pass locally.
2. `ruff check` / `ruff format --check` clean; no file > 1000 lines.
3. Contracts/docs updated if behavior or interfaces changed.
4. `TASKS.md` status, notes and session log updated.
5. No large data, weights or secrets staged.
