# Contributing to Ego2Humanoid

Thank you for your interest in improving Ego2Humanoid. This guide summarises how to set up a development
environment, the standards a change must meet, and how to submit it. The complete internal rule set, written for
both human and AI contributors, is [`HARNESS.md`](HARNESS.md); where the two differ, `HARNESS.md` applies.

## Before you start

- Read [`docs/TECH_DESIGN.md`](docs/TECH_DESIGN.md) (the pipeline as built) and, for retargeting work,
  [`docs/RETARGETING.md`](docs/RETARGETING.md).
- Check [`TASKS.md`](TASKS.md) for open tasks and known defects, and open an issue before starting a larger change.
- Review the licence register in [`docs/LICENSES.md`](docs/LICENSES.md). Several dependencies are restricted to
  non-commercial research or to certain territories.

## Development environment

```bash
git clone --recurse-submodules https://github.com/kevinlisun/ego2humanoid.git && cd ego2humanoid
uv sync --frozen --extra video --extra sim --extra hands --extra objects --extra egodex
```

[`uv`](https://github.com/astral-sh/uv) is the only package manager: add dependencies with `uv add` and commit the
updated `uv.lock`. Model stacks with conflicting pins live in per-stage Docker images under [`nodes/`](nodes/README.md).

## Standards

- **Language.** Code, identifiers, comments, commit messages and documentation are in English.
- **Minimal, focused changes.** One concern per pull request; no unrelated refactors or reformatting.
- **Style.** Typed public functions; `ruff check` and `ruff format` must pass. Source files stay below 1000 lines.
- **Explicit frames and units** in names, e.g. `T_world_camera`, `q_target_rad`, `timestamps_s`.
- **Contracts.** Changes to `src/contracts/` regenerate `contracts/*.schema.json`, update
  [`docs/DATA_CONTRACTS.md`](docs/DATA_CONTRACTS.md) and add tests.
- **Physics claims.** Never report success without an action-controlled, contact-driven rollout that passes the
  independent QA; never write simulator state after reset.
- **Data.** Never commit datasets, videos, weights, meshes over 1 MB, run outputs or credentials. Everything under
  `data/` is git-ignored.
- **Citations.** When documentation names a published method, model or dataset, cite it against
  [`docs/REFERENCES.md`](docs/REFERENCES.md) and add a new entry there if needed.

## Tests

Every module is covered by unit tests (`tests/unit/`) and an end-to-end smoke test on synthetic input
(`tests/smoke/`). Tests that need real data or a GPU are marked and skipped when the resource is missing.

```bash
MUJOCO_GL=egl uv run --frozen --extra video --extra sim --extra hands --extra objects --extra egodex \
  python -m pytest tests -q
uv run --frozen ruff check src scripts tests && uv run --frozen ruff format --check src scripts tests
```

## Submitting a change

1. Create a branch from `main` and keep commits small, with an imperative subject line.
2. Make sure tests and lint pass locally, and update the affected documentation.
3. Open a pull request that states the motivation, the change, and how it was tested (including clip IDs and
   metrics for changes that affect simulation outcomes).

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
