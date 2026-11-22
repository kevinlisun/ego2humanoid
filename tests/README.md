# `tests/` — test suite

Layout: [`unit/`](unit/README.md) (fast, per-module), [`smoke/`](smoke/README.md) (end-to-end on synthetic input),
plus `conftest.py`. Note: top-level `tests/` holds no test modules — everything lives in `unit/` and `smoke/`.

## How to run

From the repo root (see the *Tests* section of [`README.md`](../README.md)):

```bash
MUJOCO_GL=egl uv run --frozen --extra video --extra sim --extra hands --extra objects --extra egodex \
  python -m pytest tests -q
uv run --frozen ruff check src scripts tests && uv run --frozen ruff format --check src scripts tests
python -m pytest tests/unit -q            # unit only
python -m pytest tests -m smoke -q        # only tests tagged smoke
python -m pytest tests/unit/test_sim_params.py::test_default_yaml_matches_model_defaults
```

`pyproject.toml` (`[tool.pytest.ini_options]`): `testpaths = ["tests"]`; registered markers:

| Marker | Meaning | Used by |
|---|---|---|
| `smoke` | end-to-end stage test on synthetic input | `smoke/test_real2sim_loop.py`, `smoke/test_synthetic_video.py` (the other smoke files are not tagged) |
| `realdata` | needs real data under `data/` | no test currently tagged; auto-skipped when `data/` is empty |
| `system` | full pipeline on one episode | none currently; auto-skipped like `realdata` |
| `gpu` | needs a CUDA GPU | none currently; no GPU is needed by any current test |

## `conftest.py`

- `os.environ.setdefault("MUJOCO_GL", "egl")` — headless rendering for all tests.
- `DATA_DIR = <repo>/data`; `pytest_collection_modifyitems` adds a skip to tests with the `realdata` or `system`
  keyword when `data/` has no entry other than `README.md`.
- No shared fixtures; tests use pytest built-ins (`tmp_path`, `parametrize`) and local helpers.

## Skips and external requirements

| Condition | Tests |
|---|---|
| `mujoco` missing (`importorskip`) | `test_camera_fit`, `test_collision`, `test_head_camera`, `test_plan_ik`, `test_plan_visibility`, `test_retarget_align`, `test_scenario_tabletop`, `test_sharpa_hand`, `test_sim_tactile`, `smoke/test_plan_stance`, `smoke/test_scripted_pick_place` (others import `mujoco` directly and fail without the `sim` extra) |
| `h5py` missing | `test_egodex_reader` |
| Sharpa MJCF assets absent (`scripts/fetch_sharpa_assets.sh`) | `test_sharpa_hand` |
| EgoDex sample `data/raw/egodex/test/basic_pick_place/0.hdf5` absent | real-sample tests in `test_egodex_reader` |
| Real bundle `data/derived/egodex/basic_pick_place_104/real2sim_sharpa/sim_bundle.json` absent | `test_sim_bundle` |
| GR00T-WholeBodyControl submodule (`robot.dwbc.DWBC_ROBOT_DIR/policy`) or `onnxruntime` absent | `smoke/test_dwbc_walk`, `test_locomotion_goto`, `test_approach_episode` |
| Real data on disk | only the three rows above; every other test builds synthetic input in-process |

## Gotchas

- Tests import from `src/` (`stages`, `skills`, `rsi`, `contracts`, `robot`, `common`, ...); run from the repo root
  so relative `data/...` paths resolve.
- Do not write to `data/` from tests; use `tmp_path`.
- Each area README (`../src/<pkg>/README.md`) lists the tests that cover it.
