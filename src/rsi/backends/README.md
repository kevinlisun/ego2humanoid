# `src/rsi/backends/` — simulator backends for the real-to-sim loop

A backend implements `contracts.sim_backend.SimBackend` (`name`, `load`, `plan`, `rollout`, `search_start`,
`start_grid`). Adding a simulator = write a module and register a factory in `REGISTRY`.
Consumer: [`../README.md`](../README.md) (`runner.py`, `robust.py`).

## Files

| File | Responsibility | Key API |
|---|---|---|
| `__init__.py` | Registry and selection | `REGISTRY: dict[str, Callable[[], SimBackend]] = {"mujoco": _mujoco}`; `get_backend(name=None) -> SimBackend`; `DEFAULT_BACKEND="mujoco"`, `ENV_VAR="E2H_SIM_BACKEND"` |
| `mujoco.py` | MuJoCo [[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco)] implementation; the only real2sim module that touches `stages/manipulation` | `MujocoBackend` with the methods below; helpers `_reference`, `_case`, `_receptacle_geometry` |

## `MujocoBackend`

| Method | Signature | Behaviour |
|---|---|---|
| `load` | `(root: Path) -> SimSession` | `stages.manipulation.bundle.load_bundle(root)`; builds a `ClipReference` (meta, refs, `best_offset_m`, drop point, demo carry, release time, receptacle name) and wraps the `SimCase` as `session.native` |
| `plan` | `(clip, hand, params_file) -> None` | Runs `scripts/retarget_estimated.py <clip>` in a subprocess with env `HAND=<hand>`, `E2H_SIM_PARAMS=<params_file>` (`check=True`) — the "full" tier |
| `rollout` | `(session, params, start_xy_m, *, seed=0, perturb=False, video_path=None) -> RolloutRecord` | `run_rollout` on `case.config(params, (box, *extra))`; attaches a `ContactProbe`; optional `VideoRecorder` (50 fps, 480x640); returns arrays, probe arrays, object names, static flags, `pick_place_events(...).as_dict()`, buried fraction (only if a receptacle is present), receptacle geometry, start xy |
| `search_start` | `(session, params) -> (x, y)` | `start_search` over `offset_grid(params.trajectory.start_search_m)` |
| `start_grid` | `(session, params) -> list[(x, y)]` | Grid around `case.anchor` shifted by `params.objects["held"].pos_offset_m[:2]` (used by `robust_check`) |

## Env vars

`E2H_SIM_BACKEND` selects the backend when `get_backend()` gets no name. `MUJOCO_GL=egl` is needed for video
rendering headless (set by `tests/conftest.py` for tests).

## Gotchas

- Unknown names raise `ValueError` listing registered backends; the MuJoCo import is lazy (`_mujoco()`), so
  importing the package does not require `mujoco`.
- `rollout` re-derives the case per call via `sim_tier_extra`/`held_physics_kw`, so physics-only parameter
  changes need no re-plan; plan-affecting ones do (see `skills.params.resolve.needs_plan`).
- `REPO_ROOT = parents[3]` is used to locate `scripts/retarget_estimated.py`.
- Tests: `tests/unit/test_real2sim_backend.py` (also asserts by AST that loop modules import no simulator),
  `tests/smoke/test_real2sim_loop.py`.

## Adding a backend

1. Implement `contracts.sim_backend.SimBackend` (`name`, `load`, `plan`, `rollout`, `search_start`, `start_grid`).
2. Add a lazy factory to `REGISTRY` in `__init__.py` (import the simulator inside the factory).
3. Select it with `--backend <name>` or `E2H_SIM_BACKEND=<name>`.

## Related docs

[`docs/TECH_DESIGN.md`](../../../docs/TECH_DESIGN.md), [`docs/DATA_CONTRACTS.md`](../../../docs/DATA_CONTRACTS.md).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
