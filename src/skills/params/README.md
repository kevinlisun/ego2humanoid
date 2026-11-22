# `src/skills/params/` — `SimParams` resolution, hashing, diffing and scene edits

Everything that turns files / CLI / env into one validated `contracts.sim_params.SimParams`, and decides whether a
change needs a re-plan. Parent: [`../README.md`](../README.md). Consumed by [`../../rsi`](../../rsi/README.md).
Default values live in [`configs/sim_params`](../../../configs/sim_params/README.md).

## Files

| File | Responsibility | Key API |
|---|---|---|
| `resolve.py` | Layered resolution + hashes + diff | `resolve_params(overlays=(), sets=(), env=None, accepted=None) -> SimParams`; `parse_set("a.b=c")` (value parsed as YAML); `accepted_path(clip_dir, hand)` = `<clip>/real2sim_<hand>/accepted.yaml`; `params_hash(p)` / `plan_params_hash(p)` (sha256[:16] of canonical JSON; the latter over leaves whose `affects` contains `plan`); `diff_params(a, b) -> list[ParamChange(path, old, new, affects)]`; `needs_plan(changes)`; `_deep_merge` (dicts merge, lists/scalars replace) |
| `legacy_env.py` | Deprecated `E2H_*` vars → `SimParams` paths | `LEGACY_ENV` map, `legacy_overlay(env) -> dict` (logs a deprecation warning per hit) |
| `scene_overrides.py` | Apply `SimParams.objects` edits to scene objects | `apply_scene_overrides(objects, params, roles=None) -> list[BoxObject]`, `check_selectors(params, names, roles)`; `ALIASES=("held","receptacle","source")` |

## Resolution order (`resolve_params`)

defaults (`SimParams()`) → `accepted.yaml` (if given and exists) → file named by `$E2H_SIM_PARAMS` → legacy `E2H_*`
vars → overlay files in order → `--set` pairs. Later layers win. Pass `env={}` to ignore the environment
(`rsi.runner.evaluate` does this).

## Legacy env map (`legacy_env.py`)

| Env var | `SimParams` path | Parse |
|---|---|---|
| `E2H_ADJUST_STANCE`, `E2H_SCENE_OBJECTS` | `trajectory.adjust_stance`, `trajectory.scene_objects` | bool, true unless `"0"` |
| `E2H_PLACE_DWELL`, `E2H_PLACE_LEAD`, `E2H_RELEASE_OPEN` | `trajectory.place_dwell_frames`, `place_lead_frames`, `release_open_frames` | int |
| `E2H_RETREAT_M`, `E2H_MINTHICK`, `E2H_TABLE_MARGIN`, `E2H_TABLE_NEAR_EDGE` | `trajectory.retreat_height_m`, `min_object_thickness_m`, `table_margin_m`, `table_near_edge_m` | float |
| `E2H_SQUEEZE`, `E2H_GRIP_N`, `E2H_CARRY_ACCEL` | `trajectory.squeeze_m`, `grip_min_n`, `carry_accel_m_s2` | float |

## Object selectors (`scene_overrides.py`)

`params.objects` keys: `*` (all), role aliases `held` / `receptacle` / `source` (resolved via `roles` name map), or an exact
object name; applied least→most specific. Edits: `enabled=False` drops the object; `size_scale` scales half sizes and
mesh/collision vertices; `yaw_offset_deg` (mesh objects only, else `ValueError`); `pos_offset_m` added to `pos_m`;
`static`, `robot_collision` set when not `None`. Unknown selectors raise `ValueError`.

## Used by

`rsi/runner.py`, `rsi/answer.py`, `rsi/prompt.py` (hashes, merge); `scripts/sim_params.py` (resolve/diff/hash CLI),
`scripts/record_sim_state.py` (`resolve_params`), `scripts/retarget_estimated.py` (`params_hash`, `apply_scene_overrides`).

## Gotchas

- `_affects` is conservative: a path that stops resolving to a model field is treated as `{sim, plan}`; a fully resolved
  leaf without `affects` metadata is `sim`. `objects.<sel>.<field>` is looked up on `ObjectPhysics`.
- Legacy env vars are read only when `env` is `None` (i.e. `os.environ`); `scripts/sim_params.py` offers `schema`,
  `dump-defaults`, `resolve`, `diff`.
- Tests: `tests/unit/test_sim_params.py`.
