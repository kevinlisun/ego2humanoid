# `src/orchestration/` — minimal stage DAG and local runner

Used only by the contract / synthetic path (`cli/main.py synthetic-run`) and its tests; the real-clip
pipeline is driven by shell and Python scripts (`scripts/run_pipeline.sh`, `scripts/retarget_estimated.py`),
not by this package. See `docs/TECH_DESIGN.md` §2, §7 and `docs/ARCHITECTURE.md`.

## Modules

| File | Responsibility | Public API |
|---|---|---|
| `dag.py` | Declare stages and compute a topological order | `StageFn = Callable[[Path, dict[str, Path]], None]`; `Stage(name, fn, deps=())`; `Dag.add(name, fn, deps=()) -> None` (raises `ValueError` on a duplicate name); `Dag.order() -> list[str]` (raises `ValueError` on an unknown dependency; `graphlib.TopologicalSorter`) |
| `local_runner.py` | Execute a `Dag` in-process | `run_dag(dag, root) -> dict[str, Path]`; `synthetic_replay_dag(episode_id="ep_synthetic") -> Dag` |

## Behaviour

- `run_dag` creates `root/<stage_name>/` for every stage in topological order and calls
  `fn(out_dir, {dep_name: dep_out_dir})`. It returns `{stage_name: out_dir}`. The first exception aborts the
  run; downstream stages are never executed and no outputs are fabricated.
- `synthetic_replay_dag` builds two stages:
  - `retarget` — `stages.manipulation.scripted.scripted_pick_place(cfg, cube)` → `retarget/robot_traj.npz`
    (`timestamps_s`, `q_target_rad`);
  - `replay` (depends on `retarget`) — `stages.manipulation.engine.replay_episode(cfg, t, q, out, episode_id)`;
    raises `RuntimeError("replay rejected: ...")` when `report.task_success` is false.
  The scene: a 5 cm cube (`BoxObject("cube", (0.025,)*3, mass_kg=0.1)`) at `[0.38, -0.12, 0.825]`, table top
  z = 0.80 m, robot base at `T_sim_robot_base` with pelvis z = 0.793 m.
- Imports of `numpy`, `common.io`, and the manipulation modules are done inside the function so importing
  the package stays light.

## Called by

`cli/main.py::synthetic_run` (`ego2humanoid synthetic-run <out>`; needs the `sim` and `video` extras) and
`tests/smoke`.

## Example

```python
from orchestration.local_runner import run_dag, synthetic_replay_dag
outs = run_dag(synthetic_replay_dag(), "runs/synthetic")   # {"retarget": Path, "replay": Path}
```

## Gotchas

- Stage functions must write into the `out` directory they are given; dependencies are passed as
  directories, not objects.
- There is no retry, caching or parallelism; a stage whose `fn` returns normally is considered done.
- No GPU adapters are registered; perception stages are not part of any DAG here.
