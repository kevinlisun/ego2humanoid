# `configs/scenarios/` — synthetic loco-manipulation scenes

| File | Purpose |
|---|---|
| `tabletop_v1.json` | Table + obstacle + box + goal zone used by the locomotion planner, episode runner and smoke tests |

## `tabletop_v1.json` keys (world frame, metres, z up)

| Key | Unit | Meaning |
|---|---|---|
| `name` | – | Scenario id (`tabletop_v1`); copied to `ManipulationTask.scenario` |
| `robot_start_xyyaw` | m, m, rad | Base start pose `[0, 0, 0]` |
| `table.center_xy` | m | Table centre `[3.0, 0.0]` |
| `table.half_size` | m | Box half extents `[0.3, 0.6, 0.02]` (slab thickness 2·0.02) |
| `table.top_z` | m | Table top height `0.75`; four legs are generated under it |
| `table.rgba` (optional) | – | Colour; default `[0.6, 0.45, 0.3, 1]` |
| `obstacles[]` | m | `name`, `center_xy`, `half_size` (box standing on the floor, centred at z = half z); base planner must avoid them |
| `object.name` / `half_size` / `mass` | –, m, kg | Free-jointed box `box`, `[0.015, 0.015, 0.045]`, 0.1 kg |
| `object.xy_on_table` | m | Start offset from the table centre `[-0.2, 0.0]` |
| `goal_zone.xy_on_table` / `radius` | m | Goal disc offset `[-0.2, -0.12]`, radius 0.07; `rgba` optional; no collision |
| `decor[]`, `object_visual` (optional, not in v1) | – | Reconstructed scenery objects (shown, never collide) / visual mesh on the box; filled in by code for clip scenes |

## Code

- `src/stages/locomotion/tabletop.py` — `Scenario.load(path=DEFAULT_CONFIG)`; `table_top_z`, `object_start_xyz()`
  (table centre + `xy_on_table`, z = top + half z + 1 mm), `goal_xy()`, `build_spec(hand="dex3")`,
  `build(hand)`, `reset_object(m, d)`, `obstacle_boxes(robot_margin)` (table counts), `occupancy(extent=((-1,5),(-2,2)), res=0.05, robot_margin)`.
- Other readers of `scenario.cfg`: `locomotion/episode.py` (`robot_start_xyyaw`, `object.half_size`),
  `locomotion/plan/rank.py`, `plan/golden.py`, `clip_task.py` (`name`, `goal_zone`).
- Entry points: `scripts/loco_manip_episode.py` (scenario cases write `data/derived/egodex/loco_manip_<case>/`),
  `scripts/solve_sharpa_pinch.py`.

## Gotchas

- Keys `table`, `obstacles`, `object`, `goal_zone`, `name` are required by `Scenario`; others are optional (`.get`).
- Needs `mujoco` and the G1 assets (`scripts/fetch_g1_assets.sh`).
- Tests: `tests/unit/test_scenario_tabletop.py`, `tests/unit/test_contracts_locomanip.py`, `tests/unit/test_plan_*.py`,
  `tests/smoke/test_plan_stance.py`, `test_approach_episode.py`, `test_locomotion_goto.py`.
- Parent: [`../README.md`](../README.md).

## Related docs

- [`docs/TECH_DESIGN.md`](../../docs/TECH_DESIGN.md) — locomanipulation design (tracks L.x).
- Head camera used for visibility scoring: [`../cameras/README.md`](../cameras/README.md).
