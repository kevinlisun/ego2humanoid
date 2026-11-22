# `configs/sim_params/` — default simulation parameters (`sim_params/1`)

| File | Purpose |
|---|---|
| `default.yaml` | Serialised defaults of `contracts.sim_params.SimParams`; a test asserts it equals the model defaults |

Source of truth is the pydantic model `src/contracts/sim_params.py` (frozen, `extra="forbid"`; every leaf has
`unit`, `range`, `affects`, `doc`). Regenerate this file with `python scripts/sim_params.py dump-defaults --write`.
`affects`: `sim` = only the physics rollout changes (cheap), `plan` = also an input to retargeting (re-plan, see
[`src/skills/params`](../../src/skills/params/README.md)). Overlay files use the same nesting; override at run time with
`$E2H_SIM_PARAMS`, `--params file`, `--set a.b=c`. Consumers: `src/rsi` ([README](../../src/rsi/README.md)),
`scripts/retarget_estimated.py` (`SIM_PARAMS`), `stages/manipulation/mjcf_scene.py` (`SceneConfig.from_params`).

## Keys

| Key | Default | Unit | Range / meaning | Affects |
|---|---|---|---|---|
| `version` | `sim_params/1` | – | fixed literal | – |
| `objects` | `{}` | – | selector (`*`, `held`, `receptacle`, `source`, or object name) → `ObjectPhysics` override | – |
| `objects.*.mass_kg` / `friction` | null | kg / 1 | 0.005–2.0 / 0.05–3.0; null keeps the estimate | sim+plan |
| `objects.*.enabled` | true | bool | false drops the scene object | sim+plan |
| `objects.*.static` / `robot_collision` | null | bool | weld/free; collide with robot; null keeps role | sim |
| `objects.*.pos_offset_m` | [0,0,0] | m | each \|v\| ≤ 0.10 | sim+plan |
| `objects.*.yaw_offset_deg` | 0 | deg | −180…180, mesh objects only | sim+plan |
| `objects.*.size_scale` | [1,1,1] | 1 | each 0.5–2.0 | sim+plan |
| `table.top_z_m` | null | m | 0.3–1.2; null = fit to hand path | sim+plan |
| `scene.source_from` | `grip` | enum | `grip` \| `name`: which evidence picks the source object | plan |
| `scene.max_registration_m` | 0.15 | m | 0.02–0.5; larger grip-to-source shift drops the inventory | plan |
| `contact.priority` | 1 | 1 | 0–5 geom priority of scene objects | sim |
| `contact.condim` | 4 | 1 | {1,3,4,6}; 4 = torsional friction | sim |
| `contact.solref` | [0.01, 1.0] | s, 1 | time const 0.002–0.1, damping 0.1–2 | sim |
| `contact.solimp` | [0.95, 0.99, 0.001] | 1,1,m | 0.5 ≤ d0 ≤ dwidth ≤ 0.9999; width 1e-4–0.01 | sim |
| `contact.min_slide_friction` | 1.0 | 1 | 0–3, floor of object sliding friction | sim+plan |
| `contact.torsion_m` / `roll_m` | 0.01 / 0.0001 | m | 0–0.05 / 0–0.01 | sim |
| `contact.impratio` | 10 | 1 | 1–100 | sim |
| `contact.cone` | `elliptic` | enum | `elliptic` \| `pyramidal` | sim |
| `contact.noslip_iterations` | 5 | 1 | 0–50 | sim |
| `contact.table_friction` | [1.0, 0.005, 0.0001] | 1,m,m | sliding, torsional, rolling (≥ 0) | sim |
| `pd.groups.<g>.{kp,kd,max_torque_nm,armature}` | hand 8/0.4/6/0.005; wrist 40/2/20/0.01; shoulder & elbow 150/6/60/0.02; waist 300/10/100/0.02; legs 300/10/150/0.02 | N·m/rad, N·m·s/rad, N·m, kg·m² | kp>0, kd≥0, torque>0, armature≥0 | sim |
| `pd.heavy_hand_wrist_scale` | 2.5 | 1 | 1–10, Sharpa wrist actuator scale | sim |
| `pd.gravity_compensation` | true | bool | robot links carry no net weight | sim |
| `pd.timestep_s` | 0.002 | s | 0.0005–0.004 | sim |
| `pd.integrator` | `implicitfast` | enum | implicitfast, implicit, euler, rk4 | sim |

### `trajectory.*` (all affect `sim+plan`; constants of `scripts/retarget_estimated.py`)

| Keys (default) | Unit | Meaning |
|---|---|---|
| `squeeze_m` (0.012), `grip_min_n` (4.0), `carry_accel_m_s2` (3.0) | m, N, m/s² | least squeeze (tip gap = width − squeeze), least grip force, held-path acceleration limit |
| `approach_frames` (30), `close_ramp_frames` (6), `close_lead_frames` (12) | frames | approach window, open→squeezed ramp, close before lift is seen |
| `open_margin_m` (0.02), `open_pad_m` (0.015), `pad_reach_m` (0.015), `tip_floor_m` (0.006), `max_open_extra_m` (0.04) | m | open aperture margin / pad allowance / pad reach / lowest tip height / extra opening |
| `hover_clearance_m` (0.015), `max_hover_rise_m` (0.06) | m | hover above object top / highest hover |
| `place_dwell_frames` (15), `place_lead_frames` (20), `place_xy_lead_frames` (25), `release_open_frames` (6) | frames | hold after release, receptacle-frame handover, xy fade-in, finger opening |
| `retreat_frames` (8), `retreat_height_m` (0.04), `retreat_blend_frames` (10) | frames, m | retreat rise and blend back to the human path |
| `bare_place_lift_m` (0.003), `max_offset_m` (0.2), `start_search_m` (0.03) | m | bare-table put-down height, relative-shift cap, start search half range |
| `min_object_thickness_m` (0.04), `slab_max_width_m` (0.025) | m | thinner objects pinch on the table; slab-proxy collision width |
| `table_margin_m` (0.3), `table_inset_m` (0.05), `table_near_edge_m` (0.1), `scene_table_margin_m` (0.08) | m | table sizing and edge margins |
| `obstacle_margin_m` (0.02), `held_margin_m` (0.02), `held_clearance_m` (0.005), `place_clearance_m` (0.01), `receptacle_opening_margin_m` (0.0) | m | keep-off / clearance margins; tub inner wall pull-out |
| `place_retry_budget_s` (1800), `place_reach_weight_m` (0.1) | s, m | no retry pass after this wall time (0 = never); stance reach score weight (0 = off) |
| `place_stance_search`, `scene_objects`, `adjust_stance` (true), `allow_held_box_fallback` (false) | bool | re-score stance on corrected place path; simulate all inventory objects; move base only if needed; allow box when no mesh |

## Gotchas

- Legacy env vars (`E2H_SQUEEZE`, `E2H_GRIP_N`, ...) map onto `trajectory.*` and are deprecated (see the params README).
- Changing any `trajectory.*`, `objects.*` placement or `table`/`scene` key triggers a re-plan (minutes); `contact`/`pd` do not.
- Full ranges for every key: `python scripts/sim_params.py schema`. Tests: `tests/unit/test_sim_params.py`.
- Parent: [`../README.md`](../README.md).
