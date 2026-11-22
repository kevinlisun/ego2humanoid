# `configs/` — versioned configuration (no secrets, no weights)

Small JSON/YAML files read at run time. Subdirectories have their own READMEs with per-key tables:
[`cameras/`](cameras/README.md), [`grasp/`](grasp/README.md), [`scenarios/`](scenarios/README.md),
[`sim_params/`](sim_params/README.md). Design context: [`docs/TECH_DESIGN.md`](../docs/TECH_DESIGN.md).

## Top-level files

| File | Content | Read by |
|---|---|---|
| `hands.yaml` | `backend: fah` (pose backend); `tactile.backend` = `none` \| `egotac`; `tactile.npz_name` (artefact name, default `tactile_egotac.npz`) | `src/stages/reconstruction/hands/backend.py` (`load_hands_config`); overridden by `$E2H_TACTILE_BACKEND` or `retarget_estimated.py --tactile=<none\|egotac>` |
| `finger_mapping.yaml` | Human (MANO [[Romero et al., 2017](../docs/REFERENCES.md#romero2017mano)]) curl → Dex3 joint targets: `joint_order` (per hand), `finger_sources`, `open`/`closed` joint targets in rad (curl 0 / 1), `contact_links`, and a `sharpa` section | `src/robot/finger_map.py` (`CONFIG_PATH`), `src/stages/retargeting/sharpa_pinch.py` (`["sharpa"]`) |
| `dwbc_profile.json` | Measured Decoupled-WBC capability envelope on the G1 asset (`controller`, `asset`, `walk[]` with `vel`, `speed`, `yaw_rate`, `drift`, `falls`, ...) | written by `scripts/profile_dwbc.py`; not loaded at run time — `src/stages/locomotion/plan/candidates.py` hard-codes the supported postures it cites (L.3) |
| `physics_validation.yaml` | Proposed MVP QA gates (rollout attempts/min successes, controller dt, lift/settle/penetration thresholds); `correspondence_to_real: UNVERIFIED_MONOCULAR` | No Python reader found in `src/`, `scripts/` or `tests/`; treat as spec data |
| `submodules.lock.json` / `submodules.lock.template.json` | Pinned third-party commits / checkpoint hashes (the template is `TEMPLATE_UNPINNED_DO_NOT_DEPLOY`) | No Python reader found; see `HARNESS.md` and `docs/LICENSES.md` |

## Env vars touching configs

- `E2H_TACTILE_BACKEND` — overrides `hands.yaml: tactile.backend`.
- `E2H_SIM_PARAMS` — path of a YAML overlay applied over `sim_params/default.yaml` defaults (see
  [`src/skills/params`](../src/skills/params/README.md)).

## Gotchas

- Paths are resolved relative to the repo root from each module (`Path(__file__).resolve().parents[n]`), so run from any cwd.
- `profile_dwbc.py` writes to the relative path `configs/dwbc_profile.json`: run it from the repo root.
- Never commit weights, large data or secrets here (`HARNESS.md` §5).
- Tests: `tests/unit/test_sim_params.py` (default.yaml parity), `test_finger_map_checks.py`, `test_head_camera.py`,
  `test_scenario_tabletop.py`.

## Related docs

- [`docs/TECH_DESIGN.md`](../docs/TECH_DESIGN.md) — design and RSI spec sections.
- [`docs/TECH_DESIGN.md`](../docs/TECH_DESIGN.md) — design and RSI spec sections.
- [`docs/DATA_CONTRACTS.md`](../docs/DATA_CONTRACTS.md) — data contracts (`SimParams` is in `src/contracts/sim_params.py`).
- [`HARNESS.md`](../HARNESS.md) — repository rules (what may be committed).
- Per-key tables: [`cameras`](cameras/README.md), [`grasp`](grasp/README.md), [`scenarios`](scenarios/README.md),
  [`sim_params`](sim_params/README.md).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../docs/REFERENCES.md).

- <span id="cite-romero2017mano"></span>[Romero et al., 2017](../docs/REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
