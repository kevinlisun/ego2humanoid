# `src/rsi/` — real-to-sim parameter iteration (Track RSI)

Tiered evaluation of one clip under a `SimParams` set, plus the "Mode B" state machine that asks an
external agent (the Copilot relay) for parameter patches. The package is simulator-agnostic: it only
talks to `contracts.sim_backend.SimBackend`; the MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)] implementation is in [`backends/`](backends/README.md).
Feedback scoring/residuals live in [`../skills`](../skills/README.md); parameter resolution in
[`../skills/params`](../skills/params/README.md).

## Modules

| File | Responsibility | Key public API |
|---|---|---|
| `types.py` | Frozen result of one iteration | `IterResult(iter, dir, feedback, entry, feedback_md, params)` |
| `runner.py` | Mode A evaluation: resolve params → pick tier → rollout → write `iter_NNN/` + history | `real2sim_dir(clip, hand="sharpa") -> Path` (`<clip>/real2sim_<hand>`); `evaluate(clip_dir, *, hand, patch, params_files, sets, rationale, parent, mode="A", freeze_plan, offsets="best", render, force_full, backend) -> IterResult`; `init_clip(clip_dir, hand, *, backend, **kw)` (full plan with defaults, then iter 0) |
| `history.py` | Append-only `history.jsonl` (one `HistoryEntry` per line) | `iter_dir(root, n)` (`iter_{n:03d}`), `append_entry`, `read_history`, `next_iter`, `best_entry` (lowest `(not success, match_score, iter)`), `format_table` |
| `schema.py` | Leaf catalogue of `SimParams` derived from its JSON schema | `Leaf(path, unit, affects, doc, range)`, `schema_leaves()` (cached), `allowed_paths()`, `normalize_path()` (drops `[i]`, `objects.<sel>.x` → `objects.*.x`), `is_known_path()`, `leaf_paths(patch)`, `current_value(params, path)` |
| `prompt.py` | Builds the relay request | `build_request(clip, hand, feedback, feedback_md, params, history, *, bundle_plan_hash, error, max_changes=3) -> (rid, prompt_md, meta)`; `schema_table`, `feedback_hash`, `request_rid` (sha256[:16]); constants `MAX_CHANGES=3`, `HISTORY_ROWS=10` |
| `answer.py` | Validates the agent's answer | `validate_answer(text, current, history_param_hashes, max_changes=3) -> (RelayParamAnswer, SimParams)`; raises `AnswerError` |
| `loop.py` | Mode B state machine | `run_loop(real2sim_dir, relay_dir, evaluate, *, hand, max_iters=20, patience=5, success_score=0.35, max_changes=3) -> int`; `LoopState`, `load_state`, `save_state` |
| `robust.py` | Robustness check of an iteration | `robust_check(clip_dir, hand, iteration, backend=None) -> dict` |

## Behaviour (as coded)

- **Tiers** (`evaluate`): `diff_params(ref.params, params)` → if `needs_plan(changes)` and not `freeze_plan`
  (or `force_full`) the backend re-plans (full tier, ~minutes); otherwise only a rollout runs. `freeze_plan`
  with plan-affecting changes sets `plan_frozen=True` and skips the re-plan.
- **Parent**: if `parent is None` the last history iter is the parent; params come from
  `iter_<parent>/params.yaml`, else the clip reference. `params_files`/`sets` restart from defaults via
  `resolve_params(..., env={})` (env overrides deliberately ignored); `patch` is deep-merged on top.
- **Offsets**: `"best"` uses `ref.best_offset_m`; `"grid"` calls `backend.search_start`.
- **Outputs per iter** (`<clip>/real2sim_<hand>/iter_NNN/`): `params.yaml`, feedback files from
  `skills.real2sim_report.write_feedback`, `rollout.npz` (arrays + `probe_*`), optional `sim_rsi.mp4` (`render`),
  `robust.json` (`robust_check`). Root also holds `history.jsonl`, `loop_state.json`, `loop_ctx.json`.
- **Loop outcomes**: `SUCCESS` (qa.success and `match_score <= 0.35`), `PATIENCE` (`iter - best_iter >= patience`),
  `MAX_ITERS`, `AGENT_STOP` (`stop` with empty patch), `AGENT_ERROR` (3 = `MAX_INVALID` consecutive bad answers).
  `run_loop` returns `0` when DONE and `75` (`PENDING_RELAY_EXIT_CODE`) while the relay answer is pending.
  Re-running is idempotent: a request id already pending/answered is not resubmitted.
- **Answer validation order**: fence strip → JSON + `RelayParamAnswer` → dotted keys expanded → empty patch
  rejected (unless `stop`) → `> max_changes` leaves rejected → unknown paths rejected → `SimParams` validation →
  params identical to current/any earlier hash rejected. Lists and scalars count as one leaf.
- **Robust check**: grid of start offsets (`backend.start_grid`) plus seeds `(1..5)` with perturbation;
  `passed` = at least `MIN_SEED_SUCCESSES=3` seed successes.
- **Prompt**: schema table is restricted to the groups of parameters named in `feedback.hints` (all leaves if none).

## Entry points

`scripts/real2sim_eval.py` (`init`, `schema`, `eval`, history/show/robust subcommands; uses `runner`, `history`,
`schema`, `robust`) and `scripts/real2sim_loop.py` (`run_loop` with `runner.evaluate`).
Relay transport: `stages.reconstruction.scene.vlm_relay` (see [`docs/COPILOT_RELAY.md`](../../docs/COPILOT_RELAY.md)).

## Env vars

`E2H_SIM_BACKEND` (backend name, default `mujoco`; `--backend` overrides). The backend sets `HAND` and
`E2H_SIM_PARAMS` for the planning subprocess.

## Gotchas

- Imports `stages.*` (vlm_relay) and `skills.*`; `contracts/` holds the pydantic models.
- Param hash comparisons use `skills.params.resolve.params_hash`; `plan_params_hash` covers plan-affecting keys only.
- `schema_leaves` is `@cache`d: schema changes need a new process.
- See [`docs/TECH_DESIGN.md`](../../docs/TECH_DESIGN.md) for the spec sections cited in docstrings (11.x, 12).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
