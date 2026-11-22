# `src/stages/reconstruction/scene/relay_hook/` — VLM relay import hook

A stdlib-only import hook that reroutes SimFoundry [[NVIDIA GEAR, 2026](../../../../../docs/REFERENCES.md#simfoundry2026)]'s VLM calls to the file-based Copilot relay
(`../vlm_relay.py`). Context: [docs/TECH_DESIGN.md §4.10](../../../../../docs/TECH_DESIGN.md) and the
"no external VLM API in pipeline code" decision (D8, see `../vlm_relay.py` docstring).

## Request lifecycle

1. A SimFoundry stage calls the patched client; `RelayClient.answer` looks for `inbox/<request_id>.json`.
2. Missing answer: the request is written to `outbox/<id>/` and `RelayPending` is raised.
3. The process exits with code 75; a Copilot session writes the answer; the stage is re-run and consumes it.

## Modules

| File | Responsibility | Key objects |
|---|---|---|
| `sitecustomize.py` | auto-imported by Python in every process whose `PYTHONPATH` contains this directory; installs the hook when `E2H_RELAY_DIR` is set | `_load_relay()` (loads `../vlm_relay.py` by file path as `_e2h_vlm_relay`), `_PatchLoader`, `_Finder` (meta-path finder for `simfoundry.models.vlm`), `_install()` |
| `__init__.py` | empty | - |

## Behaviour

- Only active when `E2H_RELAY_DIR` is set (`if _RELAY_DIR: _install()`); otherwise a no-op.
- After `simfoundry.models.vlm` is executed, `module.genai` is replaced by
  `SimpleNamespace(Client=RelayClient.factory(_RELAY_DIR))` and `module.resolve_gemini_auth` is stubbed to
  `({}, "relay")`, so no credentials are needed.
- An `excepthook` turns an uncaught `RelayPending` into `os._exit(75)` (`PENDING_RELAY_EXIT_CODE`) after
  writing `RELAY PENDING: ...` to stderr.

## Caller

`../simfoundry_runner.py::run_round` sets `E2H_RELAY_DIR=<root>/<scene>/relay` and prepends `HOOK_DIR`
(this directory) to `PYTHONPATH` for the SimFoundry subprocess; `scripts/run_simfoundry_clip.py` drives it.

## Gotchas

- Must stay stdlib-only: it runs inside SimFoundry's own conda environments, not the project env.
- `vlm_relay.py` is loaded by path (not by package import) for the same reason; keep it free of project imports.
- Another `sitecustomize` earlier on `sys.path` would shadow this one.
- Verify it is active by running a stage with `E2H_RELAY_DIR` set and checking that `outbox/` fills instead of
  a Gemini auth error appearing.
- Tests/docs for the relay protocol itself belong to `../vlm_relay.py` (`validate_inbox` checks answers).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../../../docs/REFERENCES.md).

- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](../../../../../docs/REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
