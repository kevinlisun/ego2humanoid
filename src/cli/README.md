# `src/cli/` — `ego2humanoid` typer command line

`cli/main.py` defines `app` (typer), exposed as the console script `ego2humanoid = "cli.main:app"` in
`pyproject.toml`. It is the **contract-path** CLI (v0.2 stage contracts); the real-clip pipeline runs
through `scripts/` instead (`docs/TECH_DESIGN.md` §2, §7). No `scripts/` entry point imports it.

## Commands (`main.py`)

| Command | Behaviour |
|---|---|
| `validate EPISODE` | Parses `EpisodeInput` JSON; prints `VALID <id>: <robot_id>, pure RGB, physics_required=...` |
| `preflight EPISODE VIDEO OUT` | GPU-free check: video exists and ≥ 32 bytes, SHA256 (8 MiB blocks) equals `episode.video.video.sha256` if set, `ffprobe` finds a video stream; writes `OUT/preflight.json` (`schema_version "1.1"`, `episode_id`, `video_sha256`, `video_bytes`, `media_probe`, `robot_id`, `usage_class`, `note`) |
| `hand EPISODE VIDEO PREFLIGHT OUT` | `FAHAdapter().estimate_hands(...)` — raises `NotImplementedError` (stub) |
| `scene EPISODE VIDEO PREFLIGHT OUT` | `SimFoundryAdapter().reconstruct_scene(...)` — stub |
| `objects`, `align`, `retarget`, `replay`, `qa` | Raise `NotImplementedError` (messages cite M2/M3/M4) |
| `relay status RELAY_DIR` | Lists pending bundles in `RELAY_DIR/outbox`; exit code `75` (`vlm_relay.PENDING_RELAY_EXIT_CODE`) while any await an answer |
| `relay validate RELAY_DIR` | Checks inbox answers (`vlm_relay.validate_inbox`); exit 1 on problems, else prints `inbox OK` |
| `scene state SCENE_DIR [--duration-s 1.0] [--table-top-z-m 0.8] [--decompose/--no-decompose]` | Builds `SceneConfig` from `SCENE_DIR/objects.json` (`export_mjcf.load_scene_objects`), settles it in MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)], writes `scene_state.npz` + `scene_report.json`; exit 1 unless `all_stable` |
| `synthetic-run OUT` | `orchestration.local_runner.run_dag(synthetic_replay_dag(), OUT)`; prints the replay artefact directory |

Helper: `load_episode(path) -> EpisodeInput`. Sub-apps: `relay_app` (`relay`), `scene_app` (`scene`; note the
top-level `scene` command and the `scene` sub-app share the name).

## Inputs / outputs

- Input `episode.json` = `contracts.EpisodeInput` (see `src/contracts/README.md`, written by
  `egodex.export.export_clip`).
- `preflight.json` is the only real artefact of the stage commands; `scene state` writes next to the scene
  directory. Copilot relay protocol: `docs/COPILOT_RELAY.md`.

## Usage

```bash
uv run ego2humanoid validate episode.json
uv run ego2humanoid preflight episode.json video.mp4 runs/preflight
uv run ego2humanoid synthetic-run runs/synthetic
```

## Gotchas

- Upstream adapters (`stages/adapters.py`) are explicit stubs; the CLI never emits fake trajectories.
- The preflight manifest carries a hard-coded `schema_version` `"1.1"` while the contracts are `"1.2"`.
- `synthetic-run` is registered after the `if __name__ == "__main__"` block, so it is only available when the
  module is imported (console script, `python -c`), not through `python -m cli.main`.
- Relay commands import `stages.reconstruction.scene.vlm_relay` lazily; heavy imports (MuJoCo) happen only in
  `scene state` and `synthetic-run`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
