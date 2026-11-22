# `tests/smoke/` — end-to-end stage tests on synthetic input

Each test drives a whole stage (or a small pipeline) with generated data; none reads `data/`. Run:
`python -m pytest tests/smoke -q` (all) or `python -m pytest tests -m smoke -q` (only files carrying the `smoke`
marker). Setup, markers, `conftest.py`: [`../README.md`](../README.md). `MUJOCO_GL=egl` is set by `conftest.py`.
No test here needs a GPU.

## Files

| File | Marker | Covers | Requirements / skip rule |
|---|---|---|---|
| `test_real2sim_loop.py` | `smoke` | Mode B `rsi.loop.run_loop` with a `FakeRunner` evaluator and file relay in `tmp_path`: `test_loop_to_success`, `test_invalid_answers_then_agent_error` (3 bad answers → `AGENT_ERROR`), `test_patience_and_agent_stop`, `test_max_iters`; checks `loop_state.json`, history and exit codes ([`src/rsi`](../../src/rsi/README.md)) | pure Python (pydantic, numpy) |
| `test_synthetic_video.py` | `smoke` | `synthetic.video`: mp4 write/read roundtrip keeps frame count, pts spacing (1/30 s) and box position | video extra (cv2/ffmpeg writer) |
| `test_local_runner.py` | – | `orchestration.dag.Dag` ordering + unknown dependency, runner stops on failure, `synthetic_replay_dag` end-to-end (`qa_report.json`, `render.mp4`) | `video`/`sim` extras (renders) |
| `test_scripted_pick_place.py` | – | Scripted pick-and-place of a cube: ≥3/5 seeded rollouts succeed (`detect_pick_place`), `StateInjectionGuard` counts injected writes, `replay_episode` writes a validated bundle (`actual_states.npz`, `manifest.json`, `render.mp4`, `PHYSICS_VALIDATED`, `synthetic` source) | `mujoco` (`importorskip`); EGL for the render |
| `test_plan_stance.py` | – | Locomotion planner finds a feasible stance for `golden_task` on `configs/scenarios/tabletop_v1.json` (`visible_fraction >= 0.4`) ([scenario config](../../configs/scenarios/README.md)) | `mujoco`; G1 assets |
| `test_dwbc_walk.py` | – | DWBC policy: gravity-in-body identity, stand 2 s then walk forward (pelvis z 0.65–0.85, x > 0.5 m) | `onnxruntime`, `mujoco`, GR00T-WholeBodyControl submodule (`DWBC_ROBOT_DIR/policy`), else skipped |
| `test_locomotion_goto.py` | – | `executor.goto` reaches `(1.5, 1.0, 1.57)` within `POS_TOL_M`/`YAW_TOL_RAD` | same as above |
| `test_approach_episode.py` | – | `run_episode` on the golden task: plan → walk → arrive → arm keyframes without falling; object ends within 5 cm of the goal | same as above |

## Fixtures and helpers

No pytest fixtures besides `tmp_path`. Local helpers: `FakeRunner` (scripted `IterResult` scores, writes history) in
`test_real2sim_loop.py`; `_cfg()`/`CUBE` scene in `test_scripted_pick_place.py`.

## Gotchas

- Only `test_real2sim_loop.py` and `test_synthetic_video.py` are tagged `smoke`; `-m smoke` skips the rest.
- The DWBC tests need the git submodule and ONNX policy files (see `HARNESS.md` §5 / `configs/submodules.lock.json`);
  they are skipped, not failed, without them.
- Not every smoke test is hermetic about assets: planner/locomotion tests need the G1 assets
  (`scripts/fetch_g1_assets.sh`).
- Related: [`../unit/README.md`](../unit/README.md) for per-module tests.
- Related: [`README.md`](../../README.md) Tests section.
- Locomotion smoke tests run MuJoCo for simulated seconds and can take a minute or more on CPU.
