# `src/synthetic/` — deterministic generators for unit and smoke tests

Hand-made, seeded data that satisfies the bundle array contracts (`src/contracts/arrays.py`). Generated at test
time and never committed (`HARNESS.md` §4). World frame, z up, metres. Used by `tests/smoke/test_synthetic_video.py`,
`tests/unit/test_synthetic_events.py`, `stages/reconstruction/scene/viz.py`, `scripts/viz_stage12_sim.py` and
`scripts/viz_stage7_meshes.py`. Not used by the real-clip pipeline.

## Modules

| File | Responsibility | Public API |
|---|---|---|
| `tracks.py` | Synthetic pick-and-place arrays | `pick_place_positions(n, start_xy=(0,0), end_xy=(0.3,0.1), half_height_m=0.03, lift_m=0.12) -> (n,3)`; `object_tracks(n=150, fps=30.0, **kw)`; `hand_tracks(n=150, fps=30.0, **kw)`; `alignment_arrays(n=150)`; `robot_trajectory(m=100, fps=30.0, seed=0)`; `TABLE_Z_M = 0.0` |
| `video.py` | Moving-box H.264 MP4 | `moving_box_frames(n=30, width=160, height=120) -> uint8 (n,h,w,3)`; `write_mp4(path, frames, fps=30) -> Path` (libx264, yuv420p, PyAV); `read_frames(path) -> (frames, pts_seconds)` |

## Outputs (array names and shapes)

- `object_tracks`: `timestamps_s (n)`, `T_world_object (n,1,4,4)`, `valid (n,1)`, `visibility (n,1)` — the
  `objects` bundle. Motion (smoothstep phases): rest → lift → translate → lower → rest; centre z =
  `TABLE_Z_M + half_height_m + lift_m * lift`.
- `hand_tracks`: `timestamps_s`, `frame_ids`, `joints2d_px (n,2,21,2)` (zeros), `joints3d_est (n,2,21,3)`,
  `valid (n,2)`, `confidence (n,2)` — the `hands` bundle. Only the **right hand (index 1)** is valid; it
  follows the object plus a fixed random per-joint offset (seed 0, ±4 cm) and +5 cm in z.
- `alignment_arrays`: `T_sim_world_Rt` (identity), `scale_sim_per_world = 1.0`, `T_sim_robot_base`
  (yaw π, translation `[0.6, 0, 0]`), `contact_flags (n,2)` true for the right hand during frames
  `0.2n .. 0.8n`.
- `robot_trajectory`: `timestamps_s (m)`, `q_target_rad (m,43)`, `qvel_target_rad_s (m,43)` (analytic
  derivative of the sine targets, so velocities are consistent), `wrist_T_sim (m,2,4,4)`,
  `base_T_sim (m,4,4)` — the `retarget` bundle; amplitudes/frequencies drawn from `default_rng(seed)`.
- `read_frames` returns PTS from `frame.pts * stream.time_base` so tests can assert constant 30 fps PTS.

## Determinism

`pick_place_positions` is a pure function of its arguments; `hand_tracks` uses `default_rng(0)` and
`robot_trajectory` uses `default_rng(seed)`. Timestamps come from `common.timebase.frame_timestamps(n, fps)`.

## Gotchas

- Joint targets are random sinusoids, not a feasible grasp; they validate shapes/contracts only. For a
  physically meaningful synthetic pick-place use `stages/manipulation/scripted.py` (see
  `orchestration/local_runner.synthetic_replay_dag`).
- `hand_tracks` is deterministic via a fixed `default_rng(0)` regardless of `n`.
- `video.py` needs the `video` extra (`av`); `numpy`-only for `tracks.py`.
- Contract references: `docs/TECH_DESIGN.md` §8 (testing), `src/contracts/README.md`.
