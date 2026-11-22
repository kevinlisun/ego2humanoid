# `src/stages/qa/` — review and QA rendering

Review/QA rendering helpers: overlays, trajectory panel, the 2x2 review grid and where review artefacts live.
They produce videos for humans; none of them change pipeline results. See
[docs/TECH_DESIGN.md §5 Review rendering](../../../docs/TECH_DESIGN.md).

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `grid_video.py` | fixed-order 2x2 review video | `compose_grid(panels: Sequence[Panel], out_path, cell_hw=(480, 640), fps=50, notes=None) -> Path`; `fit_cell(frame, cell_hw)` (letterbox); `write_video(out_path, frames, fps=50)` (H.264 via PyAV); `read_video_rgb(path) -> list[frame]`; `Panel = tuple[str, Sequence[np.ndarray] \| None]` |
| `hand_overlay.py` | draw both estimated hands (+ tactile colours) on a BGR frame | `draw_frame_overlay(frame, joints2d, valid, score, active_side, min_score=0.7, region_force=None, force_scale_n=None, tactile_label="tactile") -> np.ndarray` (copy); `draw_hand(frame, joints2d, color, region_force=None, thickness=3)` (in place); `landmark_region_ids() -> (21,)`; `force_color(value) -> BGR`; `region_force_series(path, n_frames) -> (n_frames, 2, N_REGIONS) \| None` |
| `traj_panel.py` | task-space trajectory panel | `TrajectoryRecorder` (`on_record` hook, `.arrays()`); `IsoProjection` (fixed isometric map); `draw_trajectory_panel(traj, i, size_hw=(480, 640), table_top_z_m=0.8, fps=50.0, proj=None) -> RGB` |
| `review_paths.py` | review artefact locations and cleanup | `panels_dir(clip) -> Path` (creates); `existing_panels_dir(clip)`; `clean_review(clip, suffix, extra_files=()) -> list[Path]` |

## Panel layout (`grid_video.py`)

Top-left: source ego RGB with hand skeletons and tactile forces; top-right: retargeted robot in MuJoCo [[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco)]
(free 3/4 camera); bottom-left: robot composited into the reconstructed scene, ego view; bottom-right:
task-space trajectories. All panels share the 50 Hz sim clock, so the lists must have equal length; a `None`
panel becomes a placeholder labelled `PLACEHOLDER_NOTE = "not available in this run"`.

## Inputs / outputs

- `hand_overlay`: `joints2d` pixels, `valid`/`score` per hand, tactile from `tactile_egotac.npz` /
  `tactile_regions.npz`-style files with region force `(N, 2, 11)` (`N_REGIONS` from
  `stages.reconstruction.hands.tactile`). Colours: `ACTIVE_COLOR=(80,255,80)`, `PASSIVE_COLOR=(0,180,255)`,
  `LOW_SCORE_COLOR=(120,120,120)`.
- `traj_panel`: trail length `TRAIL_S = 2.0` s; colours for `left`, `right`, `object`, `camera`.
- Review files live in `<clip>/review/`; `clean_review` removes the `_REVIEW_STEMS`
  (`sim_mujoco`, `trajectory`, `ego_plate_composite`, `result_grid`, `side_by_side`) and panel files
  (`bg_rgbd.npz`, `ego_d435.npz`, `tl.mp4`, `tr.mp4`, `br.mp4`) before a re-render.

## Callers

`scripts/review_render.py` (grid_video, review_paths, traj_panel), `scripts/make_review_grid.py`
(grid_video, review_paths), `scripts/retarget_estimated.py` (grid_video), `scripts/render_side_by_side.py`
(hand_overlay).

## Environment variables

`E2H_BULK_DIR` (default `/cloudfile/nvme_tmp/egodex`): root of `<clip>/panels`, the large RGB-D / per-panel
captures kept off `data/`.

## Gotchas

- `panels_dir` falls back to `<clip>/review/panels` if the bulk root cannot be created;
  `existing_panels_dir` prefers the bulk copy only when it already exists.
- `draw_frame_overlay` works on BGR frames; `read_video_rgb`/`compose_grid` use RGB.
- `compose_grid` requires uniform frame counts across non-`None` panels (50 Hz clock).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
