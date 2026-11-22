# `src/egodex/` — EgoDex dataset reader, exporter and clip selector

EgoDex [[Hoque et al., 2025](../../docs/REFERENCES.md#hoque2025egodex)] clips are paired `N.hdf5` (ARKit poses) + `N.mp4` (1080p, 30 Hz). The pipeline itself uses only
the RGB video and the task sentence; everything else here is **ground truth for clip selection and
evaluation** (`docs/TECH_DESIGN.md` §1, §8; data layout in `data/README.md`). Stages must never read
`<clip>/gt/`.

Called by: `scripts/fetch_egodex.py`, `scripts/run_pipeline.sh`, `scripts/run_hand_eval_batch.sh`
(they import `egodex.*`) and the tests.

## Modules

| File | Responsibility | Public API |
|---|---|---|
| `clip_reader.py` | Read one HDF5 clip into arrays | `load_clip(hdf5_path) -> EgoDexClip`; `list_clips(root) -> list[Path]` (every `*.hdf5` with a sibling `.mp4`, sorted); `EgoDexClip` dataclass; constants `FPS = 30.0`, `FINGERS`, `HAND_JOINTS_21` |
| `export.py` | Write the pipeline layout of one clip | `export_clip(clip, out_dir) -> Path` (returns `out_dir/<clip_id>`) |
| `select.py` | Pick single-active-hand pick-and-place clips from GT hand motion | `score_clip(clip, path, min_travel=0.3, max_other=0.15, min_s=3.0, max_s=15.0) -> ClipScore`; `select_clips(root, **kw) -> list[ClipScore]`; `SIDES = ("left", "right")` |

## `EgoDexClip` arrays (`clip_reader.py`)

| Field | Shape | Meaning |
|---|---|---|
| `timestamps_s` | `(N,)` | `arange(N) / 30` (nominal, not video PTS) |
| `intrinsics` | `(3, 3)` | pinhole K of `camera/intrinsic` |
| `T_world_camera` | `(N, 4, 4)` | ARKit origin frame; camera is OpenCV-style after `joints_camera()` |
| `joints_world` | `(N, 2, 21, 3)` | hand axis order `(left, right)`; 21 MANO-like joints: wrist then thumb/index/middle/ring/little × 4 (`Knuckle`, `IntermediateBase`, `IntermediateTip`, `Tip`); EgoDex metacarpals dropped |
| `T_world_wrist` | `(N, 2, 4, 4)` | `leftHand` / `rightHand` transforms |
| `confidence` | `(N, 2)` | wrist confidence, `1.0` when the file has none |
| `attrs` | dict | HDF5 attrs (`description`, `environment`, ...); bytes decoded |

Methods: `n_frames`, `joints_camera() -> (N, 2, 21, 3)` (OpenCV: +x right, +y down, +z forward),
`active_hand() -> "left" | "right" | None` (parses `hand:<side>` in the `environment` attr).
`clip_id = f"{parent_dir_name}_{stem}"`.

## `export_clip` output (`<out_dir>/<clip_id>/`)

- `video.mp4` — copy of the source video.
- `episode.json` — `contracts.EpisodeInput` (`video.uri = "video.mp4"` with SHA256, `fps_nominal = 30`,
  `task_description` from the `description` attr, else the model default).
- `gt/egodex_gt.npz` — `timestamps_s`, `intrinsics`, `T_world_camera`, `joints_world`, `joints_camera`,
  `confidence`. Evaluation only.

## Selection rule

`score_clip` takes `active` from `active_hand()` or else the hand with the larger wrist path length, and
accepts when active wrist travel ≥ 0.3 m, other-hand travel ≤ 0.15 m and duration (frames / 30) is within
3–15 s. `select_clips` sorts accepted first, then by descending travel.

## Gotchas

- `h5py` is imported lazily inside `load_clip` (extra `egodex`).
- Camera/hand poses are in the ARKit origin frame, not gravity-aligned with the simulation; the pipeline
  re-derives gravity from RGB (`stages/reconstruction/align/frames.py`).
- Hand axis is `(left, right)` everywhere in the repo; contract arrays use the same order.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-hoque2025egodex"></span>[Hoque et al., 2025](../../docs/REFERENCES.md#hoque2025egodex). *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.*
