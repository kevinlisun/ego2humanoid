# `src/common/` — shared IO, hashing, timebase and transform helpers

Pure numpy/pydantic helpers with no model or simulator dependencies. Used by `stages/*`,
`robot/kinematics.py`, `skills/feedback.py`, `synthetic/tracks.py`, `evaluation/hand_metrics.py` and
`orchestration/local_runner.py`. No `scripts/*.py` imports it directly. Design: `docs/TECH_DESIGN.md` §1
(invariants) and §7; bundle conventions: `docs/DATA_CONTRACTS.md`.

## Modules

| File | Responsibility | Public API |
|---|---|---|
| `hashing.py` | Streaming SHA256 (8 MiB blocks) | `sha256_file(path) -> str` |
| `io.py` | Stage bundle IO: `manifest.json` + NPZ, never pickled | `save_npz(path, arrays) -> Path`; `load_npz(path) -> dict[str, ndarray]` (`allow_pickle=False`); `write_manifest(out_dir, manifest: BaseModel) -> Path`; `write_bundle(out_dir, manifest, arrays, npz_name, bundle=None) -> Path`; `dump_json(path, obj) -> Path` (indent 2, sorted keys); `MANIFEST_NAME = "manifest.json"` |
| `timebase.py` | Video-PTS time axes and resampling | `frame_timestamps(n_frames, fps, t0_s=0.0)`; `uniform_grid(t_start_s, t_end_s, rate_hz)`; `resample_linear(t_src_s, values, t_dst_s)`; `resample_nearest(...)`; `resample_se3(t_src_s, T, t_dst_s)` |
| `transforms.py` | SE3 / Sim3 and wxyz quaternion math | `make_se3(R, t)`, `invert_se3(T)`, `compose(*Ts)`, `apply_se3(T, points)`, `make_sim3(scale, R, t)`, `sim3_scale(T)`, `quat_wxyz_to_matrix(q)`, `matrix_to_quat_wxyz(R)`, `rotation_angle(R)`, `rot_axis_angle(axis, angle_rad)`, `umeyama_sim3(src, dst, with_scale=True)` |

## Behaviour and conventions

- `T_A_B` maps coordinates in frame B into frame A. Quaternions are scalar-first `(w, x, y, z)`.
  SE3/Sim3 are `(4, 4)` homogeneous matrices; transforms may carry leading batch axes.
- `write_bundle` validates `arrays` with `contracts.arrays.validate_arrays` when the bundle name
  (argument, else `manifest.stage`) is in `BUNDLE_SPECS`, then writes the NPZ (compressed, via
  `save_npz`) at `out_dir/npz_name` and `manifest.json` next to it. Unknown bundle names are written
  unvalidated.
- `save_npz` coerces every value with `np.asarray`; it creates parent directories.
- Time axes are seconds from the source video PTS, `float64`. `uniform_grid` includes the end point when
  it falls on the grid (`floor(... + 1e-9) + 1` samples).
- `resample_linear` interpolates each column with `np.interp` (clamps outside the source range); not for
  rotations or booleans. `resample_se3` lerps translation, slerps rotation (flips quaternion hemisphere
  for continuity) and clamps `t_dst_s` to the source range. `resample_nearest` picks the nearest source
  sample (works for bool/int data).
- `umeyama_sim3(src, dst)` takes `(N, 3)` point sets and returns the `(4, 4)` Sim3 mapping `src -> dst`
  (SE3 if `with_scale=False`); `sim3_scale` reads the scale back from the rotation block.

## Gotchas

- `sha256_file` is the hash stored in `ArtifactRef.sha256` and checked by `ego2humanoid preflight`
  (which re-implements the same 8 MiB loop inline).
- `src/common` shadows third-party packages named `common` when on `PYTHONPATH` (the EgoTac [[Zhang et al., 2026](../../docs/REFERENCES.md#zhang2026egotac)] node
  runs with an empty `PYTHONPATH` for that reason).
- `io.py` imports `contracts.arrays`; keep `contracts` free of imports from `common` to avoid a cycle.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
