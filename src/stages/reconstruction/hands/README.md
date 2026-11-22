# `src/stages/reconstruction/hands/` — hand-estimator post-processing and tactile reduction

Post-processing around the hand estimators: backend selection, landmark cleanup, and EgoTac [[Zhang et al., 2026](../../../../docs/REFERENCES.md#zhang2026egotac)] tactile
reduction. The estimators themselves live in `scripts/estimate_hands_fah.py` (FAH [[Cao, 2026](../../../../docs/REFERENCES.md#cao2026fah)], the pose backend),
`scripts/estimate_hands_wilor.py` and `scripts/estimate_tactile_egotac.py`. See
[docs/TECH_DESIGN.md §3 and §4.3](../../../../docs/TECH_DESIGN.md).

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `backend.py` | read `configs/hands.yaml`, pick tactile backend, name artefacts (pure Python) | `HandsConfig(backend="fah", tactile_backend="none", tactile_npz_name="tactile_egotac.npz")`; `load_hands_config(path=None, tactile_override=None) -> HandsConfig`; `artefact_names(cfg, focal_px) -> {"pose": "hands_fah_f<focal>.npz", ["tactile": npz_name]}` |
| `smooth.py` | gap fill, wrist outlier rejection, Gaussian smoothing | `clean_landmarks(landmarks (N,21,3), valid (N,), outlier_m=0.05, median_window=9, sigma_frames=1.5) -> (smoothed (N,21,3), kept (N,) bool)`; raises `ValueError` if < 3 valid frames |
| `tactile.py` | MANO-vertex tactile -> 11 hand regions, smoothing, thresholds, fusion | `region_of_vertex_from_skinning(lbs_weights (778,16)) -> (778,)`; `aggregate_regions(force, contact_prob, region_of_vertex, prob_mode="max") -> (force_reg, prob_reg)` both `(N,2,R)`; `smooth_time(x, method="median", window=5, alpha=0.5)`; `tactile_contact_flags(p, threshold=0.5)`; `fuse_with_geometric(tactile_flags, geometric_flags, mode="or")`; `compute_tactile_regions(force, contact_prob, region_of_vertex, *, prob_mode, smoothing, window, threshold) -> dict`; `write_tactile_regions(path, arrays) -> Path` |

## Constants

- `backend.py`: `CONFIG_PATH = <repo>/configs/hands.yaml`, `POSE_BACKENDS = ("fah",)`,
  `TACTILE_BACKENDS = ("none", "egotac")`, `TACTILE_ENV_VAR = "E2H_TACTILE_BACKEND"`,
  `DEFAULT_TACTILE_NPZ = "tactile_egotac.npz"`. Precedence: argument > env var > yaml (`tactile.backend`).
- `tactile.py`: `REGION_NAMES` = thumb/index/middle/ring/pinky `_tip` (0-4), then `_pad` (5-9), then `palm`
  (10); `N_REGIONS = 11`; `FINGERTIP_VERTEX_IDS = (744, 320, 443, 554, 671)` (MANO [[Romero et al., 2017](../../../../docs/REFERENCES.md#romero2017mano)] 778-vertex mesh).

## Inputs / outputs

- Raw EgoTac (`scripts/estimate_tactile_egotac.py`): `<clip>/tactile_egotac.npz` with `force (N,2,778)` and
  `contact_prob (N,2,778)`; axis 1 is left/right hand.
- `compute_tactile_regions` returns `force_region (N,2,11)`, `contact_prob_region (N,2,11)` (clipped to
  [0,1]) and `tactile_contact_flags (N,2,11)` bool; arrays are checked by `validate_arrays("tactile_regions")`.
  `write_tactile_regions` stores them compressed in `tactile_regions.npz` plus `region_names`.
- Pose artefact `hands_fah_f<focal>.npz` has `joints3d` and `image_wh`, read by `retarget_estimated.py`.

## Callers

`scripts/retarget_estimated.py` (`load_hands_config`, `artefact_names`, `clean_landmarks`);
`REGION_NAMES` is shared with `skills/body_state.py`, `stages/manipulation/sim_tactile.py` and
`stages/qa/hand_overlay.py`; `scripts/eval_tactile_vs_hold.py` evaluates the tactile output.

## Gotchas

- `smooth_time(method="median")` needs an odd window; `"ema"` is causal.
- `fuse_with_geometric` docstring points to `align/contact.py`, which does not exist in this package; the
  geometric flags come from elsewhere in the pipeline.
- `backend.py` must stay import-light (no numpy/torch) so schedulers can read it cheaply.
- EgoTac is optional: default `tactile.backend: none` means no tactile artefact is expected.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../../docs/REFERENCES.md).

- <span id="cite-cao2026fah"></span>[Cao, 2026](../../../../docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-romero2017mano"></span>[Romero et al., 2017](../../../../docs/REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../../../../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
