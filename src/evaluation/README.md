# `src/evaluation/` — hand-pose metrics against EgoDex ground truth

One module, `hand_metrics.py`. Ground truth is an **evaluation reference only**, never a pipeline input.
Called by `scripts/eval_hands_vs_egodex.py` (and `scripts/run_hand_eval_batch.sh` through it). Hand-pose
results and thresholds: `docs/TECH_DESIGN.md` §3 and §8; GT layout: `src/egodex/README.md`.

## `hand_metrics.py`

| Function | Returns | Notes |
|---|---|---|
| `mpjpe_m(pred, gt, valid=None)` | float (m) | Mean per-joint position error, no alignment |
| `root_relative_mpjpe_m(pred, gt, valid=None)` | float (m) | Wrist joint (`WRIST_JOINT = 0`) subtracted per frame from both skeletons |
| `sim3_aligned_mpjpe_m(pred, gt, valid=None)` | `(error_m, scale)` | One Sim3 (`umeyama_sim3`) fitted over the whole sequence; `scale` = cbrt(det R-block) of that Sim3 |
| `pa_mpjpe_m(pred, gt, valid=None)` | float (m) | Per-frame Procrustes (Sim3) alignment |
| `wrist_pose_errors(T_pred, T_gt, valid=None)` | dict | Wrist translation (m) and rotation (deg) error from `(N, 4, 4)` poses: keys `wrist_trans_mean_m`, `wrist_trans_max_m`, `wrist_rot_mean_deg`, `wrist_rot_max_deg` |
| `evaluate_hands(pred_joints, gt_joints, valid=None, pred_wrist=None, gt_wrist=None)` | dict | Full report: `frames`, `mpjpe_m`, `root_relative_mpjpe_m`, `sim3_aligned_mpjpe_m`, `sim3_scale`, `pa_mpjpe_m`, plus wrist errors when both wrist arrays are given |

## Inputs

- `pred_joints`, `gt_joints`: `(N, J, 3)` metres, same frame (EgoDex [[Hoque et al., 2025](../../docs/REFERENCES.md#hoque2025egodex)] GT camera-frame joints are in
  `gt/egodex_gt.npz::joints_camera`, hand axis `(left, right)` — pick one hand before calling).
- `valid`: `(N,)` bool, frames where both exist. Frames with non-finite values in `pred` or `gt` are
  dropped as well (`_mask` / `_err`).
- Wrist poses: `(N, 4, 4)` SE3.

## Typical use

```python
from evaluation.hand_metrics import evaluate_hands
report = evaluate_hands(pred_joints, gt_joints, valid, pred_wrist, gt_wrist)  # dict of floats
```

`scripts/eval_hands_vs_egodex.py <clip_dir>` loops over `hands_fah*.npz` and both hands, uses
`est["joints3d"][:, i]` against `gt/egodex_gt.npz::joints_camera[:n, i]` with
`valid = est["valid"][:, i] & (gt confidence > 0.5)`, and writes `<clip_dir>/hand_eval.json`.

## Gotchas

- `evaluate_hands` raises `ValueError("no valid frames to evaluate")` when nothing overlaps, so an empty
  overlap can never read as a pass.
- Plain `mpjpe_m` is only meaningful when prediction and GT share frame and metric scale; monocular
  estimates are usually judged with the root-relative / Sim3 / PA variants (`world_scale_observability`
  is `MONOCULAR_UNCERTAIN` in `contracts.bundles.HandTrackBundle`).
- Tactile evaluation (`scripts/eval_tactile_vs_hold.py`) does not use this module.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-hoque2025egodex"></span>[Hoque et al., 2025](../../docs/REFERENCES.md#hoque2025egodex). *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.*
