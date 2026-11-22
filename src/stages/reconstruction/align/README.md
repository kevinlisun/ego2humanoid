# `src/stages/reconstruction/align/` — camera, gravity and table alignment

Geometry that ties the human video to the robot simulation: gravity/table anchoring of hand landmarks, the
head-camera waist pitch, the synthetic table height, and a small monocular visual odometry. See
[docs/TECH_DESIGN.md §4.1 and §4.6](../../../../docs/TECH_DESIGN.md).

## Modules

| File | Responsibility | Key public API (verified) |
|---|---|---|
| `frames.py` | camera -> sim frame: gravity rotation + table anchoring | `working_frames(pts (N,21,3)) -> (N,) bool`; `gravity_aligned_rotation(up_in_camera, forward_in_camera=(0,0,1)) -> R_sim_camera (3,3)`; `anchor_to_table(R_sim_camera, landmarks_camera (N,21,3), side, table_top_z_m=0.80, target_xy_m=(0.35, 0.18), tip_clearance_m=0.01) -> T_sim_camera (4,4)` |
| `camera_fit.py` | pick static waist pitch so the hand stays in the D435 [[Intel RealSense](../../../../docs/REFERENCES.md#realsense-d435)] view | `human_uv(points_camera (N,3), focal_px, image_wh) -> (N,2)`; `HeadCamera(model)` with `.pose(waist_pitch_rad)` and a projection method; `score_pitch(cam, waist_pitch_rad, anchors_world, layout_world, layout_uv_ref) -> CameraScore \| None`; `best_waist_pitch(cam, anchors_world, layout_world, layout_uv_ref, grid=PITCH_GRID_RAD) -> CameraScore \| None` |
| `table_fit.py` | choose table top height (and object extension) so the grip is reachable and the robot clears the table | `fit_table_height(lm_active_sim, lm_passive_sim, side, grip_z_m, object_half_z_m, support_xy, hand, waist_pitch_rad, base_z_m, exempt=None, hold=None, pinch_gap_m=None, stride=4, z_grid_m, extend_grid_m) -> TableFit` (`lru_cache`d); `table_penetration(q_rad, support, hand, base_z_m) -> (N,2,2)`; `arm_penetration(pen, active_side, exempt) -> (N,2)`; `clearance_lift(penetration_m) -> (N,)`; `near_edge_clearance(q_rad, support, hand, base_z_m, max_lo_x_m, margin_m=0.02, stride=3, max_iters=12) -> (near_edge_x, deficit)` |
| `ego_vo.py` | SIFT frame-to-frame VO with depth-lifted PnP for walking clips | `estimate_poses(frames (T,H,W,3) uint8 RGB, K, depths (T,h,w) metric z, masks=None (True = ignore), stride=1, **kw) -> VOResult`; `smooth_poses(poses, window=15)`; `points_to_world(points_cam (T,...,3), poses) -> frame-0 camera frame` |

Dataclasses: `CameraScore(waist_pitch_rad, cost, coverage_fraction, layout_err)`;
`TableFit(top_z_m, shift_z_m, ik_err_m, penetration_m, object_half_z_m)`;
`VOResult(poses (T,4,4) T_world_cam, inliers (T,), ok (T,) bool)`.

## Inputs / outputs

- `anchor_to_table`: the lowest 2 % of fingertip heights (`TIP_IDX = (4, 8, 12, 16, 20)`) during working
  frames sit `tip_clearance_m` above the table; the mean wrist xy lands at `target_xy_m` (y mirrored for the
  right hand). Returns a rigid 4x4 `T_sim_camera`. Units: metres.
- `ego_vo`: poses are camera-to-world with frame 0 = identity, OpenCV convention. Written by
  `scripts/estimate_camera_poses.py` as `<clip>/camera_poses.npz` (`poses`, `ok`, `inliers`); inputs
  `depth_moge.npz`, `intrinsics_est.json`, `hands_fah_f645.npz` (hand pixels are masked out). Optional
  argv[2] is the stride.
- `camera_fit` / `table_fit`: take the MuJoCo [[Todorov et al., 2012](../../../../docs/REFERENCES.md#todorov2012mujoco)] G1 model / retargeted joint targets; results feed
  `scripts/retarget_estimated.py` (also `stages/locomotion/base_placement.py`, `stance_adjust.py`,
  `stages/retargeting/clip.py`).

## Constants

- `camera_fit`: `WAIST_PITCH_JOINT="waist_pitch_joint"`, `PITCH_GRID_RAD = linspace(-0.45, 0.45, 19)`,
  `SAFE_MARGIN=0.10`, `NEAR_M=0.15`, `COVERAGE_WEIGHT=10`, `LAYOUT_WEIGHT=1`.
- `table_fit`: `TABLE_Z_GRID_M` 0.70..0.86 step 0.02, `PENETRATION_TOL_M=0.002`, `CLEARANCE_MARGIN_M=0.01`,
  `BODY_CLEARANCE_M=0.02`, `OBJECT_EXTEND_GRID_M=(0.02, 0.04, 0.06)`, `SIDES=("left","right")`.
- `frames`: `REST_FRAC=0.05`, `REST_RADIUS_M=0.08`, `MIN_WORK_FRAC=0.1`.

## Gotchas

- `camera_fit.HeadCamera` looks along its own -z with +y up (MuJoCo camera), not the OpenCV human-camera
  convention used by `human_uv`.
- `fit_table_height` lazily imports `stages.locomotion.base_placement` and `stages.retargeting.kinematic`
  (circular-import avoidance); keep array arguments hashable-friendly because of `lru_cache`.
- The synthetic table need not match the video: it is lowered to what the G1 can reach (top stays below the
  elbow, ≤ 0.86 m).
- No script runs this package standalone except `estimate_camera_poses.py`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../../docs/REFERENCES.md).

- <span id="cite-realsense-d435"></span>[Intel RealSense](../../../../docs/REFERENCES.md#realsense-d435). *Intel RealSense Depth Camera D435.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
