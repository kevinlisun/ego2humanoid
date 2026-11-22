# `src/robot/` — G1 and hand model helpers (MuJoCo)

Kinematics, layout and geometry helpers for the Unitree G1 [[Unitree Robotics](../../docs/REFERENCES.md#unitree-g1)] with a **Dex3** (3-finger, 43 actuators) or a
**Sharpa Wave [[Sharpa](../../docs/REFERENCES.md#sharpa-wave)]** (5-finger, 73 actuators) hand. Numerics that depend on a compiled `mujoco.MjModel` live here;
scene building and rollouts are in `stages/manipulation`. Design: `docs/TECH_DESIGN.md` §4.2, §4.5, §6;
hand/asset facts: `assets/robots/*/README.md`, `configs/finger_mapping.yaml`. Overview of how these helpers fit
into retargeting: `docs/RETARGETING.md`.

Importing the package sets `MUJOCO_GL=egl` if unset (`robot/__init__.py`) — import it before the first MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)]
render. Called by `scripts/retarget_estimated.py` (placement, collision) and `scripts/profile_dwbc.py`
(`dwbc`, `asset`); inside `src` by `stages/retargeting`, `stages/manipulation`, `stages/locomotion`.

## Modules

| File | Responsibility | Key public API |
|---|---|---|
| `asset.py` | Probe the pinned Dex3 MJCF (`assets/robots/g1_dex3/g1_29dof_with_hand_rev_1_0.xml`): 43-DoF joint/actuator order and limits | `load_fixed_base_spec(mjcf_path=MJCF_PATH) -> MjSpec` (deletes `floating_base_joint`); `probe_robot(mjcf_path) -> RobotProbe(joint_names, actuator_names, joint_lower_rad, joint_upper_rad)`; raises if `nu != 43`, shared joints or non-hinge joints |
| `layout.py` | Actuator layout from a compiled model, no hard-coded indices | `RobotLayout.from_model(model)` (`hand = "sharpa"` if any actuator name ends `_MCP_FE`, else `"dex3"`); `.dof`, `.finger_idx(side) -> ndarray`, `.joint_idx(names) -> ndarray` |
| `kinematics.py` | Damped least-squares [[Wampler, 1986](../../docs/REFERENCES.md#wampler1986dls); [Nakamura and Hanafusa, 1986](../../docs/REFERENCES.md#nakamura1986dls)] arm IK (kinematic, no contacts) | `ArmIK(model, side="right")`: `.wrist_pose(q_full) -> (pos, R)`, `.solve(q_full, pos_m, R, iters=200, damping=1e-2, tol_m=1e-3, tol_rad=2e-3) -> IKResult(q_arm, pos_err_m, rot_err_rad, converged)`; `ARM_JOINTS`, `WRIST_BODY`, `SHOULDER_ROLL_INWARD_LIMIT_RAD = 0.05` |
| `pinch.py` | Thumb-index pinch geometry and gap solver | `PinchSolver(ik, side, side_slice)` (Dex3; tip = farthest mesh vertex of the distal link): `.tips`, `.middle_tip`, `.mirror_middle`, `.lean`, `.solve(q_full, gap_m, reg=0.02)`; `SharpaPinchSolver(ik, side, finger_idx)` (fingertip sites): `.tips`, `.lean`, `.solve(...)`; `PINCH_JOINTS`, `LEAN_WEIGHT = 0.5`, `SHARPA_LEAN_WEIGHT = 1.0` |
| `finger_map.py` | Human curl (0 open … 1 closed) → robot finger joints | `load_finger_config(path=configs/finger_mapping.yaml)`; `map_curls_to_dex3(curls, side="right", cfg=None)`; `map_curls_to_sharpa(curls, side="right", cfg=None)`; `sharpa_joint_names(side, cfg=None)`; `HUMAN_FINGERS = (thumb, index, middle, ring, pinky)` |
| `sharpa.py` | Attach Sharpa Wave hands to the G1 spec in place of Dex3 | `attach_sharpa_hands(spec, sides=("left","right"))`; `fingertip_site(side, finger)`, `fingertip_geom(side, finger)`; constants `HAND_MJCF`, `HAND_ROOT_BODY`, `ATTACH_R`, `FLANGE_POS_M`, `FINGERS` |
| `collision.py` | Arm–arm and arm–torso contact checks | `geom_sides(model, exclude_bodies)`, `arms_touching(data, side_of)`, `hand_geoms(model, side)`, `torso_geoms(model)`, `ArmCollisionChecker(model, hand_clearance_m=0.0)`: `.colliding_frames(q_traj)`, `.count(q_traj)`, `.torso_frames(q_traj, side=None)` |
| `head_camera.py` | G1 head camera (Intel D435 [[Intel RealSense](../../docs/REFERENCES.md#realsense-d435)] on `torso_link`; OpenCV pixels) | `HeadCamera.load(path=configs/cameras/g1_head_cam_v1.json)` → `height, width, fovy_deg, parent_body`; `.K`, `.fovx_deg`, `.T_world_cam(model, data)`, `.project(T_world_cam, pts_world)`; `MJ_TO_CV` |
| `floating_hand.py` | Dex3 hand on a 6-DoF floating wrist for SPIDER [[Pan et al., 2025](../../docs/REFERENCES.md#pan2025spider)]/MJWP | `build_floating_hand_spec(side) -> MjSpec`; `write_floating_hand_xml(side, out_dir) -> Path` (`dex3_<side>.xml`); `WRIST_DOF_NAMES = (tx, ty, tz, roll, pitch, yaw)` |
| `dwbc.py` | Decoupled WBC [[NVIDIA et al., 2025](../../docs/REFERENCES.md#nvidia2025gr00tn1)] lower-body policy (GR00T, ONNX) on any G1 MJCF | `load_config(robot_dir)`, `DwbcPolicy(robot_dir, cfg)`: `.reset()`, `.single_obs(...)`, `.act(q29, dq29, quat_wxyz, omega_body, cmd_vel, height_m, rpy_rad)`; `DwbcRobot(model, data, policy=None, hold_joints_q=None)`: `.set_fingers`, `.read_body`, `.apply_pd`, `.step`, `.reset_pose(base_xy_yaw, pelvis_z=0.793)`; `DwbcCommand(vel, height_m, rpy_rad)`; `gravity_in_body(quat_wxyz)` |

## Frames, units, shapes

- Joint vectors `q_full` are in **actuator order** (`(43,)` Dex3 / `(73,)` Sharpa), radians. Arm joints are
  addressed via `ArmIK.arm_idx`, hand joints via `RobotLayout.finger_idx(side)` or the `SIDE_SLICE` in
  `stages/retargeting/kinematic.py` (Dex3: right `36:43`, left `22:29`).
- `HeadCamera.T_world_cam` is `(4, 4)`; pixels are OpenCV (x right, y down, z forward); `MJ_TO_CV` flips MuJoCo's
  camera axes. Config `configs/cameras/g1_head_cam_v1.json`: 360 × 640, `fovy_deg = 42.5`, parent `torso_link`.
- `dwbc.py` constants: `CONTROL_DT_S = 0.02`, `ARM_KP/KD = 100/0.5`, `FINGER_KP/KD = 2/0.1`,
  `NUM_POLICY_JOINTS = 15`, `SINGLE_OBS_DIM = 86`; ONNX files and `g1_gear_wbc.yaml` come from
  `third_party/GR00T-WholeBodyControl/...` (`DWBC_ROBOT_DIR`).

## Gotchas

- `ArmIK` caps the inward shoulder roll at 0.05 rad (right: upper limit `min(hi, 0.05)`; left: lower limit
  `max(lo, -0.05)`) so the arm does not press into the torso.
- `ArmCollisionChecker.hand_clearance_m` is applied to the hands only on purpose (inflating the arms sweeps
  objects off the table); retargeting uses `HAND_CLEARANCE_M = {dex3: 0, sharpa: 0.03}` (`stages/retargeting/clip.py`).
- `dwbc.py` needs `onnxruntime` and the GR00T checkout; it is used by locomotion, not by the manipulation replay.
- `configs/cameras/g1_head_cam_v1.json` still names `stages/replay/mjcf_scene.py` for pose constants; the file is
  now `stages/manipulation/mjcf_scene.py` (`D435_*` constants).
- `floating_hand.py` has no non-test caller in this repo yet (tests: `test_floating_hand`, `test_sim_tactile`).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-nvidia2025gr00tn1"></span>[NVIDIA et al., 2025](../../docs/REFERENCES.md#nvidia2025gr00tn1). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*
- <span id="cite-nakamura1986dls"></span>[Nakamura and Hanafusa, 1986](../../docs/REFERENCES.md#nakamura1986dls). *Inverse Kinematic Solutions with Singularity Robustness for Robot Manipulator Control.*
- <span id="cite-pan2025spider"></span>[Pan et al., 2025](../../docs/REFERENCES.md#pan2025spider). *SPIDER: Scalable Physics-Informed Dexterous Retargeting.*
- <span id="cite-realsense-d435"></span>[Intel RealSense](../../docs/REFERENCES.md#realsense-d435). *Intel RealSense Depth Camera D435.*
- <span id="cite-sharpa-wave"></span>[Sharpa](../../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-unitree-g1"></span>[Unitree Robotics](../../docs/REFERENCES.md#unitree-g1). *G1 humanoid robot and Dex3-1 dexterous hand.*
- <span id="cite-wampler1986dls"></span>[Wampler, 1986](../../docs/REFERENCES.md#wampler1986dls). *Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods.*
