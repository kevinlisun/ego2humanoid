# Retargeting guide

> **Summary.** Human hand landmarks estimated from RGB are mapped frame by frame onto the joints of a Unitree G1 with
> a Dex3-1 or Sharpa Wave hand, corrected in closed loop against the perceived object, and validated by an
> action-controlled MuJoCo [[Todorov et al., 2012](REFERENCES.md#todorov2012mujoco)] rollout. This guide follows that chain step by step.

Contributor-oriented walkthrough of how a human hand video becomes robot joint targets. It explains the
concepts and where each step lives; constants and rationale stay in `TECH_DESIGN.md` §4, per-module APIs in
`src/stages/retargeting/README.md` and `src/robot/README.md`. When a document disagrees with the code, the
code is right.

## 1. What "retargeting" means here

Retargeting is **per-frame geometric mapping plus closed-loop correction, validated by a physics rollout**:

```text
hand landmarks (camera) ──► sim frame ──► wrist pose + finger curls ──► IK / pinch solve ──► q_target (50 Hz)
                                              ▲                                   │
                          corrections: placement, leveling, clearance,            ▼
                          hand-eye offset, grip timing, carry smoothing     MuJoCo rollout (PD servos)
                                                                                   │
                                                              success / lift / place error  ◄─ start-offset search
```

It is **not** an optimisation that includes dynamics or contacts. The embodiment gap (smaller finger count,
different hand size, position-controlled fingers, limited arm reach) is handled by explicit rules and
corrections listed in §4, and the result is checked in simulation. Contrast with SPIDER [[Pan et al., 2025](REFERENCES.md#pan2025spider)] (sampling-based MPC in a
physics engine) and interaction-mesh methods such as OmniRetarget [[Yang et al., 2025](REFERENCES.md#yang2025omniretarget)], where contact and dynamics are part of the
optimisation: `docs/notes/spider_integration.md` (only the export side exists in this repo; it is not used
by `scripts/run_pipeline.sh`).

## 2. Inputs and outputs

| | Content | Produced by |
|---|---|---|
| In | `landmarks_camera (N, 21, 3)` m per hand, OpenCV camera frame, OpenPose [[Simon et al., 2017](REFERENCES.md#simon2017hand)] / MANO [[Romero et al., 2017](REFERENCES.md#romero2017mano)] order | `hands` stage (FAH [[Cao, 2026](REFERENCES.md#cao2026fah)], or WiLoR [[Potamias et al., 2025](REFERENCES.md#potamias2025wilor)] fallback), `hands_fah_f<focal>.npz` |
| In | gravity `up_in_camera (3,)` | MoGe-2 [[Wang et al., 2025b](REFERENCES.md#wang2025moge2)] (`reconstruction/align`) |
| In | objects, held-object mesh, scene `objects.json` | `objects`, `scene`, `held_mesh` stages |
| In | human tactile contact probability (optional) | EgoTac [[Zhang et al., 2026](REFERENCES.md#zhang2026egotac)], `tactile_egotac.npz` |
| Out | `q_target_rad (M, 43 or 73)`, 50 Hz, actuator order | `retarget_rgb{,_sharpa}.npz` |
| Out | report: placement, IK errors, grasp timing, `sim_events`, success | `retarget_rgb{,_sharpa}.json` |
| Out | `alignment/` (`T_sim_camera`, contact flags) | `stages/retargeting/contact.py` |

Driver: `scripts/retarget_estimated.py <clip_dir>` (about 1100 lines, runs all steps below in order). Hand
choice: `HAND=dex3|sharpa`.

## 3. Step by step

### 3.1 Camera frame → simulation frame (`stages/reconstruction/align/frames.py`)

A rigid transform `T_sim_camera = [R | t]`, applied as `p_sim = R·p_cam + t` (`landmarks_camera_to_sim`).

- `R` (`gravity_aligned_rotation`): z = gravity up from MoGe; x = horizontal camera forward; y = z × x.
- `t` (`anchor_to_table`): monocular depth is not metrically trusted, so the motion is **anchored**:
  vertically the 2nd percentile of fingertip height over the *working* frames (wrist farther than 8 cm from
  the rest poses at both ends of the clip, `working_frames`) sits 1 cm above the table top (default 0.80 m);
  horizontally the mean wrist xy goes to `(0.35, ±0.18)` m (mirrored for the left hand).
- There is no scale term. Depth scale errors are absorbed later by the hand-eye correction (§3.6) and the
  start-offset search (§3.8).

### 3.2 Robot standing spot and waist pitch (`stages/locomotion/base_placement.py`, `align/camera_fit.py`)

The pelvis is fixed for the whole clip. `place_hand_motion` grid-searches the standing spot
(`X_GRID_M` × `Y_GRID_M`) and a static waist pitch; the score is wrist-IK error on a frame subsample plus an
arm-collision penalty, with the head-camera view as tie-breaker among equally reachable spots. Spots are scored
in parallel worker processes (fork). The result is a `Placement` carrying `T_sim_camera`.

### 3.3 Human landmarks → wrist frame and finger curls (`stages/retargeting/kinematic.py`)

- `hand_frame(lm, side, hand)`: x along fingers (wrist → middle MCP), y = palm normal from the index and pinky
  MCPs (orthogonalised), z = x × y. The sign follows `RIGHT_Y_FLIP[hand]`; a wrong sign rolls the hand 180°.
- `finger_curls(lm)`: per finger, summed bend angle along the landmark chain divided by a full-curl angle,
  clipped to `[0, 1]`.

### 3.4 Joint solve

- **Arm** (`robot/kinematics.py::ArmIK`): damped least squares [[Wampler, 1986](REFERENCES.md#wampler1986dls); [Nakamura and Hanafusa, 1986](REFERENCES.md#nakamura1986dls); [Buss, 2004](REFERENCES.md#buss2004ik)] on `mj_jacBody`, `dq = Jᵀ(JJᵀ + λI)⁻¹ e` with
  `λ = 1e-2`, 6-D error, step clip 0.2 rad, joint limits, ≤ 200 iterations, 1 mm / 2e-3 rad tolerance. 7 joints
  per arm, kinematic only. Warm start from the previous frame; restart from the neutral arm if the error
  exceeds `RESTART_POS_ERR_M` (1 cm).
- **Fingers** (`robot/finger_map.py`, `configs/finger_mapping.yaml`): curls → joint angles. Dex3 has 3 fingers
  (ring/pinky are folded onto the middle finger); Sharpa maps one robot finger per human finger and tucks
  middle/ring/pinky while pinching (`PINCH_TUCK_CURL = 0.9`).
- **Pinch** (`robot/pinch.py`): when a commanded gap exists, `PinchSolver` (Dex3) or `SharpaPinchSolver` solves
  thumb/index joints with `scipy.optimize.least_squares` [[Virtanen et al., 2020](REFERENCES.md#virtanen2020scipy)] (bounded) so the tip gap matches, with a lean term
  that keeps distal links upright and a regulariser towards the current pose.

### 3.5 Trajectory-level corrections (`stages/retargeting/clip.py`)

In order: lift landmarks above table/obstacle supports (`support_clearance.py`), level the thumb–index axis
over `LEVEL_RAMP_FRAMES` with a rate-limited turn share, push the idle wrist away from the torso and active arm
(`_avoid_arm_collisions`), then up to `MAX_CLEARANCE_ITERS` rounds of measuring robot–table penetration and
lifting or rolling the pinch. Finally resample to 50 Hz.

### 3.6 Grip window, gap schedule and hand-eye correction

- **Grip timing** (`grip_timing.py`, `tactile_grasp.py`): the close is moved to the onset of the EgoTac contact
  span that contains the pick rise (≥ 5 cm); without tactile it uses the last frame the human pinch was within
  3 cm of the object.
- **Gap schedule** (`grip.py`): closed gap = object width − squeeze, with squeeze sized from mass, friction and
  measured pinch stiffness (fingers are PD position servos, so force = stiffness × squeeze).
- **Hand-eye** (`hand_eye.py`): the robot hand is not human-sized, so the robot pinch point (forward
  kinematics) is iterated onto the object, then handed over to the receptacle frame before the release
  (`align_pinch`, ≤ 4 iterations, 5 mm tolerance, offset ≤ 0.3 m). Also: hover and straight descent
  (`settle_approach`), carry acceleration limit (`limit_carry_acceleration`), release dwell/retreat
  (`release_motion`). If > 5 % of frames end with IK error > 3 cm, corrections are backed off in a fixed
  order (place first, then grasp).

### 3.7 Place-aware retry

When the place correction had to be reduced (receptacle out of reach from the chosen spot), the script
re-executes itself through `E2H_PLACE_GOAL` so the standing spot is scored on the corrected path (up to three
passes, cache in `<clip>/.placement_first`). Details: `TECH_DESIGN.md` §4.7.

### 3.8 Rollout and start-offset search

The final `q_target` is replayed in MuJoCo with PD servos. Perceived depth is good to a few cm, so the object
start xy is searched on a 5×5 grid within ±3 cm (25 rollouts), scored by `(success, lift_height)` from
`stages/manipulation/events.py::detect_pick_place`. The best start wins; success and place error go into
`retarget_rgb*.json`.

## 4. Embodiment gap: where each difference is handled

| Difference human → robot | Handled by |
|---|---|
| Finger count (5 → 3 on Dex3) | `finger_map.py` (ring/pinky onto middle) |
| Hand size, pinch point offset | `hand_eye.py` closed-loop offset |
| Arm reach / joint limits | Standing-spot grid search, IK step/limit clamps, unreachable-fraction back-off |
| Arm–torso / arm–arm collision | `_avoid_arm_collisions`, `robot/collision.py` |
| Hand geometry vs. table | Clearance loop, pinch roll, synthetic table fit (`align/table_fit.py`) |
| Position-controlled fingers, grip force | `grip.py` squeeze sizing |
| Human jerk, release height | `limit_carry_acceleration`, `release_motion` |
| Monocular depth scale | Table/workspace anchoring, hand-eye correction, start-offset search |
| Camera field of view | Waist pitch fit with the real D435 intrinsics |

Known limitation: many rules were added for specific clips (see clip numbers in `TECH_DESIGN.md`). They are
tuned by `SimParams` leaves (`contracts/sim_params.py`, `configs/sim_params/default.yaml`), not by learning. Contact-aware optimisation (SPIDER) is the planned,
optional refinement (`TASKS.md`).

## 5. Reading and debugging tips

1. Start from `scripts/retarget_estimated.py` and follow the log lines: `[time] <stage>`, `placement`,
   `path correction: grasp X, place Y`, `table fit`, then the rollout summary.
2. Check `retarget_rgb*.json` first: IK errors, unreachable fraction, grasp timing and `sim_events` failure
   reasons usually point at the responsible step in §3.
3. Wrong hand roll (palm up/down) → `RIGHT_Y_FLIP` in `kinematic.py`.
4. Hand misses the object by a few cm → hand-eye offsets (`path correction` log line), then the depth anchor
   (§3.1).
5. Object slips or is thrown → `grip.py` squeeze, carry acceleration, friction/mass in the physics params.
6. `script_params.py` reads `sys.argv[1]` and the environment at import time; do not import it from tests or
   other scripts.
7. Tests: `tests/unit/` has retargeting cases, to run inside the repo uv venv with the extras installed (`pytest tests/unit/test_hand_eye.py tests/unit/test_grip.py tests/unit/test_contact.py`).

## 6. Where to change what

| Goal | File |
|---|---|
| New robot hand | `robot/layout.py`, `robot/finger_map.py`, `configs/finger_mapping.yaml`, a solver in `robot/pinch.py` |
| Different IK behaviour | `robot/kinematics.py` (`ArmIK`) |
| Different grasp timing | `stages/retargeting/grip_timing.py`, `tactile_grasp.py` |
| New correction / primitive | `stages/retargeting/hand_eye.py`; keep new logic in separate files (large modules) |
| Tunable constants | `SimParams` via `E2H_SIM_PARAMS`, `script_params.py` |
| Export to SPIDER | `stages/retargeting/spider_convert.py`, `robot/floating_hand.py` |

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](REFERENCES.md).

- <span id="cite-buss2004ik"></span>[Buss, 2004](REFERENCES.md#buss2004ik). *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares Methods.*
- <span id="cite-cao2026fah"></span>[Cao, 2026](REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-nakamura1986dls"></span>[Nakamura and Hanafusa, 1986](REFERENCES.md#nakamura1986dls). *Inverse Kinematic Solutions with Singularity Robustness for Robot Manipulator Control.*
- <span id="cite-pan2025spider"></span>[Pan et al., 2025](REFERENCES.md#pan2025spider). *SPIDER: Scalable Physics-Informed Dexterous Retargeting.*
- <span id="cite-potamias2025wilor"></span>[Potamias et al., 2025](REFERENCES.md#potamias2025wilor). *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.*
- <span id="cite-romero2017mano"></span>[Romero et al., 2017](REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
- <span id="cite-simon2017hand"></span>[Simon et al., 2017](REFERENCES.md#simon2017hand). *Hand Keypoint Detection in Single Images using Multiview Bootstrapping.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-virtanen2020scipy"></span>[Virtanen et al., 2020](REFERENCES.md#virtanen2020scipy). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.*
- <span id="cite-wampler1986dls"></span>[Wampler, 1986](REFERENCES.md#wampler1986dls). *Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-yang2025omniretarget"></span>[Yang et al., 2025](REFERENCES.md#yang2025omniretarget). *OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
