# Hand selection: Unitree Dex3 vs Sharpa Wave (H.1, H.6)

Scope: which hand the G1 carries in `SceneConfig(hand=...)` / `HAND=<dex3|sharpa>`, and whether the
five-finger hand is stable and physically plausible enough to drive the tactile track (P/J).

## The two candidates

| | Unitree Dex3-1 | Sharpa Wave |
| --- | --- | --- |
| Source | shipped with the G1 MJCF (`assets/robots/g1/...`) | `assets/robots/sharpa_wave/`, mujoco_menagerie @ `370c4396`, Apache-2.0 |
| Fingers | 3 (thumb, index, middle) | 5 |
| Actuators per hand | 7 | 22 |
| Mass incl. wrist | 0.78 kg | 1.77 kg |
| Tactile surfaces | none modelled (we read phalanx contacts) | 5 fingertip sites + elastomer pads |
| EgoTac regions covered | 6 of 11 (no ring/pinky) | 11 of 11 |

The whole sim/retarget path is hand-agnostic: `robot/layout.py` derives the actuator indices from the
compiled model (43 actuators with Dex3, 73 with Sharpa), `map_curls_to_sharpa` + `SharpaPinchSolver`
drive the five fingers from the same FAH curls, and artefacts are suffixed per hand so both baselines
live side by side.

## Grasp quality on clip 104 (`basic_pick_place_104`)

Same perceived object, same estimated hand motion, `RENDER=1 SCENE_DIR=data/scenes/egodex_104` (measured at the time; the retarget has since changed, see `batch_results.md` for caveats):

| | Dex3 | Sharpa |
| --- | --- | --- |
| Grasp success | yes | yes |
| Lift height | 0.141 m | 0.202 m |
| Carry distance | 0.317 m | 0.337 m |
| Object-contact steps | 806 | 566 |
| Executed arm-contact steps | 23 | 634 |
| Peak normal force, active hand | thumb_tip 11.9 N, index_tip 10.8 N | thumb_tip 2.6 N, index_tip 1.5 N, middle_tip 3.1 N, middle_pad 1.7 N |
| Mean tangential/normal ratio | 0.28 | 0.56 |

All numbers are measured on the trajectory that is actually executed (after the pinch-gap and z-offset
refinement), not on the first-pass retarget.

Dex3 holds the block in a two-point pinch that has to squeeze hard and stays in contact far longer
because it drags the object rather than lifting it cleanly; the Sharpa hand wraps it in a four-region
envelope at a fifth of the force, and the higher tangential/normal ratio shows the load is carried by
friction over a larger patch rather than by normal pre-load. That is the regime the tactile track needs,
and it is the reason Sharpa is the default for the P/J work.

The one clear cost is self-contact: the bulky hands touch each other or the torso for 634 of the
recorded steps against Dex3's 23, which is tracked as H.8.

## Stability

* 50 Hz control over a 2 ms `implicitfast` timestep is stable with 22 DoF per hand; the Sharpa MJCF's own
  per-joint armature/damping/frictionloss classes are kept (`build_scene_spec` skips those joints), only
  the arm/wrist PD is ours.
* The heavier hand needed the wrist PD retuned: `HEAVY_HAND_WRIST_SCALE["sharpa"] = 2.5` scales the wrist
  kp/kd and torque limit with the 2.3x mass ratio. Without it the left wrist yaw lagged its command by
  0.58 rad on average and a finger joint showed a 9.1 rad tracking spike (contact-driven blow-up).
* The chirality of the human-to-robot wrist frame is hand-dependent (`RIGHT_Y_FLIP` in
  `stages/retargeting/kinematic.py`). Flipping the wrong one rolls that hand 180 deg; on Dex3 it raised the
  executed arm-contact steps from 23 to 224. Superseded: the Sharpa right hand was originally mounted
  palm-outward and needed a flip; the mount was fixed (`robot/sharpa.py::ATTACH_R`) and no hand needs a flip now.
* Residual limitation (H.7/H.8): with both hands close together the bulky hands press against the torso
  and against each other, so the wrist saturates its 50 Nm limit and creeps instead of tracking. Planning
  with an 8 mm hand-hand clearance (`HAND_CLEARANCE_M` in `stages/retargeting/clip.py`) did not reduce those
  contacts, but kept the idle arm out of the carry (0.235 m -> 0.337 m); a larger margin displaced it enough
  to break the grasp. Superseded: the Sharpa clearance is now 0.03 m (clip 100 idle hand 12 mm from the active
  one); Dex3 stays 0.0.

## Decision

Sharpa Wave is the default five-finger hand. Dex3 stays supported as the lighter baseline (its
trajectories are bit-identical to the pre-Sharpa pipeline: clearance 0, wrist scale 1).
