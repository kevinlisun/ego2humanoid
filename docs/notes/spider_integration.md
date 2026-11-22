# SPIDER integration notes (T3.3)

Source: `third_party/spider` (Meta FAIR, CC BY-NC 4.0, Python 3.12, mujoco 3.7.0, mujoco-warp 3.7.0.1,
warp-lang 1.12.1). Devbox GPU: A100 80 GB. Static read only; the upstream MJWP demo has not been run yet.

## Pipeline

1. `spider/process_datasets/<dataset>.py` -> `processed/<dataset>/<robot>/<embodiment>/<task>/<id>/`
   (`spider.io.get_processed_data_dir`), with `trajectory_*.npz` (`qpos`, `qvel`, `ctrl`, `contact`) and a
   scene XML. `gmr.py` and `hot3d.py` are the closest references (humanoid and egocentric hands).
2. `spider/preprocess/`: `decompose*.py` (object collision meshes), `detect_contact.py`, `generate_xml.py`
   (robot + object scene), `ik.py` (kinematic initial guess).
3. `examples/run_mjwp.py +override=<name> task=<t> data_id=<i> robot_type=<r> embodiment_type=<e>`:
   sampling MPC (cross-entropy, `spider/optimizers/sampling.py`) on batched MuJoCo-Warp.
4. `spider/postprocess/`: success metrics and deployment prep.

Config: one dataclass `spider/config.py` (`Config`), Hydra YAML in `examples/config/` and
`examples/config/override/`. The closest override for us is `humanoid_object.yaml`
(`robot_type: unitree_g1`, `embodiment_type: humanoid_object`, `horizon 0.8`, `ctrl_dt 0.1`).

## Robot assets

`spider/assets/robots/unitree_g1/` has 29-DoF G1 MJCFs, including `g1_mocap_29dof_with_hands.xml` (7-DoF
hands per side, Dex3 naming `left_hand_thumb_0_link`, ...). It uses the same joint naming as our
`src/robot/` model; whether the joint order and limits match must be verified in T3.5 by FK/hash parity.

## Mapping to our bundles (plan)

| Ours | SPIDER |
| --- | --- |
| `hands/` bundle (camera-frame MANO landmarks) + `alignment/` (`T_sim_camera`) | hand/wrist reference trajectory in the sim frame (like `hot3d.py`) |
| `objects/` bundle (mesh, pose) | object mesh + `decompose` + scene XML via `generate_xml.py` |
| our kinematic `q_target_rad` (43-dim actuator order) | initial guess / `ik.py` replacement |
| `retarget/` bundle `q_target_rad`, `ctrl` | `ctrl` from `run_mjwp.py` output |

## Constraints and open questions

- Fixed pelvis, one or two arms: SPIDER's humanoid mode tracks base and joints; we must freeze base and legs
  (`base_pos_rew_scale`, `joint_rew_scale`) or use a reduced MJCF.
- Datasets are loaded from paths on disk, so T3.6 writes our bundles into that layout (`src/stages/retargeting/spider_convert.py`); no code from our
  stack should import SPIDER (it needs Python 3.12 + mujoco-warp: separate image, T3.4; `nodes/retarget` is currently the MuJoCo retarget/render image, not SPIDER).
- License: CC BY-NC 4.0, research only (Q6).
- Not verified yet: running `examples/run_mjwp.py` on a bundled example, GPU memory per `num_samples`,
  whether the G1 hand collision geoms support pinch grasps with the sampling noise settings.
- Our kinematic retargeter (`stages/retargeting/`) stays as the initial guess and baseline; SPIDER refinement is
  expected to help the near-miss clips (100, 116) with contact-aware optimisation.

## Demo run result (T3.3)

- `examples/run_mjwp.py` (gigahand / xhand / bimanual / p36-tea) runs end to end
  with `MUJOCO_GL=egl` (the default GL fails with `gladLoadGL error`).
- 60 sim steps, 8 optimisation steps: plan time 155 s (first run includes
  kernel compilation); final object tracking error pos 2.6 cm.
- Finding: `spider/assets/robots` has no Dex3 hand for MJWP. Dex3 appears only
  in `arctic/dex3_hand` (dexmachina npz). T3.5 must author a G1+Dex3 MJCF from
  our own robot model.

## T3.5 design decision: floating-wrist Dex3 hands

MJWP has no Dex3 asset and our arm IK already solves the arm. So `src/robot/floating_hand.py` builds a
Dex3 hand from the G1 MJCF hand subtree on a 6-DoF wrist (3 slides + 3 hinges, like SPIDER's `xhand`
models), one XML per side, with the same finger limits and PD gains as the replay. MJWP tracks the
wrist pose from our arm-IK result and optimises fingers/contacts; the refined finger trajectory
returns to the full G1 via the existing arm IK (T3.7). SPIDER reads `scene.xml` from the processed
data directory, so the converter (T3.6) composes hands + object + table there and the SPIDER
checkout stays unmodified.

## T3.7 findings (static read of `preprocess/generate_xml.py` and `ik.py`)

- Robot assets are looked up in the dataset dir, not the SPIDER package:
  `<dataset_dir>/processed/<dataset>/assets/robots/<robot_type>/{bimanual,left,right}.xml`. A `dex3`
  robot type needs no change to the SPIDER checkout.
- `generate_xml` and `ik` expect per-side sites `<side>_{thumb,index,middle,ring,pinky}_tip`, their
  `track_hand_*` / `trace_hand_*` copies and `<side>_palm`. Dex3 has no ring/pinky finger, so the
  floating-hand XML must add these sites (thumb/index/middle from `src/robot/pinch.py` tips) and the
  ring/pinky entries need handling (patch `get_robot_sites`/IK weights, or skip SPIDER IK).
- Preferred path: skip SPIDER's keypoint IK and write `trajectory_kinematic.npz` (`qpos`, `qvel`,
  `frequency`) ourselves from our retargeter (wrist 6 DoF + 7 finger joints per side, then the two object
  free joints), keeping `generate_xml` only for `scene.xml`.
- The dataset convention puts the table/floor at z = 0 (object lowest point at z = 0 at frame 0): the
  converter must subtract the table height (0.80 m) and the runner add it back.
- Objects need `visual.obj` and a `convex/` CoACD directory under the task assets.
- Open question for the first experiment: which sites MJWP's reward uses for fingertip tracking.
