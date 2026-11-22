# Clip artefacts and their physical meaning

Everything a clip run produces lives under `data/derived/egodex/<clip>/`, except the
reconstructed-scene renders, which belong to the scene and live under `data/scenes/<scene>/`.
Files marked *keep* are the ones worth inspecting or re-using; anything not listed is scratch and
may be deleted and regenerated.

Render backend: `scripts/review_render.py` (called by `scripts/retarget_estimated.py` when `RENDER=1`)
defaults to `RENDER_BACKEND=blender`. Then only the left (RGB + hands) and trajectory panels are drawn
in Python; the robot and ego panels are Blender renders of the exported twin scene
(`scripts/export_twin_scene.py`, `scripts/twin_render.py`, needing `BPY_PYTHON`), and the grid is composed
from `review/third_blender.mp4` and `review/ego_blender.mp4`. The MuJoCo-panel files described below
(`side_by_side`, `sim_mujoco`, `ego_plate_composite`, `panels/tr`, `ego_d435`, `bg_rgbd`, the scene-side
`robot_in_scene` video and the 3DGS ego panel) are only produced with `RENDER_BACKEND=mujoco`.
`E2H_OUT_TAG` appends an experiment tag to every output name (`*_bt_<stamp>`); those are scratch.

Suffix convention: files for the Sharpa Wave five-finger hand carry `_sharpa`; no suffix means the
Unitree Dex3 three-finger hand.

## Review videos (human-viewable) — `data/derived/egodex/<clip>/review/`

| File | Physical meaning |
| --- | --- |
| `result_grid{,_sharpa}.mp4` | **The main deliverable.** (Description below is the MuJoCo backend; with the default Blender backend panels 2 and 3 are the Blender third-person and head-camera renders.) 2x2 grid on the simulation clock (50 Hz): top-left the source ego RGB with the estimated hand skeletons and per-region contact force; top-right the retargeted robot executing the same motion in MuJoCo (free 3/4 camera, full scene); bottom-left what the robot itself sees, rendered from the **Unitree G1 head camera** (`d435`, see below): the simulated robot and the reconstructed object meshes come from MuJoCo, the room behind them from the **background Gaussian splat**, and the two are merged per pixel by depth (see below); its own torso and legs are drawn semi-transparent, because the camera sits 0.15 m from the chest and would otherwise cover half the frame and hide the hands; bottom-right the task-space trajectories (see below). |
| `side_by_side{,_sharpa}.mp4` | Two-panel version: perception on the left, MuJoCo replay on the right with the pick-and-place SUCCESS/FAIL verdict. |
| `sim_mujoco{,_sharpa}.mp4` | Raw MuJoCo replay only, fixed 3/4 camera, perceived object only (no reconstructed scene). The closest view of the grasp itself. |
| `ego_plate_composite{,_sharpa}.mp4` | Reference only: the robot composited over the reconstructed clean plate from the *human* camera recovered from the source video (the old ego panel). Useful to check the sim-to-video alignment; it is **not** the robot's viewpoint. |
| `third_blender.mp4`, `ego_blender.mp4` | Default backend: Blender renders of the robot (third-person and G1 head camera) used as grid panels 2 and 3. |
| `sim_state{,_sharpa}.npz`, `sim_model/`, `twin_scene/`, `twin_appearance/` | Recorded body states, the exported simulation model, and the Blender twin scene/appearance data that the Blender renders read. |
| `trajectory{,_sharpa}.npz` | The numbers behind the bottom-right panel; see the table below. |
| `panels/{tl,tr,br}{,_sharpa}.mp4` | The cheap grid panels (`tr` only with the MuJoCo backend), cached so the grid can be rebuilt without re-running the 7-minute retarget. |
| `panels/ego_d435{,_sharpa}.npz` | The MuJoCo half of the ego panel: per frame the RGB and metric depth of the simulated content (robot + reconstructed objects, sim floor and table dropped), the same for the robot's own body alone, and the world pose of the `d435` camera. |
| `panels/bg_rgbd{,_sharpa}.npz` | The splat half: the background rasterised by gsplat at those same poses (RGB, expected depth, alpha). |

Scene-side video, `data/scenes/<scene>/review/robot_in_scene{,_sharpa}.mp4`: source RGB next to the
robot acting inside the reconstructed scene (free camera). The other `stage*.mp4` files there are
the SimFoundry reconstruction stages (background removal, upsampling, mesh extraction, sim check).

### Bottom-left panel: robot-in-the-loop neural rendering

The ego panel is a depth composite of two renderers, not a 2D overlay:

* **Foreground** — MuJoCo renders the robot and the reconstructed object meshes from the `d435`
  camera, with a real depth buffer; the sim floor and table are dropped because the reconstruction
  supplies them.
* **Background** — `data/scenes/<scene>/bg_splat.npz`, about one million flat 3D Gaussians placed in
  the sim frame, rasterised by gsplat at exactly the same camera pose.
* **Merge** — per pixel, whichever of the two is nearer wins. The robot therefore occludes the table
  where it is in front of it and is occluded where it is behind, with no manual layering. Pixels
  that neither renderer covers are inpainted from the background edge.

The splat is **built**, not trained. Clip 104's camera translates 2.2 cm and rotates 2.9 deg over
the whole take (EgoDex ground truth; DA3 recovers the same), so there is no baseline for multi-view
optimisation — a splat trained on these views could only reproduce what one view plus monocular
depth already says. `stages/scene/bg_splat.py` therefore turns the stage-2 DA3 reconstruction of the
anchor view into Gaussians directly: colour from the FLUX clean plate (the only human- and
object-free image of the room), recoloured surface by surface to match the real frame because FLUX
repaints it; depth from DA3, with everything the plate removed diffused in harmonically from its
surroundings; one flat disc Gaussian per pixel, normal along the local surface normal and radius
equal to the pixel footprint. Being single-view-equivalent, it opens disocclusion holes at object
silhouettes when seen from far off the anchor; the head camera is 0.19 m away, which keeps them small.

Rebuild it with:

```bash
uv run --frozen --extra video --extra hands --extra objects \
  python scripts/build_bg_splat.py data/scenes/egodex_104 data/derived/egodex/basic_pick_place_104
```

Rasterising it needs CUDA and torch, which only the `simfoundry` conda environment has, so that step
is a separate process and the grid is then recomposed:

```bash
P=/cloudfile/nvme_tmp/egodex/basic_pick_place_104/panels
CUDA_HOME=/mnt/nvme_hot/envs/cuda_home PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0 \
  <conda>/envs/simfoundry/bin/python scripts/render_bg_splat.py \
  $P/ego_d435.npz data/scenes/egodex_104/bg_splat.npz $P/bg_rgbd.npz
uv run --frozen --extra video --extra hands --extra objects \
  python scripts/make_review_grid.py data/derived/egodex/basic_pick_place_104
```

(Large panel captures live under `/cloudfile/nvme_tmp/egodex/<clip>/panels` (`E2H_BULK_DIR` overrides);
each `RENDER=1` run first deletes that hand's old review files and panels, so `bg_rgbd` must be re-rasterised
after every re-render.)

(append `_sharpa` as the second argument of `make_review_grid.py` for the five-finger hand.)

### Bottom-right panel: task-space trajectories

An isometric view of the simulation world, metric in all three axes, so it shows the depth that the
two camera panels cannot. Ground grid = the table top plane (z = 0.80 m), 10 cm spacing.

* fat polyline + dot per side: the **wrist_yaw body origin**, i.e. the end-effector pose the
  retargeting is tracking; thin lines are the five **fingertips** (distal phalanges);
* green: the **manipulated object's centre of mass**, drawn with its full history, plus its three
  body axes (6 cm) so rotation during the carry is visible; the caption shows the lift height
  relative to the first frame;
* violet triangle: the **human egocentric camera position** (`T_sim_camera`), the viewpoint of the
  top-left panel and of `ego_plate_composite*.mp4`.

### The robot head camera (`d435`)

The bottom-left panel is rendered from a camera compiled into the scene straight from the official
Unitree description (`unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`):

```xml
<joint name="d435_joint" type="fixed">
  <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
  <parent link="torso_link"/><child link="d435_link"/>
```

so it is an Intel RealSense D435 rigidly mounted on the torso, 0.43 m above the torso origin and
pitched 47.6 deg down; with the pelvis at z = 0.793 m it sits at (0.054, 0.018, 1.267) m in the sim
world and **moves with the robot**. The vertical field of view is the D435 colour sensor at 16:9,
42.5 deg (69.4 deg horizontal). Because this pose differs from the human camera recovered from the
video (about 0.19 m further forward and 13 deg steeper), the real clean plate cannot be used as its
background: the panel shows the sim's own floor, table and the reconstructed object meshes.

`trajectory*.npz` keys: `wrist_m` (N, 2, 3) left/right wrist origin in sim world metres;
`fingertip_m` (N, 2, 5, 3) fingertips in the same frame (NaN where the hand lacks that finger, e.g.
Dex3 ring/pinky); `object_pos_m` (N, 3); `object_rot` (N, 3, 3) world-from-object rotation;
`camera_T_sim` (4, 4) ego camera pose. Sampled at 50 Hz, aligned with the review videos.

## Machine artefacts (consumed by later stages) — `data/derived/egodex/<clip>/`

| File | Physical meaning |
| --- | --- |
| `episode.json` | Clip metadata exported from EgoDex (task description etc.). |
| `video.mp4` | The source egocentric RGB clip (30 Hz). |
| `hands_fah_f<focal>.npz` | FAH hand estimate: `joints2d` px, `joints3d` m in the camera frame, `valid`, `score`; the focal length in the name is the estimated intrinsics used. |
| `intrinsics_est.json`, `gravity_est.json` | Estimated focal length, and the gravity (up) direction in the camera frame used to put the clip upright in the sim world. |
| `objects_sam3.npz` | Per-frame object masks/boxes from SAM3 segmentation. |
| `object_crop.png`, `object_mesh_raw.glb`, `object_preview.png` | The crop fed to the mesh reconstructor, the raw reconstructed mesh, and a render for review. |
| `tactile_egotac.npz` | EgoTac per-vertex contact estimate on the human hand (the perception-side tactile signal). |
| `sim_tactile{,_sharpa}.npz` | Simulated tactile: per-region normal/tangential force and binary contact read out of MuJoCo, same EgoTac region schema, so sim and human tactile are comparable by name. |
| `retarget_rgb{,_sharpa}.npz` | `timestamps_s` and `q_target_rad`: the retargeted joint trajectory actually executed. |
| `retarget_rgb{,_sharpa}.json` | The run report: grasp success, lift height, carry distance, object-contact steps, executed arm self-contact steps, tracking error, and the perceived-object parameters. |
| `alignment/` | `alignment.json` (sim-from-camera transform, table height, depth scale), `contact_flags.npz` (which human hand vertices touch the object), `tactile_raw.npz` (a copy of the EgoTac estimate used for that alignment). |
| `scene/` | Tabletop inventory from `scripts/perceive_scene_objects.py` (`inventory.json`, `crop_*.png`, `mesh_*.glb`, `points.npz`). Its objects are simulated by the retarget even when `SCENE_DIR` is set. |
| `scene_sim.pkl` | Perceived scenery saved by the retarget; read by `scripts/export_twin_scene.py` and the loco-manip stage. |
| `gt/` | EgoDex ground truth (`egodex_gt.npz`), used only for evaluation (`hand_eval*.json`), never as a pipeline input. |

## Regenerating

```bash
MUJOCO_GL=egl RENDER=1 SCENE_DIR=data/scenes/egodex_104 HAND=sharpa \
  uv run --frozen --extra video --extra sim --extra hands --extra objects --extra egodex \
  python scripts/retarget_estimated.py data/derived/egodex/basic_pick_place_104
```

`HAND` selects the robot hand (`dex3` or `sharpa`; the script defaults to `dex3`, `scripts/run_pipeline.sh` to `sharpa`) and `RENDER=1` enables the review videos. All
four panels are produced from the rollout alone, so `SCENE_DIR` is optional: it adds the
reconstructed objects to the scene, the `robot_in_scene` side-by-side video, and — if the scene also
carries a `bg_splat.npz` — the 3DGS background of the bottom-left ego panel. Without it the ego
panel shows the robot acting on the bare simulated table, and the panel label says so. None of
these outputs are committed; `data/` is ignored.

The whole chain (perception, retarget, render) is `scripts/run_pipeline.sh <clip_id>...`, optionally
`DOCKER=1` to run each stage in its per-stage image (`nodes/<node>`); see `run_pipeline.md`.

Across clips, `scripts/summarize_batch.py data/derived/egodex docs/notes/batch_results.md` collects
every clip's simulated grasp metrics (success, lift, carry, contact steps, self-collisions, IK
error) into one table next to the path of each review video.
