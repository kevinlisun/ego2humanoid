"""Review-video rendering of the retarget script (RENDER=1): sim, scene, ego, trajectory panels, grid."""

import os
from pathlib import Path

import cv2
import numpy as np

from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.manipulation.rollout import run_rollout
from stages.manipulation.sim_tactile import SimTactile, support_touch

# Blender renders every image of the deliverable; MuJoCo stays headless physics (track_bt link doc §7)
BLENDER = os.environ.get("RENDER_BACKEND", "blender") == "blender"


def render_review(
    clip,
    SUFFIX,
    HAND,
    base,
    pbox,
    extra,
    table_kw,
    r,
    t,
    est,
    sl,
    side,
    MIN_SCORE,
    pev,
    f,
    SIM_PARAMS,
    D435_IMAGE_HW,
    EGO_PANEL,
    _lap,
):
    """Render the review videos; returns ``(touch, blender_grid)``.

    ``touch`` is the active-hand table-touch summary. With the Blender backend the MuJoCo panels are
    skipped and ``blender_grid`` holds the (left, trajectory) frames the caller composes with the
    Blender views once the report exists; otherwise it is None.
    """
    from render_side_by_side import compose, overlay_frames

    from stages.manipulation.render import VideoRecorder
    from stages.qa.grid_video import compose_grid, write_video
    from stages.qa.review_paths import clean_review
    from stages.qa.review_paths import panels_dir as bulk_panels_dir
    from stages.qa.traj_panel import IsoProjection, draw_trajectory_panel

    review = clip / "review"
    review.mkdir(exist_ok=True)
    scene_review = Path(os.environ["SCENE_DIR"]) / "review" if os.environ.get("SCENE_DIR") else None
    # Stale outputs of earlier renders (this hand only) must not survive into a new review.
    clean_review(
        clip,
        SUFFIX,
        (scene_review / f"robot_in_scene{SUFFIX}.mp4",) if scene_review is not None else (),
    )
    rec = None if BLENDER else VideoRecorder(review / f"sim_mujoco{SUFFIX}.mp4", fps=50, size=(480, 640))
    rres = run_rollout(
        # The receptacle belongs in the review video too: without it the top-right panel
        # shows the block being placed onto bare table, which is not what was simulated.
        SceneConfig.from_params(
            SIM_PARAMS, T_sim_robot_base=base, objects=(pbox, *extra), hand=HAND, **table_kw
        ),
        r.timestamps_s,
        r.q_target_rad,
        on_record=rec,
        tactile=lambda m: SimTactile(m, partner="object+table"),
    )
    if BLENDER:
        sim_path, n_sim = None, int(len(rres.arrays["timestamps_s"]))
    else:
        sim_frames = list(rec.frames)
        sim_path, n_sim = rec.close(), len(sim_frames)
    sim_tac = rres.tactile.save(clip / f"sim_tactile{SUFFIX}.npz")
    # The pads separate the hand pressing on the table from gripping the object.
    touch = support_touch(rres.tactile.arrays(), rres.arrays["timestamps_s"], side)
    print("active hand on the table (tactile):", touch)
    # Human (EgoTac) regions when available, otherwise show the simulated pad forces.
    tac = clip / "alignment" / "tactile_regions.npz"
    tac_label = "human tactile"
    if not tac.exists():
        tac, tac_label = sim_tac, f"sim pad force ({HAND})"
    overlay_args = (est["joints2d"][sl], est["valid"][sl], est["score"][sl], side, float(t[0]))
    overlay_kw = dict(
        min_score=MIN_SCORE,
        tactile_regions_npz=tac if tac.exists() else None,
        tactile_label=tac_label,
    )
    # Drawn once for the side-by-side and the grid (the two used to draw it independently).
    left_frames = overlay_frames(clip, n_sim, *overlay_args, **overlay_kw)
    if not BLENDER:
        compose(
            clip,
            sim_path,
            est["joints2d"][sl],
            est["valid"][sl],
            est["score"][sl],
            side,
            t[0],
            pev.success,
            suffix=SUFFIX,
            min_score=MIN_SCORE,
            tactile_regions_npz=tac if tac.exists() else None,
            tactile_label=tac_label,
            left_frames=left_frames,
        )
    _lap("render: sim video + side-by-side")
    from skills.body_state import BodyStateRecorder, export_sim_model
    from stages.qa.traj_panel import TrajectoryRecorder
    from stages.reconstruction.scene.composite import CompositeRecorder, write_side_by_side
    from stages.reconstruction.scene.export_mjcf import load_scene_objects
    from stages.reconstruction.scene.neural_ego import EgoNeuralRecorder, composite_neural_ego

    # The reconstructed SimFoundry scene is optional: without it the robot still acts, it
    # just acts in the bare sim room with the perceived object instead of the real one.
    scene_dir = os.environ.get("SCENE_DIR")
    scene = Path(scene_dir) if scene_dir else None
    scene_objects = load_scene_objects(scene) if scene is not None else ()
    if extra:
        # The simulated receptacle was re-anchored into the retargeting frame, so the raw
        # reconstructed copy would be a second bowl standing somewhere else.
        scene_objects = tuple(o for o in scene_objects if o.name != extra[0].name) + extra
    panels_dir = bulk_panels_dir(clip)  # large captures live on the bulk volume

    if BLENDER:
        crec = erec = prec = None
    else:
        import mujoco

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.25, -0.05, 0.9)
        cam.distance, cam.azimuth, cam.elevation = 1.5, 150.0, -25.0
        crec = CompositeRecorder(
            None,
            r.placement.T_sim_camera,
            float(f),
            tuple(est["image_wh"]),
            camera=cam,
            ghost_alpha=0.0,  # third-person view: the body must stay solid
        )
        # Ego view of the same rollout from the robot's own head camera (Unitree D435) in the
        # simulated scene (table and room); the waist tilt chosen at placement aims it.
        erec = EgoNeuralRecorder(camera="d435", size=D435_IMAGE_HW, drop_background=False)
        # Reference only: the human camera composited over the reconstructed clean plate.
        plate = scene / "plate.png" if scene is not None else None
        prec = (
            CompositeRecorder(
                cv2.cvtColor(cv2.imread(str(plate)), cv2.COLOR_BGR2RGB),
                r.placement.T_sim_camera,
                float(f),
                tuple(est["image_wh"]),
                camera="ego",
            )
            if plate is not None and plate.exists()
            else None
        )
    trec = TrajectoryRecorder(T_sim_camera=r.placement.T_sim_camera)

    brec = BodyStateRecorder()

    def _record(m, d, _c=crec, _e=erec, _p=prec, _t=trec, _b=brec):
        for rec_fn in (_c, _e, _p):
            if rec_fn is not None:
                rec_fn(m, d)
        _t(m, d)
        _b(m, d)

    rollout_cfg = SceneConfig.from_params(
        SIM_PARAMS,
        T_sim_robot_base=base,
        **table_kw,
        objects=(pbox, *scene_objects),
        hand=HAND,
        ego_camera_T_sim=r.placement.T_sim_camera,
        ego_camera_fovy_deg=float(np.degrees(2 * np.arctan(float(est["image_wh"][1]) / (2 * f)))),
    )
    run_rollout(rollout_cfg, r.timestamps_s, r.q_target_rad, on_record=_record)
    if crec is not None:
        crec.close()
    _lap("render: scene + head-camera rollout")
    brec.save(review / f"sim_state{SUFFIX}.npz")
    export_sim_model(build_scene(rollout_cfg), review / "sim_model")
    if scene is not None and crec is not None:
        write_side_by_side(
            clip / "video.mp4", crec.frames, scene / f"review/robot_in_scene{SUFFIX}.mp4", float(t[0])
        )
    traj = trec.arrays()
    np.savez_compressed(review / f"trajectory{SUFFIX}.npz", **traj)
    proj = IsoProjection(
        np.concatenate(
            [
                traj["wrist_m"].reshape(-1, 3),
                traj["object_pos_m"],
                traj["camera_T_sim"][:3, 3][None],
            ]
        ),
        (480, 640),
    )
    traj_frames = [draw_trajectory_panel(traj, i, (480, 640), proj=proj) for i in range(len(traj["wrist_m"]))]
    if crec is not None:
        scene_frames = crec.frames
        erec.close()
        ego_npz = erec.save(panels_dir / f"ego_d435{SUFFIX}.npz")
        ego_frames = composite_neural_ego(ego_npz, None)
        if prec is not None:
            prec.close()
            write_video(review / f"ego_plate_composite{SUFFIX}.mp4", prec.frames, fps=50)
    panels_dir.mkdir(parents=True, exist_ok=True)
    if BLENDER:
        # Blender renders both robot views once the report exists (see ``blender_grid``).
        write_video(panels_dir / f"tl{SUFFIX}.mp4", left_frames, fps=50)
        write_video(panels_dir / f"br{SUFFIX}.mp4", traj_frames, fps=50)
        _lap("render: panels")
        return touch, (left_frames, traj_frames)
    n_grid = min(len(sim_frames), len(scene_frames))
    grid_left = (
        left_frames
        if n_grid == len(left_frames)
        else overlay_frames(clip, n_grid, *overlay_args, **overlay_kw)
    )
    write_video(panels_dir / f"tl{SUFFIX}.mp4", grid_left, fps=50)
    write_video(panels_dir / f"tr{SUFFIX}.mp4", scene_frames, fps=50)
    write_video(panels_dir / f"br{SUFFIX}.mp4", traj_frames, fps=50)
    grid_path = compose_grid(
        [
            ("RGB + estimated hands (+tactile)", grid_left),
            (f"robot in MuJoCo ({HAND})", scene_frames),
            (EGO_PANEL + ", sim scene", ego_frames),
            ("task-space trajectories", traj_frames),
        ],
        review / f"result_grid{SUFFIX}.mp4",
    )
    print("grid video:", grid_path)
    _lap("render: panels + grid")
    return touch, None
