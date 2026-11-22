"""Support/obstacle geometry and place-target resolution used by ``scripts/retarget_estimated.py``."""

import os
import pickle
from pathlib import Path

import numpy as np

from stages.locomotion.stance_adjust import adjust_stance, critical_window
from stages.manipulation.mjcf_scene import BoxObject
from stages.reconstruction.objects.grasp_proxy import detect_grasp_proxy
from stages.reconstruction.objects.place_target import resolve_from_detections, resolve_from_scene
from stages.reconstruction.objects.prompt import place_target_prompt
from stages.retargeting.clip import retarget_clip
from stages.retargeting.kinematic import landmarks_camera_to_sim
from stages.retargeting.support_clearance import Support


def object_supports(obj: BoxObject, margin_m: float) -> list[Support]:
    """``obj`` as tops to keep the hands above: one per collision piece, else its whole extent.

    A single top over the whole body kept the hand 14 cm above clip 101's open case everywhere, as
    high as its lid, so the tape could never be lowered into its 4 cm base.
    """
    pos = np.asarray(obj.pos_m, float)
    if obj.collision_pieces and obj.mesh_vertices is not None:
        return [Support.from_points(np.asarray(v, float) + pos, margin_m) for v, _ in obj.collision_pieces]
    return [Support.from_points(object_points(obj), margin_m)]


def object_points(obj: BoxObject) -> np.ndarray:
    """World points spanning ``obj``: its mesh, or its box corners."""
    pos = np.asarray(obj.pos_m, float)
    if obj.mesh_vertices is not None:
        return np.asarray(obj.mesh_vertices, float) + pos
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return corners * np.asarray(obj.half_size_m, float) + pos


def resolve_place_target(
    task_description, held_half_z_m, release_pinch_m, cand_sim, extent_m, usable, exclude_xy_m
):
    """The object this task places things onto, or None for a bare table-to-table transfer.

    The reconstructed scene is preferred: ``objects.json`` already carries named, metric, sim-frame
    meshes, so no extra perception is needed and the drop point comes straight off the geometry.
    Clips without a reconstruction fall back to the stationary SAM 3 candidate near the release.
    """

    phrase = place_target_prompt(task_description)
    if phrase is None:
        return None
    scene_dir = os.environ.get("SCENE_DIR")
    if scene_dir:
        from stages.reconstruction.scene.export_mjcf import load_scene_objects

        found = resolve_from_scene(load_scene_objects(scene_dir), phrase, held_half_z_m)
        if found is not None:
            return found
    return resolve_from_detections(cand_sim, extent_m, usable, release_pinch_m, held_half_z_m, exclude_xy_m)


def placement_retarget(clip, hand, lm, t, up, side, passive, pinch_uv_ref, adjust: bool, place_goal=None):
    """Retarget with the L.17 stance adjustment; returns ``(retarget, stance_adjustment or None)``.

    The placement search dominates the run time and depends only on the clip and the hand, so parameter
    sweeps can reuse it: E2H_RETARGET_CACHE=<dir> keeps one pickle per clip and hand.
    """
    cache_dir = None if place_goal else os.environ.get("E2H_RETARGET_CACHE")
    stance_adj = None
    cache_file = Path(cache_dir) / f"{clip.name}_{hand}.pkl" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        with open(cache_file, "rb") as f_:
            r = pickle.load(f_)
        if isinstance(r, tuple):  # (retarget, stance adjustment); older caches hold the retarget alone
            r, stance_adj = r
        print(f"placement retarget loaded from {cache_file}")
        return r, stance_adj
    r = retarget_clip(
        lm, t, up, side, passive_landmarks_camera=passive, hand=hand, pinch_uv_ref=pinch_uv_ref,
        place_goal=place_goal,
    )  # fmt: skip
    # L.17: the original stance stays unless the critical frames leave the head-camera view.
    proxy0 = detect_grasp_proxy(landmarks_camera_to_sim(lm, r.placement.T_sim_camera), t)
    if proxy0 is not None and adjust:
        adj = adjust_stance(
            lm,
            side,
            r.placement,
            critical_window(t, t[proxy0.grasp_idx], t[proxy0.release_idx]),
            pinch_uv_ref,
            hand,
        )
        stance_adj = adj
        if adj.level > 0 or adj.placement.waist_pitch_rad != r.placement.waist_pitch_rad:
            r = retarget_clip(
                lm, t, up, side, placement=adj.placement, passive_landmarks_camera=passive, hand=hand,
                pinch_uv_ref=pinch_uv_ref,
            )  # fmt: skip
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f_:
            pickle.dump((r, stance_adj), f_)
    return r, stance_adj
