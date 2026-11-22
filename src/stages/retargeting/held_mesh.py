"""Held-object mesh/box asset for the physics scene (``scripts/retarget_estimated.py``)."""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from stages.reconstruction.objects.prompt import object_prompt
from stages.retargeting.kinematic import robot_pinch_axis_xy
from stages.retargeting.script_params import (
    ANTIPODAL_EXTRA_WIDTH_M,
    BOX_FACES,
    MAX_EXTRUDE_ASPECT,
    MAX_OPEN_EXTRA_M,
    MIN_OBJECT_THICKNESS_M,
    TP,
)


class HeldMeshMissing(RuntimeError):
    """No held-object mesh and the default-box fallback is not allowed (error ``HELD_MESH_MISSING``)."""

    code = "HELD_MESH_MISSING"


def scene_held_mesh(clip: Path) -> tuple[Path, Path | None] | None:
    """Scene-inventory mesh (and its crop) of the object the SAM3 track follows at the first frame.

    The first-frame centroid of the best SAM3 candidate is projected into the inventory frame and the
    smallest inventory box that contains it wins. Needs ``objects_sam3.npz``, ``intrinsics_est.json``
    and ``scene/inventory.json``; ``None`` when any is missing or no box contains the point.
    """
    scene = clip / "scene"
    files = (clip / "objects_sam3.npz", clip / "intrinsics_est.json", scene / "inventory.json")
    if not all(f.exists() for f in files):
        return None
    sam = np.load(files[0], allow_pickle=False)
    score, cen = np.asarray(sam["score"], float), np.asarray(sam["centroid_cam"], float)
    if score.size == 0 or not np.isfinite(score[0]).any():
        return None
    p = cen[0, int(np.nanargmax(score[0]))]
    cam = json.loads(files[1].read_text())
    if not np.isfinite(p).all() or p[2] <= 0:
        return None
    u = cam["width"] / 2 + cam["focal_px"] * p[0] / p[2]
    v = cam["height"] / 2 + cam["focal_px"] * p[1] / p[2]
    best, best_area = None, np.inf
    for o in json.loads(files[2].read_text())["objects"]:
        x0, y0, x1, y1 = o["box_xyxy"]
        area = (x1 - x0) * (y1 - y0)
        if o.get("mesh") and (scene / o["mesh"]).exists() and x0 <= u <= x1 and y0 <= v <= y1:
            if area < best_area:
                best, best_area = o, area
    if best is None:
        return None
    crop = scene / best["crop"] if best.get("crop") else None
    return scene / best["mesh"], crop if crop is not None and crop.exists() else None


def resolve_held_asset(clip: Path, allow_box: bool) -> tuple[Path | None, Path | None, dict]:
    """``(mesh, crop, provenance)``: the clip's own crop mesh, else the matching scene mesh, else the box.

    The box is used only when ``allow_box`` (``trajectory.allow_held_box_fallback``); otherwise raises
    ``HeldMeshMissing`` instead of faking a successful asset.
    """
    if (clip / "object_mesh_raw.glb").exists():
        return clip / "object_mesh_raw.glb", clip / "object_crop.png", {"source": "crop"}
    hit = scene_held_mesh(clip)
    if hit is not None:
        return hit[0], hit[1], {"source": "scene", "mesh": hit[0].name}
    if not allow_box:
        raise HeldMeshMissing(
            f"no held-object mesh for {clip.name}: object_mesh_raw.glb is missing and no scene mesh matches "
            "the SAM3 object. Run the held_mesh stage (STAGES=held_mesh needs HF_HOME for SAM3), or set "
            "trajectory.allow_held_box_fallback=true to simulate a default box"
        )
    return None, None, {"source": "box"}


def build_held_mesh(clip, held, lm_hold, side, hand):
    """Held-object collision/visual asset sized for the pinch.

    ``lm_hold`` are the sim-frame landmarks of the first hold frames (they set the pinch axis).
    Returns ``(half, mesh_kw, physics, thickness, open_width, pinch_axis, extruded, tan_alpha, tex,
    provenance)``; ``provenance["source"]`` is ``crop``, ``scene`` or ``box`` (see ``resolve_held_asset``).
    """
    half = held.half_size_m
    mesh_kw, physics, tex = {}, None, None
    mesh_path, crop_path, provenance = resolve_held_asset(clip, TP.allow_held_box_fallback)
    if mesh_path is not None:
        import trimesh

        from stages.reconstruction.objects.mesh_asset import (
            antipodal_axis_xy,
            build_mesh_asset,
            footprint_chord_width,
        )
        from stages.reconstruction.objects.physics_params import MIN_MASS_KG, heuristic_physics

        raw = trimesh.load(mesh_path, force="mesh")
        if (
            os.environ.get("E2H_HELD_MUG") == "1"
        ):  # the generated mesh is a slab; hold a mug-shaped body instead
            from stages.reconstruction.objects.mug_mesh import procedural_mug

            raw = trimesh.Trimesh(*procedural_mug())
        pinch_axis = robot_pinch_axis_xy(lm_hold, side, hand=hand)
        crop_png = None if crop_path is None else cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
        has_crop = crop_png is not None and crop_png.shape[-1] == 4
        tex = build_mesh_asset(
            raw.vertices,
            raw.faces,
            held.size_m,
            pinch_axis,
            cv2.cvtColor(crop_png, cv2.COLOR_BGRA2RGBA) if has_crop else None,
        )
        verts, faces = tex.vertices, tex.faces
        # MoGe only reconstructs the visible top surface, so the mesh comes out as a 1-2 cm slab
        # whatever the object really is. A pinch at the mid-height of such a slab puts the
        # fingertips on the table; extrude it down to a graspable thickness instead.
        ext = verts.max(0) - verts.min(0)
        # Object width the pinch closes across, under the fingertips (the pinch lands on the centre).
        thickness = footprint_chord_width(verts, pinch_axis)
        # The narrowest chord of a rounded or tapered footprint ends on faces that lean towards
        # each other, and the squeeze then pushes the object out of the grip (clip 114: contact
        # normals 115-143 deg apart). Turn the object so the most parallel pair faces the fingers.
        ap_axis, tan_alpha, _ = antipodal_axis_xy(
            verts, pinch_axis, max_width_m=thickness + ANTIPODAL_EXTRA_WIDTH_M
        )
        ap_yaw = np.arctan2(pinch_axis[1], pinch_axis[0]) - np.arctan2(ap_axis[1], ap_axis[0])
        if abs(ap_yaw) > 1e-6:
            c_, s_ = np.cos(ap_yaw), np.sin(ap_yaw)
            ctr = 0.5 * (verts.max(0) + verts.min(0))
            verts = (verts - ctr) @ np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]]) + ctr
            thickness = footprint_chord_width(verts, pinch_axis)
        print(
            f"antipodal pinch: object turned {np.degrees(ap_yaw):+.0f} deg, "
            f"wedge alpha {np.degrees(np.arctan(tan_alpha)):.0f} deg, width {thickness * 100:.1f} cm"
        )
        # ... but never taller than a few times its narrow width for the Sharpa: an invented 4 cm fin
        # on an 8 mm pen (clip 108) stands on its edge, the curled open fingers graze its top on the
        # way down and any one-sided touch that high tips it over (at 0.14 mg; sliding needs 1 mg).
        height = min(MIN_OBJECT_THICKNESS_M, MAX_EXTRUDE_ASPECT[hand] * thickness)
        extruded = ext[2] < height
        if extruded:
            verts = verts * np.array([1.0, 1.0, height / max(ext[2], 1e-4)])
        ext = verts.max(0) - verts.min(0)
        half = np.array([*(ext[:2] / 2), ext[2] / 2])
        print(
            f"object mesh: pinched width {thickness * 100:.1f} cm, height {ext[2] * 100:.1f} cm"
            + (" (extruded)" if extruded else "")
        )
        # The fingers open around the whole footprint, not just its centre chord: the tongs of
        # clip 100 are 5.3 cm across at the centre but 6.1 cm overall, and the open fingers
        # landed 0.5 cm outside the edges, i.e. on top of it at the first lateral error.
        open_width = min(float(np.ptp(verts[:, :2] @ pinch_axis)), thickness + MAX_OPEN_EXTRA_M)
        phrase = object_prompt(json.loads((clip / "episode.json").read_text())["task_description"])
        volume = 0.7 * trimesh.Trimesh(verts, faces).convex_hull.volume
        phys_file = clip / "physics.json"
        physics = (
            json.loads(phys_file.read_text())
            if phys_file.exists()
            else {**heuristic_physics(phrase, volume), "source": "heuristic"}
        )
        mesh_kw = dict(
            mesh_vertices=verts,
            mesh_faces=faces,
            # A 10 g estimate lets any contact spring or wrist jerk throw the object (clip 114).
            mass_kg=max(float(physics["mass"]), MIN_MASS_KG),
            friction=(physics["friction"], 0.01, 0.001),
            rgba=(1.0, 1.0, 1.0, 1.0) if tex.texture is not None else (0.75, 0.75, 0.75, 1.0),
            mesh_uv=tex.uv,
            mesh_texture=tex.texture,
        )
    else:
        thickness = 2 * half.min()
        open_width = thickness
        # The proxy box is axis-aligned, so a pinch axis at 45 deg to it closed the thumb on one face
        # and the index on the adjacent one (clip 101 Sharpa: the box squirted out sideways). Give it
        # explicit corners, narrow side along ``pinch_axis``, so the re-yaw below turns it square to
        # the commanded pinch.
        pinch_axis = np.array([1.0, 0.0]) if half[0] <= half[1] else np.array([0.0, 1.0])
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
        mesh_kw = dict(mesh_vertices=corners, mesh_faces=BOX_FACES)
        extruded = False
        tan_alpha = 0.0
    return half, mesh_kw, physics, thickness, open_width, pinch_axis, extruded, tan_alpha, tex, provenance
