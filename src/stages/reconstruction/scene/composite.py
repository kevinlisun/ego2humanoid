"""Render the MuJoCo robot from an egocentric camera, optionally over a clean background plate.

Two camera conventions are supported. With a plate, the camera is the *human* camera recovered from
the source video (``T_sim_camera``, OpenCV axes) and the sim floor and table are dropped so the plate
supplies the real background. Without a plate, a camera compiled into the model is used directly --
in particular ``"d435"``, the robot's own head camera from the Unitree description -- and the sim
supplies the background.

Either way the camera sits at the robot's head, roughly 0.15 m from its own chest, so the torso and
legs would cover half the frame and hide the hands: they are drawn as a semi-transparent ghost
instead, which keeps the body in the picture without occluding the manipulation. Set
``ghost_alpha=0`` for third-person cameras, where the body must stay solid.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

HIDDEN_GEOM_GROUP = 5
COLLISION_GEOM_GROUP = 3


def free_camera(T_sim_camera: np.ndarray, distance: float = 1.0) -> mujoco.MjvCamera:
    """Free MuJoCo camera that looks along the optical axis of ``T_sim_camera`` (roll ignored)."""
    pos = np.asarray(T_sim_camera)[:3, 3]
    fwd = np.asarray(T_sim_camera)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pos + distance * fwd
    cam.distance = distance
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
    return cam


class CompositeRecorder:
    """``on_record`` callback: renders the scene and alpha-blends it onto ``plate_rgb``."""

    def __init__(
        self,
        plate_rgb: np.ndarray | None,
        T_sim_camera: np.ndarray,
        focal_px: float,
        source_wh: tuple[int, int],
        scale: float = 0.5,
        camera: mujoco.MjvCamera | str | None = None,
        hide_geoms: tuple[str, ...] = ("table_top",),
        ghost_body_substrings: tuple[str, ...] = ("pelvis", "hip", "knee", "ankle", "waist", "torso"),
        ghost_alpha: float = 0.45,
    ):
        w, h = source_wh
        self.size = (int(round(h * scale)), int(round(w * scale)))  # (H, W)
        # plate_rgb=None: render the full sim scene (floor, table, robot) from ``camera``, no compositing.
        self.plate = (
            None
            if plate_rgb is None
            else cv2.resize(plate_rgb, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        )
        self.fovy = float(np.degrees(2 * np.arctan(h / (2 * focal_px))))
        # A string selects a camera compiled into the model (e.g. the "ego" camera, which unlike a free
        # camera carries the measured roll); it also brings its own field of view.
        self.cam = camera if camera is not None else free_camera(T_sim_camera)
        self.hide = hide_geoms
        self.ghost_bodies = ghost_body_substrings
        self.ghost_alpha = float(ghost_alpha)
        self.frames: list[np.ndarray] = []
        self._renderer: mujoco.Renderer | None = None
        self._hide_ids: list[int] = []
        self._ghost_ids: list[int] = []
        # Geoms to leave out of a pass are parked in this group, which the render option disables, so
        # they are not drawn at all. Masking them out of the alpha channel alone is not enough: they
        # would still be rendered in front of the hands and the plate would show through.
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 1
        self._opt.geomgroup[HIDDEN_GEOM_GROUP] = 0
        self._opt.geomgroup[COLLISION_GEOM_GROUP] = 0  # Sharpa collision meshes are green, 25% opaque

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._renderer is None:
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.size[0])
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.size[1])
            self._renderer = mujoco.Renderer(model, *self.size)
            self._hide_ids = [
                i
                for i in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.hide)
                if i >= 0
            ]
            # The plate already shows the real floor and table; the robot's own body becomes a ghost.
            for g in range(model.ngeom):
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                    self._hide_ids.append(g)
                elif any(k in body for k in self.ghost_bodies):
                    self._ghost_ids.append(g)
        r = self._renderer
        # Set per call, not once: several recorders with different fields of view may share one model.
        if not isinstance(self.cam, str):
            model.vis.global_.fovy = self.fovy if self.plate is not None else 45.0
        ghost = self._ghost_ids if self.ghost_alpha > 0 else []
        if self.plate is None and not ghost:
            r.update_scene(data, self.cam)
            self.frames.append(r.render().copy())
            return
        # Without a plate the sim supplies its own background, so nothing is dropped.
        hide = self._hide_ids if self.plate is not None else []
        # Two passes: without the body (so the hands are never occluded) and with it (for the ghost).
        rgb, mask = self._pass(model, data, hide + ghost)
        out = self.plate.astype(np.float32) if self.plate is not None else rgb
        if ghost:
            body_rgb, body_mask = self._pass(model, data, hide, keep=ghost)
            g = body_mask * self.ghost_alpha
            out = body_rgb * g + out * (1 - g)
        if self.plate is None:
            self.frames.append(out.astype(np.uint8))
            return
        self.frames.append((rgb * mask + out * (1 - mask)).astype(np.uint8))

    def _pass(
        self, model: mujoco.MjModel, data: mujoco.MjData, drop: list[int], keep: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render with ``drop`` removed; return the RGB and the soft mask of the rendered geoms
        (restricted to ``keep`` when given)."""
        r = self._renderer
        assert r is not None
        saved = model.geom_group[drop].copy()
        model.geom_group[drop] = HIDDEN_GEOM_GROUP
        try:
            r.update_scene(data, self.cam, self._opt)
            rgb = r.render().copy()
            r.enable_segmentation_rendering()
            r.update_scene(data, self.cam, self._opt)
            seg = r.render()
            r.disable_segmentation_rendering()
        finally:
            model.geom_group[drop] = saved
        # A plain int: against the enum object numpy compares every pixel in Python (~150 ms a frame).
        hit = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        if keep is not None:
            hit &= np.isin(seg[..., 0], keep)
        return rgb.astype(np.float32), cv2.GaussianBlur(hit.astype(np.float32), (3, 3), 0)[..., None]

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


def write_side_by_side(
    video_path: str | Path,
    frames: list[np.ndarray],
    out_path: str | Path,
    t0_s: float,
    fps: float = 50.0,
    src_fps: float = 30.0,
    labels: tuple[str, str] = ("source RGB", "robot in reconstructed scene"),
) -> Path:
    import av

    h, w = frames[0].shape[:2]
    cap = cv2.VideoCapture(str(video_path))
    with av.open(str(out_path), "w") as out:
        stream = out.add_stream("libx264", rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = 2 * w, h, "yuv420p"
        for i, comp in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t0_s + i / fps) * src_fps)))
            ok, src = cap.read()
            if not ok:
                break
            src = cv2.cvtColor(cv2.resize(src, (w, h)), cv2.COLOR_BGR2RGB)
            panel = np.hstack([src, comp]).copy()
            for k, text in enumerate(labels):
                cv2.putText(panel, text, (20 + k * w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(panel, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return Path(out_path)
