"""Robot-in-the-loop neural rendering of the ego view: MuJoCo foreground over a Gaussian background.

The robot, its hands and the reconstructed object meshes are simulated, so they come from MuJoCo with
a real depth buffer. Everything else -- table, backdrop, floor -- belongs to the reconstructed scene
and is rasterised from the background splat (``bg_splat.py``) at the *same* camera pose. The two are
merged per pixel by depth, not by alpha, so the robot occludes the table where it is in front of it
and the table occludes the robot where it is behind, without any manual layering.

The splat is rasterised by gsplat, which needs CUDA and therefore a different environment than the
rest of the pipeline. The two halves are consequently decoupled through files: this module records
the MuJoCo half (``EgoNeuralRecorder``), ``scripts/render_bg_splat.py`` renders the splat half, and
``composite_neural_ego`` merges them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from stages.manipulation.mjcf_scene import D435_IMAGE_HW

HIDDEN_GEOM_GROUP = 5
COLLISION_GEOM_GROUP = 3
FAR_DEPTH_M = 1e4
GHOST_BODIES = ("pelvis", "hip", "knee", "ankle", "waist", "torso")


class EgoNeuralRecorder:
    """``on_record`` callback: captures what the neural compositor needs from the MuJoCo rollout.

    Per frame it stores the RGB and depth of the *simulated* content only (robot plus reconstructed
    object meshes, with the sim floor and table dropped because the splat supplies them), the same
    for the robot's own body alone (which the ego camera sees from 0.15 m and which is later ghosted
    in), and the world pose of the camera so the splat can be rendered from exactly the same place.
    """

    def __init__(
        self,
        camera: str = "d435",
        size: tuple[int, int] = D435_IMAGE_HW,
        hide_geoms: tuple[str, ...] = ("table_top",),
        ghost_body_substrings: tuple[str, ...] = GHOST_BODIES,
        drop_background: bool = True,
    ):
        self.camera = camera
        self.size = size
        # With a background splat the sim room must go, because the splat replaces it. Without one
        # there is nothing to put in its place, so the sim's own floor and table are kept.
        self.hide = hide_geoms if drop_background else ()
        self.drop_background = drop_background
        self.ghost_bodies = ghost_body_substrings
        self.rgb: list[np.ndarray] = []
        self.depth: list[np.ndarray] = []
        self.body_rgb: list[np.ndarray] = []
        self.body_mask: list[np.ndarray] = []
        self.cam_T_world: list[np.ndarray] = []
        self.fovy_deg = 0.0
        self._renderer: mujoco.Renderer | None = None
        self._cam_id = -1
        self._hide_ids: list[int] = []
        self._ghost_ids: list[int] = []
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 1
        self._opt.geomgroup[HIDDEN_GEOM_GROUP] = 0
        self._opt.geomgroup[COLLISION_GEOM_GROUP] = 0  # Sharpa collision meshes are green, 25% opaque

    def _setup(self, model: mujoco.MjModel) -> None:
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.size[0])
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.size[1])
        self._renderer = mujoco.Renderer(model, *self.size)
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if self._cam_id < 0:
            raise ValueError(f"camera {self.camera!r} is not compiled into the model")
        self.fovy_deg = float(model.cam_fovy[self._cam_id])
        self._hide_ids = [
            i for i in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.hide) if i >= 0
        ]
        for g in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            if self.drop_background and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                self._hide_ids.append(g)
            elif any(k in body for k in self.ghost_bodies):
                self._ghost_ids.append(g)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._renderer is None:
            self._setup(model)
        rgb, depth, _ = self._pass(model, data, self._hide_ids + self._ghost_ids)
        brgb, bdepth, bseg = self._pass(model, data, self._hide_ids, keep=self._ghost_ids)
        self.rgb.append(rgb)
        self.depth.append(depth)
        self.body_rgb.append(brgb)
        self.body_mask.append(bseg)
        t = np.eye(4)
        t[:3, :3] = np.asarray(data.cam_xmat[self._cam_id]).reshape(3, 3)
        t[:3, 3] = np.asarray(data.cam_xpos[self._cam_id])
        self.cam_T_world.append(t)
        _ = bdepth

    def _pass(
        self, model: mujoco.MjModel, data: mujoco.MjData, drop: list[int], keep: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self._renderer
        assert r is not None
        saved = model.geom_group[drop].copy()
        model.geom_group[drop] = HIDDEN_GEOM_GROUP
        try:
            r.update_scene(data, self.camera, self._opt)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(data, self.camera, self._opt)
            depth = r.render().copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            r.update_scene(data, self.camera, self._opt)
            seg = r.render()
            r.disable_segmentation_rendering()
        finally:
            model.geom_group[drop] = saved
        # A plain int: against the enum object numpy compares every pixel in Python (~150 ms a frame).
        hit = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        if keep is not None:
            hit &= np.isin(seg[..., 0], keep)
        depth = np.where(hit, depth, FAR_DEPTH_M).astype(np.float32)
        return rgb.astype(np.uint8), depth, hit.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rgb=np.stack(self.rgb),
            depth=np.stack(self.depth),
            body_rgb=np.stack(self.body_rgb),
            body_mask=np.stack(self.body_mask).astype(np.float32),
            cam_T_world=np.stack(self.cam_T_world).astype(np.float32),
            fovy_deg=np.float32(self.fovy_deg),
            size=np.asarray(self.size, np.int32),
        )
        return path

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


def composite_neural_ego(
    ego_npz: str | Path,
    bg_npz: str | Path | None,
    ghost_alpha: float = 0.45,
    hole_fill_px: int = 6,
) -> list[np.ndarray]:
    """Merge the MuJoCo ego capture with the rasterised background splat, by depth."""
    ego = np.load(ego_npz)
    rgb, depth = ego["rgb"].astype(np.float32), ego["depth"]
    out = rgb.copy()
    if bg_npz is not None and Path(bg_npz).exists():
        bg = np.load(bg_npz)
        bg_rgb, bg_depth, bg_alpha = bg["rgb"].astype(np.float32), bg["depth"], bg["alpha"]
        n = min(len(out), len(bg_rgb))
        out, rgb, depth = out[:n], rgb[:n], depth[:n]
        bg_rgb, bg_depth, bg_alpha = bg_rgb[:n], bg_depth[:n], bg_alpha[:n]
        # A Gaussian built from one view leaves holes where the new view sees around an edge; the
        # sim's own render of the same region is the only honest filler we have.
        seen = bg_alpha > 0.5
        if hole_fill_px:
            k = np.ones((hole_fill_px, hole_fill_px), np.uint8)
            seen = np.stack([cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0 for s in seen])
        front = seen & (bg_depth < depth)
        out = np.where(front[..., None], bg_rgb, rgb)
        # Where neither the splat nor the sim drew anything the ego camera is looking past the
        # reconstruction; extend the background into it rather than showing a black band.
        empty = (~seen) & (depth >= FAR_DEPTH_M)
        out = np.stack(
            [
                cv2.inpaint(o.astype(np.uint8), e.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
                if e.any()
                else o
                for o, e in zip(out, empty, strict=True)
            ]
        )
    body_rgb = ego["body_rgb"].astype(np.float32)[: len(out)]
    g = ego["body_mask"][: len(out), ..., None] * ghost_alpha
    out = body_rgb * g + out * (1 - g)
    return [f.astype(np.uint8) for f in np.clip(out, 0, 255)]
