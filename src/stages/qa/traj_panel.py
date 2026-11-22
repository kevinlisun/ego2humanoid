"""Task-space trajectory panel: end-effector and fingertip paths, object pose, camera pose.

The panel is an isometric view of the simulation world (z up, the table plane as the ground grid).
It shows where the robot's hands and the manipulated object actually went, which the two camera
renders cannot convey: the rendered views are projections, so depth along the optical axis is
ambiguous, while this panel is metric in all three axes.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

from stages.manipulation.sim_tactile import FINGERS, SIDES, _region_bodies, detect_hand

TRAIL_S = 2.0  # length of the fading history drawn behind each marker
COLORS = {  # BGR
    "left": (255, 170, 60),
    "right": (90, 220, 255),
    "object": (90, 230, 120),
    "camera": (200, 160, 255),
    "grid": (70, 70, 70),
}


class TrajectoryRecorder:
    """``on_record`` hook collecting wrist, fingertip and object poses in sim world coordinates."""

    def __init__(self, object_body: str = "perceived", T_sim_camera: np.ndarray | None = None):
        self.object_body = object_body
        self.T_sim_camera = None if T_sim_camera is None else np.asarray(T_sim_camera, float)
        self._ids: dict[str, list[int]] | None = None
        self.wrist: list[np.ndarray] = []
        self.tips: list[np.ndarray] = []
        self.object_pos: list[np.ndarray] = []
        self.object_rot: list[np.ndarray] = []

    def _resolve(self, model: mujoco.MjModel) -> None:
        hand = detect_hand(model)
        wrists, tips = [], []
        for side in SIDES:
            wrists.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"))
            regions = _region_bodies(side, hand)
            for finger in FINGERS:
                bodies = regions.get(f"{finger}_tip", ())
                ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
                tips.append(next((i for i in ids if i >= 0), -1))  # -1: the hand lacks this finger
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.object_body)
        self._ids = {"wrist": wrists, "tips": tips, "object": [obj]}

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._ids is None:
            self._resolve(model)
        assert self._ids is not None
        self.wrist.append(np.array([data.xpos[i] for i in self._ids["wrist"]]))
        self.tips.append(np.array([data.xpos[i] if i >= 0 else [np.nan] * 3 for i in self._ids["tips"]]))
        obj = self._ids["object"][0]
        self.object_pos.append(data.xpos[obj].copy() if obj >= 0 else np.full(3, np.nan))
        self.object_rot.append(data.xmat[obj].reshape(3, 3).copy() if obj >= 0 else np.full((3, 3), np.nan))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "wrist_m": np.asarray(self.wrist),  # (N, 2, 3) left/right wrist_yaw body origin
            "fingertip_m": np.asarray(self.tips).reshape(len(self.tips), 2, len(FINGERS), 3),
            "object_pos_m": np.asarray(self.object_pos),  # (N, 3) object centre of mass
            "object_rot": np.asarray(self.object_rot),  # (N, 3, 3) object orientation
            "camera_T_sim": np.eye(4) if self.T_sim_camera is None else self.T_sim_camera,
        }


class IsoProjection:
    """Fixed isometric world -> pixel map, fitted once so the whole trajectory stays in frame."""

    def __init__(self, points_m: np.ndarray, size_hw: tuple[int, int], margin_px: int = 56):
        self.size = size_hw
        uv = self._raw(points_m.reshape(-1, 3))
        uv = uv[np.isfinite(uv).all(1)]
        lo, hi = uv.min(0), uv.max(0)
        span = np.maximum(hi - lo, 1e-3)
        h, w = size_hw
        self.scale = float(min((w - 2 * margin_px) / span[0], (h - 2 * margin_px) / span[1]))
        self.offset = np.array([w / 2, h / 2]) - self.scale * (lo + hi) / 2 * np.array([1, -1])

    @staticmethod
    def _raw(p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p, float))
        u = (p[:, 0] - p[:, 1]) * np.cos(np.pi / 6)
        v = (p[:, 0] + p[:, 1]) * np.sin(np.pi / 6) + p[:, 2]
        return np.stack([u, v], axis=-1)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        """World metres (..., 3) -> integer pixels (..., 2); v grows upwards, pixels downwards."""
        uv = self._raw(p) * self.scale * np.array([1, -1]) + self.offset
        return uv.reshape((*np.shape(p)[:-1], 2))


def _polyline(img, proj, pts, color, thickness=2):
    px = proj(np.asarray(pts))
    px = px[np.isfinite(px).all(-1)]
    if len(px) >= 2:
        cv2.polylines(img, [px.astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def _grid(img, proj, z_m, x_range, y_range, step=0.1):
    for x in np.arange(x_range[0], x_range[1] + 1e-6, step):
        _polyline(img, proj, [[x, y_range[0], z_m], [x, y_range[1], z_m]], COLORS["grid"], 1)
    for y in np.arange(y_range[0], y_range[1] + 1e-6, step):
        _polyline(img, proj, [[x_range[0], y, z_m], [x_range[1], y, z_m]], COLORS["grid"], 1)


def draw_trajectory_panel(
    traj: dict[str, np.ndarray],
    i: int,
    size_hw: tuple[int, int] = (480, 640),
    table_top_z_m: float = 0.80,
    fps: float = 50.0,
    proj: IsoProjection | None = None,
) -> np.ndarray:
    """Render frame ``i`` of the trajectory panel as an RGB image of ``size_hw``."""
    wrist, obj = traj["wrist_m"], traj["object_pos_m"]
    cam_pos = np.asarray(traj["camera_T_sim"])[:3, 3]
    if proj is None:
        proj = IsoProjection(np.concatenate([wrist.reshape(-1, 3), obj, cam_pos[None]]), size_hw)
    img = np.full((*size_hw, 3), 24, np.uint8)
    xs, ys = wrist[..., 0], wrist[..., 1]
    _grid(
        img,
        proj,
        table_top_z_m,
        (float(np.floor(xs.min() * 10) / 10), float(np.ceil(xs.max() * 10) / 10)),
        (float(np.floor(ys.min() * 10) / 10), float(np.ceil(ys.max() * 10) / 10)),
    )
    start = max(0, i - int(TRAIL_S * fps))
    for h, side in enumerate(SIDES):
        _polyline(img, proj, wrist[start : i + 1, h], COLORS[side], 2)
        for f in range(traj["fingertip_m"].shape[2]):
            _polyline(img, proj, traj["fingertip_m"][start : i + 1, h, f], COLORS[side], 1)
        p = proj(wrist[i, h]).astype(int)
        cv2.circle(img, tuple(p), 6, COLORS[side], -1, cv2.LINE_AA)
    if np.isfinite(obj[i]).all():
        _polyline(img, proj, obj[: i + 1], COLORS["object"], 2)  # the object keeps its full history
        c = proj(obj[i]).astype(int)
        cv2.circle(img, tuple(c), 7, COLORS["object"], -1, cv2.LINE_AA)
        for axis, col in enumerate(((60, 60, 255), (60, 255, 60), (255, 120, 60))):
            tip = obj[i] + 0.06 * traj["object_rot"][i][:, axis]
            cv2.line(img, tuple(c), tuple(proj(tip).astype(int)), col, 2, cv2.LINE_AA)
    cam = proj(cam_pos).astype(int)
    cv2.drawMarker(img, tuple(cam), COLORS["camera"], cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
    cv2.putText(img, "cam", (cam[0] + 10, cam[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["camera"], 1)

    lift = float(obj[i, 2] - obj[0, 2]) if np.isfinite(obj[i]).all() else float("nan")
    lines = [
        (f"t = {i / fps:5.2f} s", (220, 220, 220)),
        ("left wrist + fingertips", COLORS["left"]),
        ("right wrist + fingertips", COLORS["right"]),
        (f"object (lift {lift * 100:+.1f} cm)", COLORS["object"]),
        ("ego camera", COLORS["camera"]),
    ]
    for k, (text, col) in enumerate(lines):
        y = 78 + 22 * k
        cv2.rectangle(img, (8, y - 15), (8 + 11 * len(text), y + 6), (24, 24, 24), -1)
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
