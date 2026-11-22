"""L.5: the G1 head camera model shared by planner, renderer and exporter.

OpenCV convention for pixels (x right, y down, z forward). The camera is rigidly attached to
``torso_link`` so its pose always follows the measured forward kinematics (waist included).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stages.manipulation.mjcf_scene import D435_POS_IN_TORSO_M, _d435_camera_rotation

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/cameras/g1_head_cam_v1.json"
MJ_TO_CV = np.diag([1.0, -1.0, -1.0])  # MuJoCo cam (x right, y up, -z fwd) -> OpenCV


@dataclass(frozen=True)
class HeadCamera:
    height: int
    width: int
    fovy_deg: float
    parent_body: str = "torso_link"

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> HeadCamera:
        c = json.loads(Path(path).read_text())
        return cls(c["image_hw"][0], c["image_hw"][1], c["fovy_deg"], c["parent_body"])

    @property
    def K(self) -> np.ndarray:
        f = 0.5 * self.height / np.tan(np.radians(self.fovy_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])

    @property
    def fovx_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(0.5 * self.width / self.K[0, 0])))

    def T_world_cam(self, model, data) -> np.ndarray:
        """4x4 camera-to-world pose (OpenCV axes) from the measured torso FK."""
        b = model.body(self.parent_body)
        R_wt = np.asarray(data.xmat[b.id]).reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R_wt @ _d435_camera_rotation() @ MJ_TO_CV
        T[:3, 3] = data.xpos[b.id] + R_wt @ np.array(D435_POS_IN_TORSO_M)
        return T

    def project(self, T_world_cam: np.ndarray, pts_world: np.ndarray):
        """Return pixel coordinates (N, 2) and depth (N,) of world points."""
        pc = (np.linalg.inv(T_world_cam) @ np.c_[pts_world, np.ones(len(pts_world))].T)[:3].T
        uv = (self.K @ pc.T).T
        return uv[:, :2] / uv[:, 2:3], pc[:, 2]
