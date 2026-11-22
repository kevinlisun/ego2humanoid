"""Tabletop scenario (L.4): robot + floor + table + obstacles + one rigid object + goal zone.

Everything is derived from a JSON config; the 2D occupancy footprint comes from the same boxes.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robot.asset import MJCF_PATH
from robot.sharpa import attach_sharpa_hands
from stages.manipulation.mjcf_scene import add_object, mesh_shape

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/scenarios/tabletop_v1.json"
GOAL_RGBA = (0.1, 0.8, 0.2, 0.4)


@dataclass(frozen=True)
class Scenario:
    cfg: dict

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> Scenario:
        return cls(json.loads(Path(path).read_text()))

    @property
    def table_top_z(self) -> float:
        return float(self.cfg["table"]["top_z"])

    def object_start_xyz(self) -> np.ndarray:
        t, o = self.cfg["table"], self.cfg["object"]
        xy = np.array(t["center_xy"]) + np.array(o["xy_on_table"])
        return np.array([*xy, self.table_top_z + o["half_size"][2] + 1e-3])

    def goal_xy(self) -> np.ndarray:
        return np.array(self.cfg["table"]["center_xy"]) + np.array(self.cfg["goal_zone"]["xy_on_table"])

    def build_spec(self, hand: str = "dex3") -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
        if hand == "sharpa":
            attach_sharpa_hands(spec)
        w = spec.worldbody
        t = self.cfg["table"]
        hx, hy, hz = t["half_size"]
        w.add_geom(
            name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
            pos=[*t["center_xy"], t["top_z"] - hz], rgba=t.get("rgba", [0.6, 0.45, 0.3, 1]),
        )  # fmt: skip
        for i, sx in enumerate((-1, 1)):
            for j, sy in enumerate((-1, 1)):
                w.add_geom(
                    name=f"table_leg_{i}{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.02, 0.02, (t["top_z"] - 2 * hz) / 2],
                    pos=[
                        t["center_xy"][0] + sx * (hx - 0.03),
                        t["center_xy"][1] + sy * (hy - 0.03),
                        (t["top_z"] - 2 * hz) / 2,
                    ],
                    rgba=[0.4, 0.3, 0.2, 1],
                )  # fmt: skip
        for ob in self.cfg["obstacles"]:
            h = ob["half_size"]
            w.add_geom(
                name=f"obstacle_{ob['name']}", type=mujoco.mjtGeom.mjGEOM_BOX, size=h,
                pos=[*ob["center_xy"], h[2]], rgba=[0.5, 0.5, 0.55, 1],
            )  # fmt: skip
        gz = self.cfg["goal_zone"]
        w.add_geom(
            name="goal_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[gz["radius"], 0.0005, 0],
            pos=[*self.goal_xy(), self.table_top_z + 0.0005], rgba=list(gz.get("rgba", GOAL_RGBA)), contype=0, conaffinity=0,
        )  # fmt: skip
        for ob in self.cfg.get("decor", ()):
            # Reconstructed scenery: shown, never touched (the robot cannot collide with it).
            add_object(spec, dataclasses.replace(ob, static=True, collision_pieces=()))
            for g in spec.body(ob.name).geoms:
                g.contype = g.conaffinity = 0
        o = self.cfg["object"]
        body = w.add_body(name=o["name"], pos=list(self.object_start_xyz()))
        body.add_freejoint(name=f"{o['name']}_free")
        visual = self.cfg.get("object_visual")
        body.add_geom(
            name=f"{o['name']}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=o["half_size"], mass=o["mass"],
            friction=[1.0, 0.005, 0.0001], rgba=[0.9, 0.2, 0.2, 0.0 if visual is not None else 1],
        )  # fmt: skip
        if visual is not None:
            # The perceived mesh rides on the collision proxy; it only makes the render match the video.
            shape, material = mesh_shape(spec, visual)
            body.add_geom(
                name=f"{o['name']}_visual", mass=1e-6, contype=0, conaffinity=0, rgba=list(visual.rgba),
                **shape, **({"material": material} if material else {}),
            )  # fmt: skip
        return spec

    def build(self, hand: str = "dex3") -> tuple[mujoco.MjModel, mujoco.MjData]:
        m = self.build_spec(hand).compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d

    def reset_object(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        j = m.joint(f"{self.cfg['object']['name']}_free")
        a = j.qposadr[0]
        d.qpos[a : a + 7] = [*self.object_start_xyz(), 1, 0, 0, 0]
        d.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0

    def obstacle_boxes(self, robot_margin: float = 0.0) -> list[tuple[float, float, float, float]]:
        """Axis-aligned XY boxes (xmin, ymin, xmax, ymax) the robot base must avoid; the table counts."""
        out = []
        t = self.cfg["table"]
        items = [(t["center_xy"], t["half_size"])] + [
            (o["center_xy"], o["half_size"]) for o in self.cfg["obstacles"]
        ]
        for c, h in items:
            out.append(
                (
                    c[0] - h[0] - robot_margin,
                    c[1] - h[1] - robot_margin,
                    c[0] + h[0] + robot_margin,
                    c[1] + h[1] + robot_margin,
                )
            )
        return out

    def occupancy(
        self, extent=((-1.0, 5.0), (-2.0, 2.0)), res: float = 0.05, robot_margin: float = 0.0
    ) -> np.ndarray:
        """Boolean grid [ny, nx]; True = blocked for the robot base footprint (inflated by margin)."""
        (x0, x1), (y0, y1) = extent
        xs, ys = np.arange(x0, x1, res) + res / 2, np.arange(y0, y1, res) + res / 2
        gx, gy = np.meshgrid(xs, ys)
        grid = np.zeros_like(gx, dtype=bool)
        for xmin, ymin, xmax, ymax in self.obstacle_boxes(robot_margin):
            grid |= (gx >= xmin) & (gx <= xmax) & (gy >= ymin) & (gy <= ymax)
        return grid
