"""Sharpa precision pinch (L.16/L.18): thumb-index pad pinch of a small object held beside the palm.

Poses are solved once with a small least-squares on fingertip sites in the wrist-link frame and stored in
``configs/grasp/sharpa_pinch_v1.json``; at run time they are blended open -> touch -> squeeze. The
other fingers stay extended.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

from robot.sharpa import fingertip_site

ROOT = Path(__file__).resolve().parents[3]
POSES_PATH = ROOT / "configs/grasp/sharpa_pinch_v1.json"
MAPPING_PATH = ROOT / "configs/finger_mapping.yaml"
# Pinch point in the wrist-yaw-link frame: fingers extend +x, the palm normal is +y, thumb side is +z.
PINCH_POINT_WRIST = np.array([0.102, 0.07, 0.02])
# Power-wrap blend between the open and closed mapping poses (thin vertical box in the finger cup).
WRAP_BLEND = {"open": 0.0, "touch": 0.7, "squeeze": 1.0}
OPEN_GAP_M, TOUCH_MARGIN_M, SQUEEZE_M = 0.14, 0.005, 0.006
THUMB = ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")
INDEX = ("MCP_FE", "MCP_AA", "PIP", "DIP")


def _mapping() -> dict:
    return yaml.safe_load(MAPPING_PATH.read_text())["sharpa"]


def finger_pose(kind: str, side: str = "right") -> dict[str, float]:
    """All hand joints at the ``open`` or ``closed`` (fist) values of the generic mapping."""
    y = _mapping()
    out = {}
    for finger, suffixes in y["joint_suffixes"].items():
        src = y[kind]["thumb" if finger == "thumb" else "finger"]
        out.update({f"{side}_{finger}_{s}": float(src[s]) for s in suffixes})
    return out


def _tip_in_wrist(model, data, side: str, finger: str) -> np.ndarray:
    wrist = data.xpos[model.body(f"{side}_wrist_yaw_link").id]
    R = data.xmat[model.body(f"{side}_wrist_yaw_link").id].reshape(3, 3)
    return R.T @ (data.site_xpos[model.site(fingertip_site(side, finger)).id] - wrist)


def solve_pinch(model: mujoco.MjModel, object_width_m: float, side: str = "right") -> dict:
    """Joint sets for the open / touch / squeeze stages of a pinch of the given width."""
    data = mujoco.MjData(model)
    names = [f"{side}_thumb_{s}" for s in THUMB] + [f"{side}_index_{s}" for s in INDEX]
    jid = [model.joint(n).id for n in names]
    adr = [model.jnt_qposadr[j] for j in jid]
    lo, hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
    base = finger_pose("open", side)
    x0 = np.array([base[n] for n in names])

    def tips(x):
        data.qpos[adr] = x
        mujoco.mj_kinematics(model, data)
        return _tip_in_wrist(model, data, side, "thumb"), _tip_in_wrist(model, data, side, "index")

    out, x, x_touch = {}, x0.copy(), None
    for stage, gap in (
        ("touch", object_width_m + TOUCH_MARGIN_M),
        ("squeeze", object_width_m - SQUEEZE_M),
        ("open", OPEN_GAP_M),
    ):
        dx = np.array([gap / 2, 0, 0])
        target_thumb, target_index = PINCH_POINT_WRIST + dx, PINCH_POINT_WRIST - dx
        free = np.arange(len(names)) >= len(THUMB) if stage == "open" else np.ones(len(names), bool)
        # open stage: the thumb stays in its neutral open pose (out of the way above the object) and only
        # the index tip is placed behind the object; the thumb then sweeps in during touch.

        def residual(v, tt=target_thumb, ti=target_index, free=free, stage=stage):
            full = x0.copy()
            full[free] = v
            t, i = tips(full)
            r = [i - ti, 0.01 * (full - x0)]
            return np.concatenate(r if stage == "open" else [t - tt, *r])

        start = x_touch if stage == "open" else x
        x = x0.copy()
        x[free] = least_squares(residual, start[free], bounds=(lo[free], hi[free])).x
        if stage == "touch":
            x_touch = x.copy()
        t, i = tips(x)
        out[stage] = {
            "joints": {n: round(float(v), 4) for n, v in zip(names, x, strict=True)},
            "gap_m": round(float(np.linalg.norm(t - i)), 4),
            "tip_err_m": round(float(np.linalg.norm(i - target_index)), 4)
            if stage == "open"
            else round(float(np.linalg.norm(t - target_thumb) + np.linalg.norm(i - target_index)), 4),
        }
    return out


class PinchPoses:
    """Run-time access to the stored pinch joint sets (right-hand names; mirrored by prefix)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @classmethod
    def load(cls, path: Path = POSES_PATH) -> PinchPoses:
        return cls(json.loads(Path(path).read_text()))

    @property
    def object_width_m(self) -> float:
        return float(self.cfg["object_width_m"])

    def pose(self, stage: str, side: str = "right") -> dict[str, float]:
        """``carry`` (fist) or a power-wrap blend open -> closed (``open``/``touch``/``squeeze``)."""
        closed = finger_pose("closed", side)
        if stage == "carry":
            return closed
        opened = finger_pose("open", side)
        s = WRAP_BLEND[stage]
        return {k: opened[k] + s * (closed[k] - opened[k]) for k in opened}
