"""Floating-wrist Dex3 hand model for SPIDER/MJWP (which ships no Dex3 asset).

The hand subtree of the G1 MJCF (from the wrist-yaw link) hangs off a 6-DoF wrist (three slides,
three hinges) like SPIDER's ``xhand`` models. MJWP then optimises fingers and contacts while the
wrist tracks the arm-IK wrist pose; the arm itself is not simulated by MJWP.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mujoco

from robot.asset import ASSET_DIR, load_fixed_base_spec
from robot.pinch import _tip_local

WRIST_DOF_NAMES = ("tx", "ty", "tz", "roll", "pitch", "yaw")
_AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1))
_MASS_KG = 1e-3
WRIST_KP, WRIST_KV = 300.0, 20.0
FINGER_KP, FINGER_KV = 8.0, 0.4  # same as the replay PD (`PDConfig`)


def build_floating_hand_spec(side: str) -> mujoco.MjSpec:
    """Spec with prefix-free hand joint names (``<side>_hand_*``) plus ``<side>_float_<dof>`` joints."""
    if side not in ("left", "right"):
        raise ValueError(side)
    warnings.filterwarnings("ignore", message="Attaching a spec with an empty prefix")
    src = load_fixed_base_spec()
    dst = mujoco.MjSpec()
    dst.meshdir = str(ASSET_DIR / "meshes")
    dst.option.timestep = src.option.timestep
    dst.compiler.degree = False
    parent = dst.worldbody
    for i, (dof, axis) in enumerate(zip(WRIST_DOF_NAMES, _AXES, strict=True)):
        body = parent.add_body(name=f"{side}_float_{dof}_link")
        body.add_joint(
            name=f"{side}_float_{dof}",
            type=mujoco.mjtJoint.mjJNT_SLIDE if i < 3 else mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
        )
        body.add_geom(  # carries the inertia of the massless wrist chain; never collides
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-3, 0, 0], mass=_MASS_KG, contype=0, conaffinity=0
        )
        parent = body
    parent.add_frame().attach_body(src.body(f"{side}_wrist_yaw_link"), "", "")
    for act in [a for a in dst.actuators if a.target == f"{side}_wrist_yaw_joint"]:
        dst.delete(act)
    dst.delete(dst.joint(f"{side}_wrist_yaw_joint"))  # replaced by the 6-DoF wrist
    for act in dst.actuators:  # hand actuators came with the attached subtree
        act.set_to_position(kp=FINGER_KP, kv=FINGER_KV)
    for dof in WRIST_DOF_NAMES:
        name = f"{side}_float_{dof}"
        act = dst.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=WRIST_KP, kv=WRIST_KV)
    _add_spider_sites(dst, side)
    return dst


# SPIDER expects five fingertip sites per hand; Dex3 has three fingers, so ring/pinky alias the middle tip
# (SPIDER's keypoint IK must not be used with them, see docs/notes/spider_integration.md).
_TIP_BODY = {
    "thumb": "thumb_2",
    "index": "index_1",
    "middle": "middle_1",
    "ring": "middle_1",
    "pinky": "middle_1",
}


def _add_spider_sites(spec: mujoco.MjSpec, side: str) -> None:
    model = spec.compile()
    for finger, link in _TIP_BODY.items():
        name = f"{side}_hand_{link}_link"
        body = spec.body(name)
        tip = _tip_local(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for prefix in ("", "track_hand_", "trace_hand_"):
            body.add_site(name=f"{prefix}{side}_{finger}_tip", pos=list(tip), size=[0.005, 0, 0])
    spec.body(f"{side}_wrist_yaw_link").add_site(name=f"{side}_palm", pos=[0, 0, 0], size=[0.01, 0, 0])


def write_floating_hand_xml(side: str, out_dir: str | Path) -> Path:
    """Write the model with absolute mesh paths; returns the XML path."""
    spec = build_floating_hand_spec(side)
    spec.compile()  # validate before writing
    out = Path(out_dir) / f"dex3_{side}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return out
