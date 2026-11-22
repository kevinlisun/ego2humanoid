"""G1 with the Sharpa Wave five-finger tactile hand in place of the Dex3 three-finger hand.

The Dex3 hand (3 fingers, 7 joints per side) has a large embodiment gap to the human hand: the
MANO thumb/index/middle/ring/pinky curls have to be collapsed onto two opposing fingers. Sharpa
Wave is a 22-DoF five-finger hand with compliant fingertip pads (modelled as elastomer collision
geoms), so every human finger maps to a robot finger and the pads give a tactile readout.

The combined model is built in memory with ``MjSpec``: the Dex3 finger bodies, palm geoms and
actuators are deleted from each wrist, and the Sharpa hand body is attached to the wrist frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "robots" / "sharpa_wave"
HAND_MJCF = {side: ASSET_DIR / f"{side}_hand.xml" for side in ("left", "right")}
HAND_ROOT_BODY = {"left": "left_hand_C_MC", "right": "right_hand_C_MC"}
DEX3_FINGER_ROOTS = ("thumb_0", "index_0", "middle_0")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WRIST_OVERLAP_BODIES = ("thumb_CMC_VL", "thumb_MC")  # thumb base bodies that overlap the G1 wrist links

# Sharpa frame: fingers extend +z, they flex toward +x (palm normal), index->pinky spread along +y.
# G1 wrist-yaw frame: fingers extend +x, the Dex3 thumb opposes along -y, fingers spread along z.
SHARPA_TO_WRIST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# The two G1 wrist frames are *not* mirrored (both arms use the same joint-frame convention), so an
# anthropomorphic pair must have opposite palm normals in them: the Dex3 palms face -y (left) and +y
# (right), i.e. both inward. Attaching both Sharpa hands with SHARPA_TO_WRIST alone points both palms
# at -y, which leaves the right palm facing outward -- a valid right hand, but rolled 180 deg. The
# roll can be undone in the wrist command instead, but that burns most of the right arm's reachable
# workspace (unreachable frames went from 0 % to ~80 % on the right-handed clips), so undo it here.
ROLL_X_PI = np.diag([1.0, -1.0, -1.0])
ATTACH_R = {"left": SHARPA_TO_WRIST, "right": ROLL_X_PI @ SHARPA_TO_WRIST}
# Flange offset in the wrist-yaw frame. The Dex3 palm starts at x=0.0415; the Sharpa palm root is
# the hand's own wrist, and its fingertips reach ~0.20 m, so mounting it at the wrist origin keeps
# the wrist-to-fingertip length close to both the Dex3 hand (~0.19 m) and a human hand. The offset
# is expressed in the wrist frame, so it follows the per-side rotation.
FLANGE_POS_M = {"left": (0.0, 0.003, 0.0), "right": (0.0, -0.003, 0.0)}


def _quat_wxyz(R: np.ndarray) -> list[float]:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q.tolist()


def _strip_dex3(spec, side: str) -> None:
    """Delete one side's Dex3 finger bodies, their actuators and the palm geoms."""
    wrist = spec.body(f"{side}_wrist_yaw_link")
    for geom in list(wrist.geoms):
        if geom.meshname == f"{side}_hand_palm_link":
            spec.delete(geom)
    for root in DEX3_FINGER_ROOTS:
        body = spec.body(f"{side}_hand_{root}_link")
        names = _subtree_joint_names(body)
        for act in list(spec.actuators):
            if act.target in names:
                spec.delete(act)
        spec.delete(body)


def _subtree_joint_names(body) -> set[str]:
    names = {j.name for j in body.joints}
    for child in body.bodies:
        names |= _subtree_joint_names(child)
    return names


def attach_sharpa_hands(spec, sides: tuple[str, ...] = ("left", "right")):
    """Replace the Dex3 hands of ``spec`` (a G1 ``MjSpec``) with Sharpa Wave hands, in place."""
    import mujoco

    for side in sides:
        _strip_dex3(spec, side)
        hand = mujoco.MjSpec.from_file(str(HAND_MJCF[side]))
        _namespace_assets(hand, f"{side}_")
        wrist = spec.body(f"{side}_wrist_yaw_link")
        frame = wrist.add_frame(pos=list(FLANGE_POS_M[side]), quat=_quat_wxyz(ATTACH_R[side]))
        frame.attach_body(hand.body(HAND_ROOT_BODY[side]), "", "")
        # The thumb base sits inside the G1 wrist links once mounted (7 mm deep, ~650 N of permanent
        # contact force fighting the wrist actuators); that overlap is a mounting artefact, not a contact.
        for thumb in WRIST_OVERLAP_BODIES:
            for link in ("wrist_pitch_link", "wrist_yaw_link"):
                spec.add_exclude(bodyname1=f"{side}_{thumb}", bodyname2=f"{side}_{link}")
    # The compliant elastomer pads need an elliptic friction cone with a high impedance ratio,
    # otherwise fingertip contacts slip; the Sharpa MJCF sets these and attaching keeps the parent's.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    return spec


def _namespace_assets(hand, prefix: str) -> None:
    """Left and right hands share mesh/material/geom names; attaching both needs unique ones.

    Names that already carry the side (joints, sites, bodies, fingertip geoms) are left alone so
    the rest of the pipeline can address them directly.
    """
    import xml.etree.ElementTree as ET

    bodies = list(hand.bodies)
    geoms = [g for b in bodies for g in b.geoms]
    renamed: dict[str, str] = {}
    for group in (hand.meshes, hand.materials, hand.textures):
        for item in group:
            if item.name and not item.name.startswith(prefix):
                renamed[item.name] = item.name = prefix + item.name
    for item in geoms + [s for b in bodies for s in b.sites]:
        if item.name and not item.name.startswith(prefix):
            item.name = prefix + item.name
    for geom in geoms:  # asset references on the elements themselves
        geom.material = renamed.get(geom.material, geom.material)
        geom.meshname = renamed.get(geom.meshname, geom.meshname)
    classes = {d.get("class") for d in ET.parse(HAND_MJCF[prefix[:-1]]).iter("default")}
    for name in filter(None, classes):  # and on the default classes they inherit from
        default = hand.find_default(name)
        if default is not None:
            default.geom.material = renamed.get(default.geom.material, default.geom.material)
            default.geom.meshname = renamed.get(default.geom.meshname, default.geom.meshname)


def fingertip_site(side: str, finger: str) -> str:
    return f"{side}_{finger}_fingertip"


def fingertip_geom(side: str, finger: str) -> str:
    return f"{side}_{finger}_fingertip_collision"
