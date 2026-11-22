"""MuJoCo tactile readout: per-region normal/tangential force and binary contact, sampled per frame.

Region names are the EgoTac region schema (5 fingertips, 5 finger pads, palm) so sim and human
tactile are directly comparable by name. Both robot hands are supported:

* ``dex3``   - three fingers, so ring/pinky have no counterpart (``has_region=False``, always zero);
* ``sharpa`` - the Sharpa Wave five-finger hand, every region is present.

Hand axis order is (left, right). Tips are the distal phalanx of each finger, pads the middle and
proximal phalanges, and the remaining hand bodies (wrist, palm/metacarpals) are read out as ``palm``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from stages.reconstruction.hands.tactile import REGION_NAMES

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CONTACT_FORCE_THRESHOLD_N = 0.05

# Dex3: one distal link per finger (tip), one proximal link (pad); ring/pinky do not exist.
_DEX3_TIP = {
    "thumb": ("hand_thumb_2_link",),
    "index": ("hand_index_1_link",),
    "middle": ("hand_middle_1_link",),
}
_DEX3_PAD = {
    "thumb": ("hand_thumb_1_link",),
    "index": ("hand_index_0_link",),
    "middle": ("hand_middle_0_link",),
}
_DEX3_PALM = ("wrist_yaw_link", "hand_thumb_0_link")
# Sharpa Wave: ``<side>_<finger>_{PP,MP,DP}`` phalanges plus the ``C_MC``/``*_MC`` metacarpals.
_SHARPA_PALM = ("wrist_yaw_link", "hand_C_MC", "thumb_MC", "pinky_MC")


def _region_bodies(side: str, hand: str) -> dict[str, tuple[str, ...]]:
    """Region name -> body names for one side; missing bodies are skipped by the caller."""
    if hand == "dex3":
        tip, pad, palm = _DEX3_TIP, _DEX3_PAD, _DEX3_PALM
    elif hand == "sharpa":
        tip = {f: (f"{f}_DP",) for f in FINGERS}
        pad = {f: (f"{f}_MP", f"{f}_PP") for f in FINGERS}
        palm = _SHARPA_PALM
    else:
        raise ValueError(f"unknown hand {hand!r}")
    out = {f"{f}_tip": tuple(f"{side}_{n}" for n in names) for f, names in tip.items()}
    out |= {f"{f}_pad": tuple(f"{side}_{n}" for n in names) for f, names in pad.items()}
    out["palm"] = tuple(f"{side}_{n}" for n in palm)
    return out


def detect_hand(model: mujoco.MjModel) -> str:
    """``"sharpa"`` when a Sharpa palm body is compiled into ``model``, else ``"dex3"``."""
    for side in SIDES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_C_MC") >= 0:
            return "sharpa"
    return "dex3"


class SimTactile:
    """Accumulates one (2, R) force/contact sample per ``update`` call.

    ``partner``: which geoms count as the touched surface: ``"object"`` (default), ``"table"`` or
    ``"object+table"``. Objects are bodies in ``object_names`` (default: every free-joint body).
    ``hand``: ``"auto"`` (default, from the compiled model), ``"dex3"`` or ``"sharpa"``.

    The table share of the force is also kept on its own (``support_normal_force_n``): the pads are the
    only sensor that tells the hand touching the support plane apart from gripping the object (the
    arm joint torques change by 1-3 N m on such a contact, within their 20 N m motion transients).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        object_names: tuple[str, ...] | None = None,
        partner: str = "object",
        table_name: str = "table",
        threshold_n: float = CONTACT_FORCE_THRESHOLD_N,
        hand: str = "auto",
    ):
        if partner not in ("object", "table", "object+table"):
            raise ValueError(partner)
        self.hand = detect_hand(model) if hand == "auto" else hand
        self.region_names = REGION_NAMES
        self.threshold_n = threshold_n
        present = {r: False for r in REGION_NAMES}
        self._geom_slot: dict[int, tuple[int, int]] = {}  # geom -> (hand, region)
        for h, side in enumerate(SIDES):
            for region, names in _region_bodies(side, self.hand).items():
                for name in names:
                    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    if b < 0:
                        continue
                    for g in np.where(model.geom_bodyid == b)[0]:
                        self._geom_slot[int(g)] = (h, REGION_NAMES.index(region))
                        present[region] = True
        self.has_region = np.array([present[r] for r in REGION_NAMES])
        if object_names is None:
            object_names = tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j]))
                for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            )
        self._partner_body: dict[int, str] = {}
        self._table_name = table_name
        for name in object_names if "object" in partner else ():
            self._add_partner(model, name)
        if "table" in partner:
            self._add_partner(model, table_name)
        self._force: list[np.ndarray] = []
        self._tangential: list[np.ndarray] = []
        self._contact: list[np.ndarray] = []
        self._object: list[np.ndarray] = []
        self._support: list[np.ndarray] = []
        self._wrench = np.zeros(6)

    def _add_partner(self, model: mujoco.MjModel, name: str) -> None:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b >= 0:
            for g in np.where(model.geom_bodyid == b)[0]:
                self._partner_body[int(g)] = name

    def sample(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current (normal (2,R), tangential (2,R), contact (2,R) bool, object_name (2,)), not recorded."""
        force, tangential, contact, names, _ = self._sample(model, data)
        return force, tangential, contact, names

    def _sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, ...]:
        force = np.zeros((len(SIDES), len(REGION_NAMES)))
        tangential = np.zeros_like(force)
        support = np.zeros_like(force)
        per_obj: list[dict[str, float]] = [{} for _ in SIDES]
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for hand_geom, other in ((g1, g2), (g2, g1)):
                if hand_geom in self._geom_slot and other in self._partner_body:
                    mujoco.mj_contactForce(model, data, i, self._wrench)
                    h, r = self._geom_slot[hand_geom]
                    fn = abs(float(self._wrench[0]))
                    force[h, r] += fn
                    tangential[h, r] += float(np.hypot(self._wrench[1], self._wrench[2]))
                    name = self._partner_body[other]
                    if name == self._table_name:
                        support[h, r] += fn
                    per_obj[h][name] = per_obj[h].get(name, 0.0) + fn
                    break
        force[:, ~self.has_region] = 0.0
        tangential[:, ~self.has_region] = 0.0
        support[:, ~self.has_region] = 0.0
        contact = force > self.threshold_n
        names = np.array([max(d, key=d.get) if d else "" for d in per_obj], dtype="<U64")
        return force, tangential, contact, names, support

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        force, tangential, contact, names, support = self._sample(model, data)
        self._support.append(support)
        self._force.append(force)
        self._tangential.append(tangential)
        self._contact.append(contact)
        self._object.append(names)

    def arrays(self) -> dict[str, np.ndarray]:
        n, r = len(self._force), len(REGION_NAMES)
        return {
            "normal_force_n": np.asarray(self._force, float).reshape(n, 2, r),
            "tangential_force_n": np.asarray(self._tangential, float).reshape(n, 2, r),
            "contact": np.asarray(self._contact, bool).reshape(n, 2, r),
            "region_names": np.array(REGION_NAMES),
            "has_region": self.has_region.copy(),
            "object_name": np.asarray(self._object, dtype="<U64").reshape(n, 2),
            "support_normal_force_n": np.asarray(self._support, float).reshape(n, 2, r),
            "support_contact": np.asarray(self._support, float).reshape(n, 2, r) > self.threshold_n,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.arrays())
        return path


def support_touch(arrays: dict[str, np.ndarray], timestamps_s: np.ndarray, side: str) -> dict | None:
    """When and where one hand's pads pressed on the support plane (table), or None if they never did."""
    h = SIDES.index(side)
    force = np.asarray(arrays["support_normal_force_n"], float)[:, h]
    n = min(len(force), len(timestamps_s))
    force, on = force[:n], np.asarray(arrays["support_contact"], bool)[:n, h]
    if not on.any():
        return None
    frames = np.where(on.any(axis=1))[0]
    names = [str(r) for r in arrays["region_names"]]
    return {
        "first_s": float(timestamps_s[frames[0]] - timestamps_s[0]),
        "last_s": float(timestamps_s[frames[-1]] - timestamps_s[0]),
        "fraction": float(len(frames) / n),
        "peak_n": float(force.max()),
        "regions": [names[r] for r in np.where(on.any(axis=0))[0]],
    }
