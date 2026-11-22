"""Simulation state for the Blender replay (Track BT, BT.16).

MuJoCo stays the only physics engine. ``BodyStateRecorder`` is an ``on_record`` hook that stores the world
pose of every body and camera; ``export_sim_model`` writes the constant part (visual geoms in their body
frames, mesh data). A renderer needs nothing else: ``geom pose = body pose @ geom local pose``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from stages.manipulation.sim_tactile import SimTactile
from stages.reconstruction.hands.tactile import REGION_NAMES

STATE_VERSION = "sim_state/2"  # v2 adds pad_force_N, pad_geom_names, object_contact; v1 files lack them
COLLISION_GROUP = 3  # geoms in this group are collision proxies, never rendered
CAMERAS = ("d435", "ego")


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, i: int) -> str:
    return mujoco.mj_id2name(model, kind, i) or f"{kind.name}_{i}"


def _pad_geoms(tac: SimTactile, model: mujoco.MjModel) -> list[int]:
    """Named geoms of the fingertip/finger-pad regions (palm excluded), the ones the tactile readout sums."""
    keep = {i for i, r in enumerate(REGION_NAMES) if r.endswith(("_tip", "_pad"))}
    ids = [g for g, (_, r) in tac._geom_slot.items() if r in keep]
    return sorted(g for g in ids if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))


class BodyStateRecorder:
    """``on_record`` hook: body and camera poses plus finger-pad contact force per record (sim world frame).

    Pad force uses the ``SimTactile`` geom->region map and partner set (free-joint objects + table, as the
    "sim pad force" panel); ``object_contact`` is true when any hand geom (palm included) touches an object.
    """

    def __init__(self, cameras: tuple[str, ...] = CAMERAS):
        self.cameras = cameras
        self._tac: SimTactile | None = None
        self._pad_ids: list[int] = []
        self._pad_names: list[str] = []
        self.pad_force: list[np.ndarray] = []
        self.object_contact: list[bool] = []
        self._cam_ids: list[int] = []
        self._body_names: list[str] = []
        self._cam_names: list[str] = []
        self.t: list[float] = []
        self.xpos: list[np.ndarray] = []
        self.xquat: list[np.ndarray] = []
        self.cam_pos: list[np.ndarray] = []
        self.cam_mat: list[np.ndarray] = []

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self._body_names:
            self._body_names = [_name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in self.cameras]
            self._cam_ids = [i for i in ids if i >= 0]
            self._cam_names = [c for c, i in zip(self.cameras, ids, strict=False) if i >= 0]
            self._tac = SimTactile(model, partner="object+table")
            self._pad_ids = _pad_geoms(self._tac, model)
            self._pad_names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self._pad_ids]
        self._record_contacts(model, data)
        self.t.append(float(data.time))
        self.xpos.append(data.xpos.astype(np.float32))
        self.xquat.append(data.xquat.astype(np.float32))
        self.cam_pos.append(data.cam_xpos[self._cam_ids].astype(np.float32))
        self.cam_mat.append(data.cam_xmat[self._cam_ids].reshape(-1, 3, 3).astype(np.float32))

    def _record_contacts(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        tac = self._tac
        assert tac is not None
        col = {g: i for i, g in enumerate(self._pad_ids)}
        force = np.zeros(len(self._pad_ids), np.float32)
        touching = False
        wrench = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            for hand_geom, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if hand_geom in tac._geom_slot and other in tac._partner_body:
                    mujoco.mj_contactForce(model, data, i, wrench)
                    fn = abs(float(wrench[0]))
                    if hand_geom in col:
                        force[col[hand_geom]] += fn
                    if tac._partner_body[other] != tac._table_name and fn > tac.threshold_n:
                        touching = True
                    break
        self.pad_force.append(force)
        self.object_contact.append(touching)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "version": np.array(STATE_VERSION),
            "t": np.asarray(self.t, np.float64),
            "body_names": np.array(self._body_names),
            "xpos": np.asarray(self.xpos),  # (N, B, 3)
            "xquat": np.asarray(self.xquat),  # (N, B, 4) wxyz
            "camera_names": np.array(self._cam_names),
            "cam_pos": np.asarray(self.cam_pos),  # (N, C, 3)
            "cam_mat": np.asarray(self.cam_mat),  # (N, C, 3, 3) columns x right, y up, z backward (MuJoCo)
            "pad_force_N": np.asarray(self.pad_force, np.float32).reshape(len(self.t), len(self._pad_names)),
            "pad_geom_names": np.array(self._pad_names, dtype="<U64"),  # (P,) collision geoms of the pads
            "object_contact": np.asarray(self.object_contact, bool),  # (N,)
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(path, **self.arrays())
        return path


def export_sim_model(model: mujoco.MjModel, out_dir: str | Path) -> Path:
    """Write ``sim_model.json`` + ``sim_model_meshes.npz``: every renderable geom in its body frame."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, np.ndarray] = {}
    geoms = []
    for g in range(model.ngeom):
        if int(model.geom_group[g]) == COLLISION_GROUP:
            continue
        gtype = mujoco.mjtGeom(int(model.geom_type[g]))
        entry = {
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])),
            "type": gtype.name.removeprefix("mjGEOM_").lower(),
            "size": [float(x) for x in model.geom_size[g]],
            "pos": [float(x) for x in model.geom_pos[g]],
            "quat": [float(x) for x in model.geom_quat[g]],
            "rgba": [float(x) for x in model.geom_rgba[g]],
            "collides": bool(model.geom_contype[g] or model.geom_conaffinity[g]),
        }
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            key = f"m{mid}"
            if key not in meshes:
                v0, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                f0, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                meshes[f"{key}_v"] = model.mesh_vert[v0 : v0 + vn].astype(np.float32)
                meshes[f"{key}_f"] = model.mesh_face[f0 : f0 + fn].astype(np.int32)
            entry["mesh"] = key
        geoms.append(entry)
    np.savez_compressed(out / "sim_model_meshes.npz", **meshes)
    # Pad force is read on collision proxies, which are not rendered: list them with their body so a
    # renderer can colour that body's visual geoms.
    pads = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])),
        }
        for g in _pad_geoms(SimTactile(model, partner="object+table"), model)
    ]
    (out / "sim_model.json").write_text(
        json.dumps(
            {
                "version": STATE_VERSION,
                "timestep": float(model.opt.timestep),
                "geoms": geoms,
                "pad_geoms": pads,
            },
            indent=1,
        )
    )
    return out / "sim_model.json"
