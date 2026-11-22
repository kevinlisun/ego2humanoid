"""Sim bundle: the frozen plan plus references of one clip, as npz + json (Track RSI, RSI.8).

Replaces the `E2H_DUMP` pickle. `allow_pickle=False`; a bundle is enough to rebuild the physics
scene and roll the planned trajectory out again with different physics parameters.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.sim_params import SimParams
from skills.params.resolve import params_hash, plan_params_hash
from stages.manipulation.mjcf_scene import BoxObject
from stages.manipulation.sim_eval import SimCase

BUNDLE_VERSION = "sim_bundle/1"
_MESH_KEYS = ("mesh_vertices", "mesh_faces", "mesh_uv", "mesh_texture")


@dataclass
class SimBundle:
    case: SimCase
    meta: dict[str, Any]  # json part: clip, hand, side, params, phases_t_s, best_offset_m, ...
    refs: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def params(self) -> SimParams:
        return SimParams.model_validate(self.meta["params"])

    @property
    def best_offset_m(self) -> tuple[float, float]:
        return tuple(self.meta["best_start_xy_m"])


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _put_geometry(arrays: dict, prefix: str, kw: dict) -> dict:
    """Store mesh arrays of `kw` under `prefix`; return the scalar remainder (json-able)."""
    rest = {}
    for k, v in kw.items():
        if k in _MESH_KEYS:
            if v is not None:
                arrays[f"{prefix}_{k}"] = np.asarray(v)
        elif k == "collision_pieces":
            for j, (pv, pf) in enumerate(v):
                arrays[f"{prefix}_piece{j}_v"] = np.asarray(pv)
                arrays[f"{prefix}_piece{j}_f"] = np.asarray(pf)
            rest["n_pieces"] = len(v)
        else:
            rest[k] = _jsonable(v)
    return rest


def _get_geometry(z, prefix: str, rest: dict) -> dict:
    kw = dict(rest)
    n_pieces = int(kw.pop("n_pieces", 0))
    for k in _MESH_KEYS:
        if f"{prefix}_{k}" in z:
            kw[k] = z[f"{prefix}_{k}"]
    if n_pieces:
        kw["collision_pieces"] = tuple(
            (z[f"{prefix}_piece{j}_v"], z[f"{prefix}_piece{j}_f"]) for j in range(n_pieces)
        )
    for k in ("friction", "rgba"):
        if k in kw:
            kw[k] = tuple(kw[k])
    return kw


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def save_bundle(
    out_dir: Path,
    case: SimCase,
    params: SimParams,
    *,
    clip: str,
    side: str,
    best_start_xy_m: tuple[float, float],
    phases_t_s: dict[str, tuple[float, float]],
    refs: dict[str, np.ndarray] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "timestamps_s": case.timestamps_s,
        "q_target_rad": case.q_target_rad,
        "T_sim_robot_base": case.T_sim_robot_base,
        "anchor_m": case.anchor,
        "half_size_m": case.half_size_m,
    }
    if case.drop_point_m is not None:
        arrays["drop_point_m"] = np.asarray(case.drop_point_m, float)
    held_rest = _put_geometry(arrays, "held", case.mesh_kw)
    objects = []
    for i, o in enumerate(case.extra):
        rest = _put_geometry(
            arrays,
            f"obj{i}",
            dict(
                mesh_vertices=o.mesh_vertices,
                mesh_faces=o.mesh_faces,
                mesh_uv=o.mesh_uv,
                mesh_texture=o.mesh_texture,
                collision_pieces=o.collision_pieces,
            ),
        )
        objects.append(
            dict(
                name=o.name,
                half_size_m=_jsonable(o.half_size_m),
                pos_m=_jsonable(o.pos_m),
                mass_kg=o.mass_kg,
                friction=_jsonable(o.friction),
                rgba=_jsonable(o.rgba),
                static=o.static,
                robot_collision=o.robot_collision,
                **rest,
            )
        )
    for k, v in (refs or {}).items():
        arrays[f"ref_{k}"] = np.asarray(v)
    meta = dict(
        version=BUNDLE_VERSION,
        clip=clip,
        hand=case.hand,
        side=side,
        params_hash=params_hash(params),
        plan_params_hash=plan_params_hash(params),
        params=json.loads(params.model_dump_json()),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_git_commit(),
        table_kw=_jsonable(case.table_kw),
        table_z=float(case.table_z),
        held=held_rest,
        objects=objects,
        receptacle_name=case.receptacle_name,
        base_flags={k: list(v) for k, v in case.base_flags.items()},
        held_base=None if case.held_base is None else _jsonable(case.held_base),
        demo_carry_m=float(case.demo_carry_m),
        release_t_s=float(case.release_t_s),
        best_start_xy_m=[float(best_start_xy_m[0]), float(best_start_xy_m[1])],
        phases_t_s={k: [float(a), float(b)] for k, (a, b) in phases_t_s.items()},
        **(extra_meta or {}),
    )
    np.savez_compressed(out_dir / "sim_bundle.npz", **arrays)
    (out_dir / "sim_bundle.json").write_text(json.dumps(meta, indent=1))
    return out_dir / "sim_bundle.json"


def load_bundle(out_dir: Path) -> SimBundle:
    meta = json.loads((out_dir / "sim_bundle.json").read_text())
    if meta.get("version") != BUNDLE_VERSION:
        raise ValueError(f"unsupported sim bundle version {meta.get('version')!r}")
    z = np.load(out_dir / "sim_bundle.npz", allow_pickle=False)
    extra = []
    for i, o in enumerate(meta["objects"]):
        o = dict(o)
        kw = _get_geometry(z, f"obj{i}", {k: o.pop(k) for k in list(o) if k == "n_pieces"})
        extra.append(
            BoxObject(
                name=o["name"],
                half_size_m=tuple(o["half_size_m"]),
                pos_m=tuple(o["pos_m"]),
                mass_kg=o["mass_kg"],
                friction=tuple(o["friction"]),
                rgba=tuple(o["rgba"]),
                static=o["static"],
                robot_collision=o["robot_collision"],
                **kw,
            )
        )
    table_kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["table_kw"].items()}
    case = SimCase(
        hand=meta["hand"],
        T_sim_robot_base=z["T_sim_robot_base"],
        table_kw=table_kw,
        timestamps_s=z["timestamps_s"],
        q_target_rad=z["q_target_rad"],
        anchor=z["anchor_m"],
        half_size_m=z["half_size_m"],
        table_z=meta["table_z"],
        mesh_kw=_get_geometry(z, "held", meta["held"]),
        extra=tuple(extra),
        drop_point_m=z["drop_point_m"] if "drop_point_m" in z else None,
        demo_carry_m=meta["demo_carry_m"],
        release_t_s=meta["release_t_s"],
        base_flags={k: tuple(v) for k, v in meta["base_flags"].items()},
        held_base=None if meta["held_base"] is None else (meta["held_base"][0], tuple(meta["held_base"][1])),
        receptacle_name=meta["receptacle_name"],
    )
    refs = {k[4:]: z[k] for k in z.files if k.startswith("ref_")}
    return SimBundle(case, meta, refs)
