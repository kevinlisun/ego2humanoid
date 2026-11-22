"""Export a clip's simulated scene to a neutral folder for the Blender renderer (Track BT, BT.10).

    uv run --extra video --extra hands --with mujoco python scripts/export_twin_scene.py <clip_dir> [tag]

``tag`` selects ``review/trajectory_sharpa<tag>.npz`` (default ``_bt``). Writes ``review/twin_scene/``:
``scene.json`` (table, camera, object list), ``<name>.npz`` (mesh in the body frame + texture) and
``traj.npz`` (held-object, wrist and fingertip trajectories in the sim frame).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np


def _dump(o, out: Path, kind: str, name: str) -> dict:
    arrays = {
        "vertices": np.asarray(o.mesh_vertices, np.float32),
        "faces": np.asarray(o.mesh_faces, np.int32),
    }
    if o.mesh_uv is not None and o.mesh_texture is not None:
        arrays |= {"uv": np.asarray(o.mesh_uv, np.float32), "texture": np.asarray(o.mesh_texture, np.uint8)}
    np.savez_compressed(out / f"{name}.npz", **arrays)
    return {
        "name": name,
        "body": o.name,
        "kind": kind,
        "pos_m": [float(x) for x in o.pos_m],
        "rgba": [float(x) for x in o.rgba],
        "half_size_m": [float(x) for x in o.half_size_m],
        "static": bool(o.static),
        "mesh": f"{name}.npz",
    }


def main() -> None:
    clip = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "_bt"
    out = clip / "review" / "twin_scene"
    out.mkdir(parents=True, exist_ok=True)
    scene = pickle.load(open(clip / "scene_sim.pkl", "rb"))
    traj = np.load(clip / "review" / f"trajectory_sharpa{tag}.npz")
    objects = [_dump(scene["pbox"], out, "held", "held_object")]
    objects += [_dump(o, out, "scene", o.name) for o in scene["extra"]]
    intr = json.loads((clip / "intrinsics_est.json").read_text())
    meta = {
        "table": {k: np.asarray(v).tolist() for k, v in scene["table_kw"].items()},
        "camera_T_sim": traj["camera_T_sim"].tolist(),
        "focal_px": intr["focal_px"],
        "image_wh": [intr["width"], intr["height"]],
        "objects": objects,
        "tag": tag,
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(
        out / "traj.npz",
        **{k: traj[k] for k in ("wrist_m", "fingertip_m", "object_pos_m", "object_rot")},
    )
    print(f"exported {len(objects)} objects, {len(traj['object_pos_m'])} frames -> {out}")


if __name__ == "__main__":
    main()
