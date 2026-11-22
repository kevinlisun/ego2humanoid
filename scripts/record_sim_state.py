"""Replay a clip's cached retarget through MuJoCo and write the Blender link files (Track BT, BT.16).

    uv run --extra video --extra sim --extra hands --extra objects python scripts/record_sim_state.py <clip> [tag]

Uses ``scene_sim.pkl`` and ``retarget_rgb_sharpa<tag>.npz`` (no perception, no render, ~1 min). Writes
``review/sim_state_sharpa<tag>.npz`` and ``review/sim_model/{sim_model.json,sim_model_meshes.npz}``.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

from skills.body_state import BodyStateRecorder, export_sim_model
from skills.params.resolve import resolve_params
from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.manipulation.rollout import run_rollout


def main() -> None:
    clip = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "_bt"
    scene = pickle.load(open(clip / "scene_sim.pkl", "rb"))
    ret = np.load(clip / f"retarget_rgb_sharpa{tag}.npz")
    cfg = SceneConfig.from_params(
        resolve_params(),
        T_sim_robot_base=scene["base"],
        **scene["table_kw"],
        objects=(scene["pbox"], *scene["extra"]),
        hand="sharpa",
    )
    rec = BodyStateRecorder()
    res = run_rollout(cfg, ret["timestamps_s"], ret["q_target_rad"], on_record=rec)
    review = clip / "review"
    out = rec.save(review / f"sim_state_sharpa{tag}.npz")
    export_sim_model(build_scene(cfg), review / "sim_model")
    print(f"{len(rec.t)} records, {res.contact_steps_with_object} contact steps -> {out}")


if __name__ == "__main__":
    main()
