"""Run the loco-manipulation episode (plan -> walk -> arrive -> grasp/lift/place) on scenario variants.

    uv run python scripts/loco_manip_episode.py --list
    uv run python scripts/loco_manip_episode.py --clip 0 --clip 1 --clip 10 --clip 100
    uv run python scripts/loco_manip_episode.py --case golden --case obj_left [--jobs 4] [--no-video]

Clips run the retargeted wrist/fingers (L.17) and write to ``data/derived/egodex/basic_pick_place_<n>/``
(``loco_manip_result.json`` and ``review/loco_manip.mp4``).
Scenario cases write ``data/derived/egodex/loco_manip_<case>/{result.json,review/<case>.mp4}``.
"""

from __future__ import annotations

import argparse
import copy
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT_ROOT = Path("data/derived/egodex")
PLACE_TOL_M = 0.05

# Scenario overrides per case (object/goal are offsets on the table, robot start is x, y, yaw).
CASES: dict[str, dict] = {
    "golden": {},
    "start_left": {"robot_start_xyyaw": [0.0, 0.6, 0.0]},
    "start_right": {"robot_start_xyyaw": [0.0, -0.6, 0.0]},
    "start_yawed": {"robot_start_xyyaw": [0.0, 0.0, 0.6]},
    "start_far": {"robot_start_xyyaw": [-1.0, 0.0, 0.0]},
    "obj_left": {"object": {"xy_on_table": [-0.2, 0.12]}, "goal_zone": {"xy_on_table": [-0.2, 0.0]}},
    "obj_right": {"object": {"xy_on_table": [-0.2, -0.12]}, "goal_zone": {"xy_on_table": [-0.2, -0.24]}},
    "obj_deep": {"object": {"xy_on_table": [-0.1, 0.0]}, "goal_zone": {"xy_on_table": [-0.1, -0.12]}},
    "goal_far": {"goal_zone": {"xy_on_table": [-0.2, -0.25]}},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) else v
    return out


CLIP_ROBOT_RADIUS_M = 0.17  # the torso is above the table top, so the pelvis may stand closer to its edge
CLIP_VIS_MIN = (0.1, 0.0)  # the clip wrist poses leave little freedom, so the head-camera gate is relaxed


def run_case(name: str, video: bool) -> dict:
    warnings.filterwarnings("ignore")
    from contracts.locomanip import LocoManipFailure
    from stages.locomotion.clip_task import load_clip
    from stages.locomotion.episode import run_episode
    from stages.locomotion.plan.golden import golden_task
    from stages.locomotion.tabletop import Scenario

    clip = name.startswith("clip:")
    if clip:
        cc = load_clip(name[5:])
        sc, out_dir, stem = cc.scenario, OUT_ROOT / cc.clip, "loco_manip"
    else:
        sc, out_dir, stem = (
            Scenario(_merge(Scenario.load().cfg, CASES[name])),
            OUT_ROOT / f"loco_manip_{name}",
            name,
        )
    mp4 = str(out_dir / "review" / f"{stem}.mp4") if video else None
    row: dict = {"case": name}
    try:
        if clip:
            for vis_min in (
                CLIP_VIS_MIN,
                (0.0, 0.0),
            ):  # fall back to no visibility gate if nothing is feasible
                r = run_episode(
                    sc, cc.task, cc.side, video=mp4, poses=cc.poses, extra_stances=cc.native_stances,
                    plan_kw={"hand": "sharpa", "hand_table_ok": True, "vis_min": vis_min, "arrival_slack_m": 0.04,
                             "robot_radius_m": CLIP_ROBOT_RADIUS_M},
                )  # fmt: skip
                if r.get("failure") != LocoManipFailure.NO_FEASIBLE_STATIC_MANIPULATION_POSE:
                    break
        else:
            r = run_episode(sc, golden_task(sc), video=mp4)
    except Exception as e:  # keep the batch going; the failure is part of the report
        r = {"ok": False, "failure": f"EXCEPTION {type(e).__name__}: {e}"}
    obj = r.get("object_final_xyz")
    err = None if obj is None else float(np.linalg.norm(np.asarray(obj)[:2] - sc.goal_xy()))
    arrival = r.get("arrival")
    row.update(
        ok=bool(r.get("ok")),
        placed=err is not None and err < PLACE_TOL_M,
        lifted_m=r.get("object_max_lift_m"),
        place_err_m=err,
        failure=str(r.get("failure")) if r.get("failure") else None,
        arrived=getattr(arrival, "arrived", None),
        visible_fraction=r.get("visible_fraction_measured"),
        min_pelvis_z=r.get("min_pelvis_z"),
        video=mp4,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("loco_manip_result.json" if clip else "result.json")).write_text(
        json.dumps(row, indent=1, default=str)
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", help="case name (default: all scenario cases)")
    ap.add_argument("--clip", action="append", help="EgoDex clip number or id (e.g. 0, 100)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CASES))
        return
    names = [f"clip:{c}" for c in a.clip or []] + (a.case or ([] if a.clip else list(CASES)))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run_case, names, [not a.no_video] * len(names)))
    print(f"{'case':14} {'ok':5} {'placed':7} {'err_m':7} {'vis':5} failure")
    for r in rows:
        e = "-" if r["place_err_m"] is None else f"{r['place_err_m']:.3f}"
        v = "-" if r["visible_fraction"] is None else f"{r['visible_fraction']:.2f}"
        print(f"{r['case']:14} {r['ok']!s:5} {r['placed']!s:7} {e:7} {v:5} {r['failure'] or ''}")


if __name__ == "__main__":
    main()
