"""Solve and store the Sharpa pinch joint sets (configs/grasp/sharpa_pinch_v1.json)."""

from __future__ import annotations

import argparse
import json

from stages.locomotion.tabletop import Scenario
from stages.retargeting.sharpa_pinch import PINCH_POINT_WRIST, POSES_PATH, solve_pinch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=float, default=0.03, help="object width between the pads (m)")
    args = ap.parse_args()
    m, _ = Scenario.load().build("sharpa")
    stages = solve_pinch(m, args.width)
    cfg = {
        "object_width_m": args.width,
        "pinch_point_wrist_m": PINCH_POINT_WRIST.tolist(),
        "stages": stages,
    }
    POSES_PATH.write_text(json.dumps(cfg, indent=1) + "\n")
    for k, v in stages.items():
        print(k, "gap", v["gap_m"], "err", v["tip_err_m"])


if __name__ == "__main__":
    main()
