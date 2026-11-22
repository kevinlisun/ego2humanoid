"""Robustness check of a real2sim iteration (spec section 12): start-offset grid plus seeded perturbations."""

import json
from pathlib import Path

import yaml

from contracts.sim_backend import SimBackend
from contracts.sim_params import SimParams
from rsi.backends import get_backend
from rsi.history import iter_dir
from rsi.runner import real2sim_dir

MIN_SEED_SUCCESSES = 3
SEEDS = (1, 2, 3, 4, 5)


def robust_check(clip_dir: Path, hand: str, iteration: int, backend: SimBackend | None = None) -> dict:
    backend = backend or get_backend()
    root = real2sim_dir(Path(clip_dir), hand)
    out = iter_dir(root, iteration)
    params = SimParams.model_validate(yaml.safe_load((out / "params.yaml").read_text()))
    session = backend.load(root)

    def run(start_xy, seed: int = 0) -> bool:
        rec = backend.rollout(session, params, start_xy, seed=seed, perturb=True)
        return bool(rec.pick_place["success"])

    grid = [run(xy) for xy in backend.start_grid(session, params)]
    seeds = [run(session.reference.best_offset_m, s) for s in SEEDS]
    report = {
        "iter": iteration,
        "backend": backend.name,
        "grid_success_fraction": sum(grid) / len(grid),
        "seed_successes": sum(seeds),
        "seeds": len(seeds),
        "passed": sum(seeds) >= MIN_SEED_SUCCESSES,
    }
    (out / "robust.json").write_text(json.dumps(report, indent=1))
    return report
