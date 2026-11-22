"""Simulator decoupling of the real2sim loop: import boundary, registry, and a fake backend."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from contracts.sim_backend import ClipReference, RolloutRecord, SimBackend, SimSession
from contracts.sim_params import SimParams
from rsi import robust
from rsi.backends import REGISTRY, get_backend

SRC = Path(__file__).resolve().parents[2] / "src"
SIM_FREE = [
    *(SRC / "rsi").glob("*.py"),
    *(SRC / "skills" / "params").glob("resolve.py"),
    *(SRC / "skills").glob("*.py"),
    SRC / "contracts" / "sim_backend.py",
]
# body_state records a live simulator for the render scripts and is not part of the loop.
ALLOWED = {"body_state.py"}


def _imports(path: Path) -> list[str]:
    out = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append(("." * n.level) + (n.module or ""))
    return out


@pytest.mark.parametrize("path", SIM_FREE, ids=lambda p: p.name)
def test_loop_modules_do_not_import_a_simulator(path):
    if path.name in ALLOWED:
        pytest.skip("simulator-side recorder")
    bad = [m for m in _imports(path) if m.split(".")[0] == "mujoco" or "replay" in m.split(".")]
    assert not bad, f"{path.name} imports simulator code: {bad}"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown sim backend"):
        get_backend("nope")


def test_mujoco_is_registered_and_satisfies_protocol():
    assert "mujoco" in REGISTRY
    assert isinstance(get_backend("mujoco"), SimBackend)


class FakeBackend:
    name = "fake"

    def __init__(self, successes):
        self.successes = iter(successes)
        self.calls = []

    def load(self, root):
        return SimSession(ClipReference(meta={}, best_offset_m=(0.1, 0.2)), None)

    def plan(self, clip, hand, params_file):
        raise AssertionError("robust never re-plans")

    def rollout(self, session, params, start_xy_m, *, seed=0, perturb=False, video_path=None):
        self.calls.append((tuple(start_xy_m), seed, perturb))
        return RolloutRecord({}, {}, ["held"], [False], {"success": next(self.successes)})

    def search_start(self, session, params):
        return (0.0, 0.0)

    def start_grid(self, session, params):
        return [(0.0, 0.0), (0.01, 0.0)]


def test_robust_check_runs_on_any_backend(tmp_path):
    root = tmp_path / "real2sim_sharpa"
    (root / "iter_000").mkdir(parents=True)
    params = SimParams()
    (root / "iter_000" / "params.yaml").write_text(yaml.safe_dump(json.loads(params.model_dump_json())))
    fake = FakeBackend([True, False] + [True, True, True, False, False])
    rep = robust.robust_check(tmp_path, "sharpa", 0, fake)
    assert rep["backend"] == "fake"
    assert rep["grid_success_fraction"] == 0.5
    assert rep["seed_successes"] == 3 and rep["passed"]
    assert [c[1] for c in fake.calls] == [0, 0, 1, 2, 3, 4, 5]
    assert all(c[2] for c in fake.calls)
    assert fake.calls[2][0] == (0.1, 0.2)
    assert np.isclose(rep["grid_success_fraction"], 0.5)
