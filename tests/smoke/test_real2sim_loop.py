import json
from pathlib import Path

import pytest

from contracts.real2sim import HistoryEntry, QaSummary, Real2SimFeedback
from contracts.sim_params import SimParams
from rsi.history import append_entry, iter_dir, next_iter
from rsi.loop import load_state, run_loop
from rsi.types import IterResult
from skills.params.resolve import _deep_merge, params_hash

pytestmark = pytest.mark.smoke


class FakeRunner:
    def __init__(self, root: Path, scores: list[float]):
        self.root, self.scores, self.params, self.calls = root, scores, SimParams(), []

    def __call__(self, patch, rationale, parent) -> IterResult:
        self.calls.append((patch, parent))
        if patch:
            self.params = SimParams.model_validate(
                _deep_merge(json.loads(self.params.model_dump_json()), patch)
            )
        n = next_iter(self.root)
        score = self.scores[min(n, len(self.scores) - 1)]
        ok = score <= 0.35
        h = params_hash(self.params)
        entry = HistoryEntry(
            iter=n,
            parent=parent,
            mode="B",
            params_hash=h,
            plan_params_hash="p",
            success=ok,
            match_score=score,
        )
        append_entry(self.root, entry)
        fb = Real2SimFeedback(
            clip="c0", hand="sharpa", iter=n, params_hash=h, qa=QaSummary(success=ok), match_score=score
        )
        d = iter_dir(self.root, n)
        d.mkdir(parents=True)
        return IterResult(n, d, fb, entry, f"# iter {n} score {score}", self.params)


def _answer(relay: Path, state_root: Path, patch: dict | str, stop=False):
    rid = load_state(state_root).pending_rid
    text = patch if isinstance(patch, str) else json.dumps({"patch": patch, "rationale": "r", "stop": stop})
    (relay / "inbox" / f"{rid}.json").write_text(json.dumps({"answered_by": "copilot-relay", "text": text}))
    return rid


def test_loop_to_success(tmp_path):
    root, relay = tmp_path / "real2sim", tmp_path / "relay"
    runner = FakeRunner(root, [1.5, 1.0, 0.3])
    assert run_loop(root, relay, runner) == 75
    assert len(runner.calls) == 1
    first = load_state(root).pending_rid
    assert run_loop(root, relay, runner) == 75  # idempotent while pending
    assert len(runner.calls) == 1 and load_state(root).pending_rid == first
    assert len(list((relay / "outbox").iterdir())) == 1

    _answer(relay, root, {"contact": {"impratio": 5.0}})
    assert run_loop(root, relay, runner) == 75
    assert runner.calls[-1] == ({"contact": {"impratio": 5.0}}, 0)
    assert load_state(root).iter == 1 and load_state(root).best_iter == 1

    _answer(relay, root, {"contact": {"noslip_iterations": 9}})
    assert run_loop(root, relay, runner) == 0
    state = load_state(root)
    assert (state.status, state.outcome, state.best_iter, state.iter) == ("DONE", "SUCCESS", 2, 2)
    assert run_loop(root, relay, runner) == 0 and len(runner.calls) == 3


def test_invalid_answers_then_agent_error(tmp_path):
    root, relay = tmp_path / "real2sim", tmp_path / "relay"
    runner = FakeRunner(root, [1.5])
    run_loop(root, relay, runner)
    rids = {_answer(relay, root, "not json")}
    assert run_loop(root, relay, runner) == 75
    assert load_state(root).invalid_streak == 1
    rids.add(_answer(relay, root, {"bogus": {"x": 1}}))
    assert run_loop(root, relay, runner) == 75
    assert "bogus" in (relay / "outbox" / load_state(root).pending_rid / "prompt.md").read_text()
    rids.add(_answer(relay, root, {}))
    assert run_loop(root, relay, runner) == 0
    state = load_state(root)
    assert state.outcome == "AGENT_ERROR" and len(rids) == 3 and len(runner.calls) == 1


def test_patience_and_agent_stop(tmp_path):
    root, relay = tmp_path / "real2sim", tmp_path / "relay"
    runner = FakeRunner(root, [1.0, 2.0, 2.0])
    run_loop(root, relay, runner, patience=2)
    _answer(relay, root, {"contact": {"impratio": 5.0}})
    assert run_loop(root, relay, runner, patience=2) == 75
    _answer(relay, root, {"contact": {"impratio": 6.0}})
    assert run_loop(root, relay, runner, patience=2) == 0
    assert load_state(root).outcome == "PATIENCE"

    root2 = tmp_path / "r2"
    runner2 = FakeRunner(root2, [1.0])
    run_loop(root2, relay, runner2)
    _answer(relay, root2, {}, stop=True)
    assert run_loop(root2, relay, runner2) == 0
    assert load_state(root2).outcome == "AGENT_STOP"


def test_max_iters(tmp_path):
    root, relay = tmp_path / "real2sim", tmp_path / "relay"
    runner = FakeRunner(root, [2.0, 1.9])
    assert run_loop(root, relay, runner, max_iters=1) == 0
    assert load_state(root).outcome == "MAX_ITERS"
