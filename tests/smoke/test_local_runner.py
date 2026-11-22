import pytest

from orchestration.dag import Dag
from orchestration.local_runner import run_dag, synthetic_replay_dag


def test_dag_order_and_unknown_dep():
    d = Dag()
    d.add("b", lambda o, u: None, ("a",))
    with pytest.raises(ValueError):
        d.order()
    d.add("a", lambda o, u: None)
    assert d.order() == ["a", "b"]


def test_runner_stops_on_failure(tmp_path):
    ran = []
    d = Dag()
    d.add("a", lambda o, u: (_ for _ in ()).throw(RuntimeError("boom")))
    d.add("b", lambda o, u: ran.append("b"), ("a",))
    with pytest.raises(RuntimeError):
        run_dag(d, tmp_path)
    assert ran == []


def test_synthetic_dag_end_to_end(tmp_path):
    outs = run_dag(synthetic_replay_dag(), tmp_path)
    assert (outs["replay"] / "qa_report.json").exists()
    assert (outs["replay"] / "render.mp4").exists()
