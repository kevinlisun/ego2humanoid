import json

import pytest

from contracts.real2sim import HistoryEntry, QaSummary, Real2SimFeedback
from contracts.sim_params import SimParams
from rsi.answer import AnswerError, validate_answer
from rsi.history import append_entry, best_entry, format_table, iter_dir, next_iter, read_history
from rsi.prompt import build_request
from skills.params.resolve import params_hash
from stages.reconstruction.scene.vlm_relay import read_answer, relay_status, submit_request


def _entry(i, score, success=False, **kw):
    return HistoryEntry(
        iter=i, params_hash=f"h{i}", plan_params_hash="p", success=success, match_score=score, **kw
    )


def test_submit_and_read_answer(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    d = submit_request(tmp_path / "relay", "r1", "hi", [b"png", img], {"kind": "k"})
    assert (d / "prompt.md").read_text() == "hi"
    assert sorted(p.name for p in (d / "images").iterdir()) == ["00.png", "01.jpg"]
    assert json.loads((d / "request.json").read_text())["id"] == "r1"
    assert relay_status(tmp_path / "relay").pending == ["r1"]
    assert read_answer(tmp_path / "relay", "r1") is None
    (tmp_path / "relay/inbox/r1.json").write_text(json.dumps({"answered_by": "copilot-relay", "text": "t"}))
    assert read_answer(tmp_path / "relay", "r1")["text"] == "t"


def test_history_roundtrip_and_best(tmp_path):
    assert next_iter(tmp_path) == 0
    for e in (_entry(0, 1.5), _entry(1, 2.0, success=True), _entry(2, 0.4), _entry(3, 1.0, success=True)):
        append_entry(tmp_path, e)
    hist = read_history(tmp_path)
    assert next_iter(tmp_path) == 4
    assert best_entry(hist).iter == 3
    assert best_entry([]) is None
    assert iter_dir(tmp_path, 7).name == "iter_007"
    assert "| 3 | - | rollout |" in format_table(hist)


def _ok(patch, **kw):
    return validate_answer(json.dumps({"patch": patch, "rationale": "r", **kw}), SimParams(), set())


def test_valid_answer_merges():
    ans, params = _ok({"objects": {"held": {"friction": 1.2}}, "contact": {"noslip_iterations": 9}})
    assert params.objects["held"].friction == 1.2 and params.contact.noslip_iterations == 9
    assert ans.rationale == "r"


def test_fenced_and_dotted_answer():
    text = '```json\n{"patch": {"contact.impratio": 5.0}}\n```'
    _, params = validate_answer(text, SimParams(), set())
    assert params.contact.impratio == 5.0


@pytest.mark.parametrize(
    "patch",
    [
        {},
        {"contact": {"nope": 1}},
        {"contact": {"impratio": 1000}},
        {"contact": {"impratio": 5, "torsion_m": 0.02, "roll_m": 0.001, "noslip_iterations": 7}},
        {"contact": {"impratio": 10.0}},
    ],
)
def test_invalid_answers(patch):
    with pytest.raises(AnswerError):
        _ok(patch)


def test_not_json_and_repeat_of_history():
    with pytest.raises(AnswerError):
        validate_answer("sure!", SimParams(), set())
    target = SimParams.model_validate({"contact": {"impratio": 5.0}})
    with pytest.raises(AnswerError, match="earlier iteration"):
        validate_answer('{"patch": {"contact": {"impratio": 5.0}}}', SimParams(), {params_hash(target)})


def test_stop_with_empty_patch():
    ans, params = validate_answer('{"patch": {}, "stop": true}', SimParams(), set())
    assert ans.stop and params == SimParams()


def _feedback(hints=()):
    return Real2SimFeedback(
        clip="c0",
        hand="sharpa",
        iter=2,
        params_hash="x",
        qa=QaSummary(success=False),
        match_score=1.0,
        hints=hints,
    )


def test_build_request():
    from contracts.real2sim import FeedbackHint

    fb = _feedback((FeedbackHint(pattern="SLIP", param="objects.held.friction"),))
    rid, prompt, meta = build_request("c0", "sharpa", fb, "# md text", SimParams(), [_entry(0, 1.0)])
    assert meta["id"] == rid
    assert meta["kind"] == "real2sim_params" and meta["max_changes"] == 3
    assert "contact.impratio" in meta["allowed_paths"] and "objects.*.friction" in meta["allowed_paths"]
    assert "# md text" in prompt and "`objects.*.friction`" in prompt and "`contact.impratio`" not in prompt
    rid2, prompt2, _ = build_request("c0", "sharpa", fb, "# md text", SimParams(), [], error="bad key")
    assert rid2 != rid and "bad key" in prompt2
