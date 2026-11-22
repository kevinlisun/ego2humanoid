"""Mode B state machine (spec 11.1): evaluate -> request -> wait for the relay answer -> repeat.

State lives in ``<real2sim_dir>/loop_state.json``; the last evaluated result needed to validate an
answer or re-ask lives in ``loop_ctx.json``. Re-running is idempotent: a request id is submitted once.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from contracts.real2sim import Real2SimFeedback
from contracts.sim_params import SimParams
from rsi.answer import AnswerError, validate_answer
from rsi.history import best_entry, read_history
from rsi.prompt import build_request
from rsi.types import IterResult
from stages.reconstruction.scene.vlm_relay import (
    PENDING_RELAY_EXIT_CODE,
    read_answer,
    relay_status,
    submit_request,
)

Evaluate = Callable[[dict | None, str, int | None], IterResult]

SUCCESS_SCORE = 0.35
MAX_INVALID = 3
STATE_FILE = "loop_state.json"
CTX_FILE = "loop_ctx.json"


@dataclass
class LoopState:
    iter: int = -1
    status: Literal["RUN", "WAIT", "DONE"] = "RUN"
    best_iter: int | None = None
    pending_rid: str | None = None
    invalid_streak: int = 0
    outcome: str | None = None  # SUCCESS | PATIENCE | MAX_ITERS | AGENT_STOP | AGENT_ERROR
    next: dict[str, Any] | None = field(default=None)  # validated answer awaiting evaluation


def load_state(root: Path) -> LoopState:
    path = root / STATE_FILE
    return LoopState(**json.loads(path.read_text())) if path.exists() else LoopState()


def save_state(root: Path, state: LoopState) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / STATE_FILE).write_text(json.dumps(asdict(state), indent=2))


def _save_ctx(root: Path, res: IterResult) -> None:
    ctx = {
        "params": json.loads(res.params.model_dump_json()),
        "feedback": json.loads(res.feedback.model_dump_json()),
        "feedback_md": res.feedback_md,
        "plan_params_hash": res.entry.plan_params_hash,
    }
    (root / CTX_FILE).write_text(json.dumps(ctx))


def _load_ctx(root: Path) -> tuple[SimParams, Real2SimFeedback, str, str]:
    ctx = json.loads((root / CTX_FILE).read_text())
    return (
        SimParams.model_validate(ctx["params"]),
        Real2SimFeedback.model_validate(ctx["feedback"]),
        ctx["feedback_md"],
        ctx["plan_params_hash"],
    )


def _stop_outcome(
    res: IterResult, best_iter: int, max_iters: int, patience: int, success_score: float
) -> str | None:
    if res.feedback.qa.success and res.feedback.match_score <= success_score:
        return "SUCCESS"
    if res.iter - best_iter >= patience:
        return "PATIENCE"
    if res.iter + 1 >= max_iters:
        return "MAX_ITERS"
    return None


def _submit(root: Path, relay_dir: Path, state: LoopState, hand: str, error: str, max_changes: int) -> None:
    params, feedback, md, plan_hash = _load_ctx(root)
    rid, prompt, meta = build_request(
        feedback.clip,
        hand,
        feedback,
        md,
        params,
        read_history(root),
        bundle_plan_hash=plan_hash,
        error=error,
        max_changes=max_changes,
    )
    already = rid in relay_status(relay_dir).pending + relay_status(relay_dir).answered
    if not already:
        submit_request(relay_dir, rid, prompt, [], meta)
    state.status, state.pending_rid = "WAIT", rid


def _summary(root: Path, state: LoopState) -> str:
    best = best_entry(read_history(root))
    head = f"real2sim loop DONE outcome={state.outcome} iters={state.iter + 1} best_iter={state.best_iter}"
    if best is None:
        return head
    return f"{head} best_score={best.match_score:.3f} success={best.success}"


def run_loop(
    real2sim_dir: str | Path,
    relay_dir: str | Path,
    evaluate: Evaluate,
    *,
    hand: str = "sharpa",
    max_iters: int = 20,
    patience: int = 5,
    success_score: float = SUCCESS_SCORE,
    max_changes: int = 3,
) -> int:
    """Advance the loop as far as possible; return 0 when DONE and 75 while a relay answer is pending."""
    root, relay = Path(real2sim_dir), Path(relay_dir)
    state = load_state(root)
    while True:
        if state.status == "DONE":
            print(_summary(root, state))
            return 0
        if state.status == "RUN":
            nxt = state.next or {"patch": None, "rationale": "", "parent": None}
            res = evaluate(nxt["patch"], nxt["rationale"], nxt["parent"])
            _save_ctx(root, res)
            entries = read_history(root)
            if all(e.iter != res.iter for e in entries):
                entries.append(res.entry)
            best = best_entry(entries)
            state.iter, state.best_iter, state.next, state.pending_rid = (
                res.iter,
                best.iter if best else res.iter,
                None,
                None,
            )
            state.outcome = _stop_outcome(res, state.best_iter, max_iters, patience, success_score)
            if state.outcome:
                state.status = "DONE"
            else:
                _submit(root, relay, state, hand, "", max_changes)
            save_state(root, state)
            continue
        assert state.pending_rid is not None
        try:
            raw = read_answer(relay, state.pending_rid)
        except ValueError as e:
            raw, bad = {}, f"answer file is not valid JSON: {e}"
        else:
            bad = ""
        if raw is None:
            print(f"pending relay request {state.pending_rid}")
            return PENDING_RELAY_EXIT_CODE
        params, _, _, _ = _load_ctx(root)
        hashes = {e.params_hash for e in read_history(root)}
        if not bad:
            try:
                answer, _ = validate_answer(str(raw.get("text", "")), params, hashes, max_changes)
            except AnswerError as e:
                bad = str(e)
        if bad:
            state.invalid_streak += 1
            if state.invalid_streak >= MAX_INVALID:
                state.status, state.outcome, state.pending_rid = "DONE", "AGENT_ERROR", None
            else:
                _submit(root, relay, state, hand, bad, max_changes)
        else:
            state.invalid_streak = 0
            if answer.stop and not answer.patch:
                state.status, state.outcome, state.pending_rid = "DONE", "AGENT_STOP", None
            else:
                state.status, state.pending_rid = "RUN", None
                state.next = {"patch": answer.patch, "rationale": answer.rationale, "parent": state.iter}
        save_state(root, state)
