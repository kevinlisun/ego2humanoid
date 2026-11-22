"""Mode B request: `prompt.md` and `request.json` for one `real2sim_params` relay request."""

import hashlib
import json
from typing import Any

from contracts.real2sim import HistoryEntry, Real2SimFeedback
from contracts.sim_params import SimParams
from rsi.history import format_table
from rsi.schema import Leaf, allowed_paths, current_value, normalize_path, schema_leaves
from skills.params.resolve import params_hash

MAX_CHANGES = 3
HISTORY_ROWS = 10

ROLE = """\
You tune the physics parameters (`SimParams`) of a MuJoCo replay of one human manipulation clip so
that the simulated robot reproduces the human's pick-and-place. You see the spatio-temporal feedback
of the last iteration and answer with a small parameter patch."""

RULES = """\
- Change one hypothesis per iteration (1-3 keys); prefer `sim`-only keys first (cheap re-run).
- If the last iteration was worse than an earlier one, revert toward the best iteration's values.
- Never propose code, QA-threshold or perception changes; a needed structural change goes into
  `rationale` and `stop` stays false unless no parameter can help.
- Use only paths from the schema table (object overrides are `objects.<selector>.<field>`)."""

ANSWER_FORMAT = """\
Reply with exactly one JSON object (no prose around it) and write it as the `text` of
`inbox/<rid>.json` = `{"answered_by": "copilot-relay", "text": "<that JSON>"}`:

```json
{"patch": {"objects": {"held": {"friction": 1.2}}}, "rationale": "one line", "stop": false}
```

`patch` is a nested partial `SimParams` with at most %d leaf values (lists such as `solref` count as
one leaf); `stop: true` with an empty patch ends the loop."""


def feedback_hash(feedback: Real2SimFeedback, feedback_md: str) -> str:
    blob = feedback.model_dump_json() + "\x00" + feedback_md
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def request_rid(clip: str, hand: str, it: int, p_hash: str, fb_hash: str, error: str = "") -> str:
    h = hashlib.sha256("\x00".join([clip, hand, str(it), p_hash, fb_hash, error]).encode())
    return h.hexdigest()[:16]


def _select_leaves(feedback: Real2SimFeedback) -> list[Leaf]:
    leaves = list(schema_leaves())
    hinted = {normalize_path(h.param) for h in feedback.hints}
    hinted &= {leaf.path for leaf in leaves}
    if not hinted:
        return leaves
    groups = {p.rsplit(".", 1)[0] for p in hinted}
    return [leaf for leaf in leaves if leaf.path.rsplit(".", 1)[0] in groups]


def _cell(value: Any) -> str:
    return "-" if value is None else json.dumps(value)


def schema_table(feedback: Real2SimFeedback, params: SimParams) -> str:
    rows = ["| path | current | unit | range | affects | doc |", "|---|---|---|---|---|---|"]
    for leaf in _select_leaves(feedback):
        if leaf.path.startswith("objects.*"):
            current = "per selector"
        else:
            current = _cell(current_value(params, leaf.path))
        affects = "+".join(leaf.affects)
        rows.append(f"| `{leaf.path}` | {current} | {leaf.unit} | {leaf.range} | {affects} | {leaf.doc} |")
    return "\n".join(rows)


def build_request(
    clip: str,
    hand: str,
    feedback: Real2SimFeedback,
    feedback_md: str,
    params: SimParams,
    history: list[HistoryEntry],
    *,
    bundle_plan_hash: str = "",
    error: str = "",
    max_changes: int = MAX_CHANGES,
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(rid, prompt_md, request_json)``; a non-empty ``error`` is appended and changes the rid."""
    p_hash = params_hash(params)
    rid = request_rid(clip, hand, feedback.iter, p_hash, feedback_hash(feedback, feedback_md), error)
    sections = [
        f"# real2sim_params request {rid} - clip {clip} - {hand} - iter {feedback.iter}",
        ROLE,
        "## Rules\n" + RULES,
        "## Feedback of the last iteration\n" + feedback_md.strip(),
        "## Parameter schema (hinted parameters and their groups)\n" + schema_table(feedback, params),
        f"## History (last {HISTORY_ROWS})\n" + format_table(history[-HISTORY_ROWS:]),
        "## Answer format\n" + ANSWER_FORMAT % max_changes,
    ]
    if error:
        sections.append(f"## Your previous answer was rejected\n{error.strip()}\nFix it and answer again.")
    meta = {
        "id": rid,
        "kind": "real2sim_params",
        "clip": clip,
        "hand": hand,
        "iter": feedback.iter,
        "params_hash": p_hash,
        "bundle_plan_hash": bundle_plan_hash,
        "allowed_paths": allowed_paths(),
        "max_changes": max_changes,
    }
    return rid, "\n\n".join(sections) + "\n", meta
