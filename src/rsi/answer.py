"""Validation of a relay answer to a `real2sim_params` request (spec 11.3)."""

import json
import re
from typing import Any

from pydantic import ValidationError

from contracts.real2sim import RelayParamAnswer
from contracts.sim_params import SimParams
from rsi.schema import is_known_path, leaf_paths
from skills.params.resolve import _deep_merge, params_hash


class AnswerError(ValueError):
    """The answer cannot be applied; the message is appended to the next request."""


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _expand_dotted(patch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in patch.items():
        if isinstance(val, dict):
            val = _expand_dotted(val)
        node: Any = val
        for part in reversed(key.split(".")):
            node = {part: node}
        out = _deep_merge(out, node)
    return out


def validate_answer(
    text: str, current: SimParams, history_param_hashes: set[str], max_changes: int = 3
) -> tuple[RelayParamAnswer, SimParams]:
    body = text.strip()
    if m := _FENCE.match(body):
        body = m.group(1)
    try:
        raw = json.loads(body)
        answer = RelayParamAnswer.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as e:
        raise AnswerError(f"answer is not a valid {{patch, rationale, stop}} JSON object: {e}") from e
    patch = _expand_dotted(answer.patch)
    leaves = leaf_paths(patch)
    if answer.stop and not leaves:
        return answer, current
    if not leaves:
        raise AnswerError("patch is empty")
    if len(leaves) > max_changes:
        raise AnswerError(f"patch changes {len(leaves)} parameters, at most {max_changes} allowed")
    unknown = [p for p in leaves if not is_known_path(p)]
    if unknown:
        raise AnswerError(f"unknown parameter path(s): {', '.join(unknown)}")
    merged = _deep_merge(json.loads(current.model_dump_json()), patch)
    try:
        params = SimParams.model_validate(merged)
    except ValidationError as e:
        raise AnswerError(f"patch violates the SimParams constraints: {e}") from e
    if params_hash(params) == params_hash(current) or params_hash(params) in history_param_hashes:
        raise AnswerError(
            "resulting parameters are identical to an earlier iteration; propose a new hypothesis"
        )
    return RelayParamAnswer(patch=patch, rationale=answer.rationale, stop=answer.stop), params
