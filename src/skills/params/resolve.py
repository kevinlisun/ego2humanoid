"""Resolution, hashing and diffing of `SimParams` (Track RSI, RSI.3)."""

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from contracts.sim_params import ObjectPhysics, SimParams
from skills.params.legacy_env import legacy_overlay

ENV_OVERLAY = "E2H_SIM_PARAMS"
ACCEPTED_FILE = "accepted.yaml"


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val  # lists and scalars replace
    return out


def _load_overlay(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: overlay must be a mapping")
    return data


def parse_set(pair: str) -> dict:
    """`a.b=c` -> {"a": {"b": c}} with the value parsed as YAML (numbers, lists, bools)."""
    path, sep, raw = pair.partition("=")
    if not sep or not path:
        raise ValueError(f"expected a.b=value, got {pair!r}")
    node: Any = yaml.safe_load(raw)
    for key in reversed(path.split(".")):
        node = {key: node}
    return node


def accepted_path(clip_dir: str | Path, hand: str) -> Path:
    """Parameters promoted by `real2sim_eval.py accept` (they passed the robustness check)."""
    return Path(clip_dir) / f"real2sim_{hand}" / ACCEPTED_FILE


def resolve_params(
    overlays: Iterable[str | Path] = (),
    sets: Iterable[str] = (),
    env: dict | None = None,
    accepted: str | Path | None = None,
) -> SimParams:
    """Defaults -> accepted -> $E2H_SIM_PARAMS -> legacy `E2H_*` vars -> overlay files -> `--set` pairs.

    `accepted` is the clip's `accepted.yaml` (ignored if absent); later layers win, so the env overlay
    and everything after it can still override a promoted parameter set.
    """
    env = os.environ if env is None else env
    data: dict = _canonical(SimParams())
    if accepted is not None and Path(accepted).exists():
        data = _deep_merge(data, _load_overlay(accepted))
    if env.get(ENV_OVERLAY):
        data = _deep_merge(data, _load_overlay(env[ENV_OVERLAY]))
    data = _deep_merge(data, legacy_overlay(env))
    for path in overlays:
        data = _deep_merge(data, _load_overlay(path))
    for pair in sets:
        data = _deep_merge(data, parse_set(pair))
    return SimParams.model_validate(data)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, val in node.items():
            out.update(_flatten(val, f"{prefix}.{key}" if prefix else key))
        return out or {prefix: {}}
    return {prefix: node}


def _canonical(params: SimParams) -> dict:
    return json.loads(params.model_dump_json())


def params_hash(params: SimParams) -> str:
    blob = json.dumps(_canonical(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def plan_params_hash(params: SimParams) -> str:
    """Hash of the plan-affecting leaves only: equal hashes mean a cached plan is still valid."""
    flat = _flatten(_canonical(params))
    plan = {k: v for k, v in flat.items() if "plan" in _affects(k)}
    blob = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _affects(path: str) -> frozenset[str]:
    """`affects` metadata of a dotted leaf path; anything unresolvable is conservatively `plan`."""
    model: Any = SimParams
    parts = path.split(".")
    if parts[0] == "objects":  # objects.<selector>.<field>
        parts = parts[2:] if len(parts) >= 3 else []
        model = ObjectPhysics
    for part in parts:
        if not (isinstance(model, type) and issubclass(model, BaseModel)) or part not in model.model_fields:
            break
        info = model.model_fields[part]
        extra = info.json_schema_extra
        if isinstance(extra, dict) and "affects" in extra:
            return frozenset(extra["affects"])
        model = info.annotation
    else:
        return frozenset({"sim"})
    return frozenset({"sim", "plan"})


@dataclass(frozen=True)
class ParamChange:
    path: str
    old: Any
    new: Any
    affects: frozenset[str]


def diff_params(a: SimParams, b: SimParams) -> list[ParamChange]:
    fa, fb = _flatten(_canonical(a)), _flatten(_canonical(b))
    changes = []
    for path in sorted(set(fa) | set(fb)):
        if fa.get(path) != fb.get(path):
            changes.append(ParamChange(path, fa.get(path), fb.get(path), _affects(path)))
    return changes


def needs_plan(changes: Iterable[ParamChange]) -> bool:
    return any("plan" in c.affects for c in changes)
