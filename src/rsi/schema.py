"""`SimParams` leaf catalogue (path, unit, range, affects, doc) read from its JSON schema."""

import re
from dataclasses import dataclass
from functools import cache
from typing import Any

from contracts.sim_params import SimParams

_INDEX = re.compile(r"\[\d+\]$")


@dataclass(frozen=True)
class Leaf:
    path: str  # dotted; object selectors appear as `objects.*.<field>`
    unit: str
    affects: tuple[str, ...]
    doc: str
    range: str


def _range_text(node: dict[str, Any]) -> str:
    if "enum" in node:
        return "{" + ", ".join(str(v) for v in node["enum"]) + "}"
    if node.get("type") == "boolean" or any(o.get("type") == "boolean" for o in node.get("anyOf", [])):
        return "{true, false}"
    lo = node.get("minimum", node.get("exclusiveMinimum"))
    hi = node.get("maximum", node.get("exclusiveMaximum"))
    if lo is None and hi is None:
        for opt in node.get("anyOf", []):
            if "minimum" in opt or "maximum" in opt:
                return _range_text(opt)
        return "see doc"
    return f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"


def _resolve(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in node:
        return defs[node["$ref"].rsplit("/", 1)[-1]]
    return node


def _walk(node: dict[str, Any], defs: dict[str, Any], prefix: str) -> list[Leaf]:
    node = _resolve(node, defs)
    if "properties" in node:
        out: list[Leaf] = []
        for name, child in node["properties"].items():
            if name == "version":
                continue
            out += _walk(child, defs, f"{prefix}.{name}" if prefix else name)
        return out
    if isinstance(node.get("additionalProperties"), dict):
        return _walk(node["additionalProperties"], defs, f"{prefix}.*")
    return [
        Leaf(
            prefix,
            node.get("unit", ""),
            tuple(node.get("affects", ("sim",))),
            node.get("doc", ""),
            _range_text(node),
        )
    ]


@cache
def schema_leaves() -> tuple[Leaf, ...]:
    schema = SimParams.model_json_schema()
    return tuple(_walk(schema, schema.get("$defs", {}), ""))


def allowed_paths() -> list[str]:
    return [leaf.path for leaf in schema_leaves()]


def normalize_path(path: str) -> str:
    """Drop a trailing `[i]` index and map `objects.<selector>.x` to `objects.*.x`."""
    path = _INDEX.sub("", path)
    parts = path.split(".")
    if parts[0] == "objects" and len(parts) >= 3:
        parts[1] = "*"
    return ".".join(parts)


def is_known_path(path: str) -> bool:
    return normalize_path(path) in set(allowed_paths())


def leaf_paths(patch: dict[str, Any], prefix: str = "") -> list[str]:
    """Dotted leaf paths of a nested patch; lists and scalars are leaves."""
    out: list[str] = []
    for key, val in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict) and val:
            out += leaf_paths(val, path)
        else:
            out.append(path)
    return out


def current_value(params: SimParams, path: str) -> Any:
    node: Any = params.model_dump(mode="json")
    for part in _INDEX.sub("", path).split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node
