"""SimParams CLI: schema | resolve | diff | dump-defaults (Track RSI)."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from contracts.sim_params import SimParams
from skills.params.resolve import diff_params, needs_plan, params_hash, resolve_params

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "configs" / "sim_params" / "default.yaml"


def _dump(params: SimParams) -> str:
    return yaml.safe_dump(json.loads(params.model_dump_json()), sort_keys=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema")
    sub.add_parser("dump-defaults").add_argument("--write", action="store_true")
    r = sub.add_parser("resolve")
    r.add_argument("--params", action="append", default=[])
    r.add_argument("--set", action="append", default=[], dest="sets")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args(argv)

    if args.cmd == "schema":
        print(json.dumps(SimParams.model_json_schema(), indent=2))
    elif args.cmd == "dump-defaults":
        text = _dump(SimParams())
        if args.write:
            DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULTS_PATH.write_text(text)
        else:
            print(text, end="")
    elif args.cmd == "resolve":
        p = resolve_params(args.params, args.sets)
        print(_dump(p), end="")
        print(f"# params_hash: {params_hash(p)}", file=sys.stderr)
    else:
        a, b = (resolve_params([f], env={}) for f in (args.a, args.b))
        changes = diff_params(a, b)
        for c in changes:
            print(f"{c.path}: {c.old} -> {c.new}  [{'+'.join(sorted(c.affects))}]")
        print(f"needs_plan: {needs_plan(changes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
