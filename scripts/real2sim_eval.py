"""Agentic real2sim CLI (Track RSI, Mode A): init | schema | eval | history | show | robust | accept.

Run with HAND=sharpa semantics (the `--hand` default). Typical session:
  real2sim_eval.py init <clip>                      # full pipeline run, bundle, iteration 0
  real2sim_eval.py show <clip>                      # read feedback.md
  real2sim_eval.py eval <clip> --set objects.held.friction=1.2 --rationale "slip in carry"
  real2sim_eval.py history <clip>
"""

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

from rsi.history import format_table, iter_dir, read_history
from rsi.runner import evaluate, init_clip, real2sim_dir
from rsi.schema import schema_leaves


def _print_result(res) -> None:
    print(res.feedback_md)
    print(f"\n[{res.dir}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="sharpa")
    ap.add_argument("--backend", default=None, help="simulator backend (default $E2H_SIM_BACKEND or mujoco)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").add_argument("clip", type=Path)
    s = sub.add_parser("schema")
    s.add_argument("--affects", choices=("sim", "plan"))
    e = sub.add_parser("eval")
    e.add_argument("clip", type=Path)
    e.add_argument("--params", action="append", default=[])
    e.add_argument("--set", action="append", default=[], dest="sets")
    e.add_argument("--freeze-plan", action="store_true")
    e.add_argument("--offsets", choices=("best", "grid"), default="best")
    e.add_argument("--render", action="store_true")
    e.add_argument("--parent", type=int)
    e.add_argument("--rationale", default="")
    for name in ("history",):
        sub.add_parser(name).add_argument("clip", type=Path)
    sh = sub.add_parser("show")
    sh.add_argument("clip", type=Path)
    sh.add_argument("--iter", type=int)
    for name in ("robust", "accept"):
        p = sub.add_parser(name)
        p.add_argument("clip", type=Path)
        p.add_argument("--iter", type=int, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "schema":
        for leaf in schema_leaves():
            if args.affects is None or args.affects in leaf.affects:
                print(json.dumps(dataclasses.asdict(leaf)))
        return 0
    from rsi.backends import get_backend

    backend = get_backend(args.backend)
    root = real2sim_dir(args.clip, args.hand)
    if args.cmd == "init":
        _print_result(init_clip(args.clip, hand=args.hand, backend=backend))
    elif args.cmd == "eval":
        _print_result(
            evaluate(
                args.clip,
                hand=args.hand,
                backend=backend,
                params_files=args.params,
                sets=args.sets,
                freeze_plan=args.freeze_plan,
                offsets=args.offsets,
                render=args.render,
                parent=args.parent,
                rationale=args.rationale,
            )
        )
    elif args.cmd == "history":
        print(format_table(read_history(root)))
    elif args.cmd == "show":
        n = args.iter if args.iter is not None else read_history(root)[-1].iter
        print((iter_dir(root, n) / "feedback.md").read_text())
    elif args.cmd == "robust":
        from rsi.robust import robust_check

        report = robust_check(args.clip, args.hand, args.iter, backend)
        print(json.dumps(report, indent=1))
        return 0 if report["passed"] else 1
    elif args.cmd == "accept":
        rpt = iter_dir(root, args.iter) / "robust.json"
        if not rpt.exists() or not json.loads(rpt.read_text())["passed"]:
            print("accept needs a passing `robust` run for this iteration", file=sys.stderr)
            return 1
        shutil.copyfile(iter_dir(root, args.iter) / "params.yaml", root / "accepted.yaml")
        print(f"accepted iteration {args.iter} -> {root / 'accepted.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
