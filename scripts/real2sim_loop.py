"""Mode B real2sim loop: evaluate, ask the Copilot relay for a parameter patch, repeat.

Usage: python scripts/real2sim_loop.py <clip_dir> [--max-iters 20] [--patience 5]
       [--relay-dir DIR] [--hand sharpa]
Exit code 0 = DONE, 75 = a relay request awaits a Copilot answer (see <relay-dir>/outbox).
"""

import argparse
import sys
from pathlib import Path

from rsi.loop import run_loop

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--relay-dir", type=Path, default=ROOT / "relay")
    ap.add_argument("--hand", default="sharpa")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    from rsi.backends import get_backend
    from rsi.runner import evaluate, real2sim_dir

    backend = get_backend(args.backend)
    clip_dir = args.clip_dir.resolve()

    def step(patch, rationale, parent):
        return evaluate(
            clip_dir,
            hand=args.hand,
            patch=patch,
            rationale=rationale,
            parent=parent,
            mode="B",
            backend=backend,
        )

    return run_loop(
        real2sim_dir(clip_dir, args.hand),
        args.relay_dir,
        step,
        hand=args.hand,
        max_iters=args.max_iters,
        patience=args.patience,
    )


if __name__ == "__main__":
    sys.exit(main())
