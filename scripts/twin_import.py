"""CLI: import an authored mesh into ``data/assets/twin/<id>/`` (Track BT, BT.3).

    uv run --extra objects python scripts/twin_import.py mug.glb --id mug_white --category mug \\
        --size 0.08 0.08 0.095 --hsv 0 0.02 0.95 --density 900 \\
        --licence CC0 --author owner --hollow-floor 0.006
"""

import argparse
import json
from pathlib import Path

from stages.reconstruction.objects.twin_import import import_twin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("data/assets/twin"))
    ap.add_argument(
        "--size", type=float, nargs=3, metavar=("X", "Y", "Z"), help="bbox metres; default: keep source scale"
    )
    ap.add_argument("--hsv", type=float, nargs=3, action="append", required=True, metavar=("H", "S", "V"))
    ap.add_argument("--synonym", action="append", default=[])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--density", type=float, help="kg/m^3 of the solid-equivalent object")
    g.add_argument("--mass", type=float, help="kg")
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument(
        "--hollow-floor", type=float, default=None, help="inner floor height of an open container (m)"
    )
    ap.add_argument("--licence", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--max-pieces", type=int, default=16)
    a = ap.parse_args()

    physics = {"friction": a.friction, **({"density_kg_m3": a.density} if a.density else {"mass_kg": a.mass})}
    report = import_twin(
        a.src,
        a.out_root / a.id,
        asset_id=a.id,
        category=a.category,
        size_m=tuple(a.size) if a.size else None,
        colour_hsv=[tuple(h) for h in a.hsv],
        physics=physics,
        licence=a.licence,
        author=a.author,
        synonyms=a.synonym,
        hollow_inner_floor_z_m=a.hollow_floor,
        max_pieces=a.max_pieces,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["validation"] else 0)


if __name__ == "__main__":
    main()
