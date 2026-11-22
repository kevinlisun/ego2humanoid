"""Fetch a few EgoDex test clips via remote zip (no full 16 GB download).

    python scripts/fetch_egodex.py basic_pick_place -n 12

Downloads hdf5 first, keeps only clips accepted by the selector, then fetches their mp4. Data is
CC-BY-NC-ND: keep under gitignored ``data/raw/egodex`` and never redistribute.
"""

import argparse
from pathlib import Path

from remotezip import RemoteZip

from egodex.clip_reader import load_clip
from egodex.select import score_clip

URL = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("-n", type=int, default=10)
ap.add_argument("--out", type=Path, default=Path("data/raw/egodex"))
a = ap.parse_args()

z = RemoteZip(URL)
h5s = sorted(n for n in z.namelist() if f"/{a.task}/" in n and n.endswith(".hdf5"))
kept = 0
for name in h5s:
    if kept >= a.n:
        break
    mp4 = name[:-5] + ".mp4"
    if not (a.out / name).exists():
        z.extract(name, a.out)
    path = a.out / name
    if score_clip(load_clip(path), path).accepted:
        if not (a.out / mp4).exists():
            z.extract(mp4, a.out)
        kept += 1
    else:
        path.unlink()
print("kept", kept, "of", len(h5s))
