"""Trim a clip's ``video.mp4`` to the manipulation window (for your own long videos).

    uv run --extra video python scripts/trim_clip.py data/derived/egodex/loco_manip_allinone 49.5 60

The untouched original goes to ``<clip>/orig/video_full.mp4`` (once) and ``orig/trim.json`` records the window.
``video.mp4`` becomes the H.264 window and its sha256 in ``episode.json`` is refreshed. Delete the perception
outputs of the clip afterwards (they were computed on the old frames) and re-run ``scripts/run_pipeline.sh``.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import av


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("start_s", type=float)
    ap.add_argument("end_s", type=float)
    a = ap.parse_args()
    d = a.clip_dir
    full = d / "orig" / "video_full.mp4"
    if not full.exists():
        full.parent.mkdir(exist_ok=True)
        shutil.move(d / "video.mp4", full)
    src = av.open(str(full))
    vin = src.streams.video[0]
    fps = vin.average_rate
    out = av.open(str(d / "video.mp4"), "w")
    vout = out.add_stream("libx264", rate=fps)
    vout.width, vout.height, vout.pix_fmt = vin.width, vin.height, "yuv420p"
    vout.options = {"crf": "18"}
    n = 0
    for fr in src.decode(video=0):
        t = float(fr.pts * vin.time_base)
        if t < a.start_s:
            continue
        if t >= a.end_s:
            break
        for p in vout.encode(av.VideoFrame.from_ndarray(fr.to_ndarray(format="rgb24"), format="rgb24")):
            out.mux(p)
        n += 1
    for p in vout.encode():
        out.mux(p)
    out.close()
    sha = hashlib.sha256((d / "video.mp4").read_bytes()).hexdigest()
    ep = json.loads((d / "episode.json").read_text())
    ep["video"]["video"]["sha256"] = sha
    (d / "episode.json").write_text(json.dumps(ep, indent=2))
    (d / "orig" / "trim.json").write_text(json.dumps({"start_s": a.start_s, "end_s": a.end_s, "frames": n}))
    print(f"{n} frames, {n / float(fps):.2f} s -> {d / 'video.mp4'}")


if __name__ == "__main__":
    main()
