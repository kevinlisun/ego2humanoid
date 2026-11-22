#!/usr/bin/env python
"""LLM + RSI loop that tunes the Blender twin's look to match the real video (Track BT, BT.26).

One call = one round, resumable. The LLM is the Copilot relay (D8: no external API):

  1. render the twin at the real camera (``render_twin_sim.py --view human``) with the current appearance,
  2. score it against the real frame (hand area masked) and keep it only if it beats the best so far,
  3. write a relay request (real | twin | diff image + prompt with the params, schema and history),
  4. exit 75 until ``inbox/<id>.json`` holds ``{"text": "{\\"edits\\": [...], \\"rationale\\": \\"...\\"}"}``,
  5. apply the bounded edits (``apply_edits``) and start the next round.

State lives in ``<review>/twin_appearance/`` (``state.json``, ``best.json``, ``round_NN.png``, ``relay/``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from contracts.twin_appearance import TwinAppearance, apply_edits
from skills.twin_compare import compare, diff_image, hand_mask, side_by_side
from stages.reconstruction.scene.vlm_relay import PENDING_RELAY_EXIT_CODE, RelayClient, RelayPending

BPY_PYTHON = "/tmp/bpy_venv/bin/python"
MODEL = "copilot-relay"
PROMPT = """You tune the look of a Blender digital twin so that it matches the real photo (left image).
Images, left to right: REAL | TWIN (current render, same camera) | DIFF (red = large difference, grey = ignored hand).
Only appearance may change: shape class, size, colour, roughness, light, table colour. Use the objects
you can see; names are in `objects_in_scene`. Prefer `tray` for open boxes/lids (wall_m = rim thickness),
`box` for flat blocks and phones, `cylinder` for round things, `mesh` when the generated mesh is already good.
Texture `projected` paints the real photo onto the object; use `flat` + color_rgb when it looks smeared.
Sizes are full extents in metres in the object frame. Make few, confident edits per round.

Reply with JSON only: {{"edits": [{{"path": "objects.<name>.<field>" | "table.<field>" | "light.<field>" |
"room.<field>", "value": ...}}], "rationale": "<one sentence>"}}

Fields (value ranges are clamped): {schema}

Current appearance:
{params}

Score history (0 = identical, lower is better): {history}
Per-object score now: {rois}
"""


def _real_frame(clip: Path, scene: dict, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {index} of {clip / 'video.mp4'}")
    w, h = scene["image_wh"]
    return cv2.resize(bgr, (w, h))[..., ::-1]


def _ignore_mask(clip: Path, scene: dict, index: int) -> np.ndarray | None:
    path = clip / "hands_fah.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    j = d["joints3d"][min(index, len(d["joints3d"]) - 1)]  # (hands, 21, 3), camera frame
    ok = d["valid"][min(index, len(d["valid"]) - 1)]
    w, h = scene["image_wh"]
    f = scene["focal_px"]
    pts = []
    for hand, good in zip(j, ok, strict=False):
        if good and (hand[:, 2] > 0.05).all():
            pts.append(
                np.stack([f * hand[:, 0] / hand[:, 2] + w / 2, f * hand[:, 1] / hand[:, 2] + h / 2], -1)
            )
    return hand_mask(np.stack(pts), (w, h)) if pts else None


def _rois(scene: dict, shape_hw: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Per-object ROI = the object's projected bounding box in the real camera, padded."""
    t = np.linalg.inv(np.asarray(scene["camera_T_sim"], float))  # stored as T_sim_camera
    f, (w, h) = scene["focal_px"], scene["image_wh"]
    out = {}
    for o in scene["objects"]:
        if o.get("static") and o["name"] == "table":
            continue
        c = (t @ np.append(o["pos_m"], 1.0))[:3]
        if c[2] <= 0.05:
            continue
        r = float(np.max(o["half_size_m"])) * 1.3
        u, v, rp = f * c[0] / c[2] + w / 2, f * c[1] / c[2] + h / 2, f * r / c[2]
        box = (int(u - rp), int(v - rp), int(u + rp), int(v + rp))
        out[o["name"]] = (max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h))
    return {k: v for k, v in out.items() if v[2] > v[0] and v[3] > v[1]}


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img[..., ::-1])[1].tobytes()


def run(args: argparse.Namespace) -> int:
    clip = Path(args.clip)
    review = clip / "review"
    work = review / "twin_appearance"
    work.mkdir(exist_ok=True)
    scene = json.loads((review / "twin_scene" / "scene.json").read_text())
    state_path = work / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"round": 0, "history": [], "best": None}
    )
    cur_path = review / "twin_appearance.json"
    app = TwinAppearance.model_validate_json(cur_path.read_text()) if cur_path.exists() else TwinAppearance()
    cur_path.write_text(app.model_dump_json(indent=2))

    real = _real_frame(clip, scene, args.frame)
    ignore = _ignore_mask(clip, scene, args.frame)
    rois = _rois(scene, real.shape[:2])
    client = RelayClient(work / "relay")

    while state["round"] < args.rounds:
        n = state["round"]
        png = work / f"round_{n:02d}.png"
        if not png.exists():
            subprocess.run(
                [
                    BPY_PYTHON,
                    "scripts/render_twin_sim.py",
                    str(clip),
                    "--view",
                    "human",
                    "--width",
                    str(args.width),
                    "--samples",
                    str(args.samples),
                    "--device",
                    args.device,
                    "--appearance",
                    str(cur_path),
                    "--out",
                    str(png),
                ],
                check=True,
            )
        twin = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
        res = compare(real, twin, ignore, rois)
        best = state["best"]
        if best is None or res["score"] < best["score"]:
            state["best"] = {"round": n, "score": res["score"]}
            (work / "best.json").write_text(cur_path.read_text())
        elif (work / "best.json").exists():  # roll back a regression
            app = TwinAppearance.model_validate_json((work / "best.json").read_text())
        if not any(h["round"] == n for h in state["history"]):  # a resumed round is already logged
            state["history"].append(
                {"round": n, "score": round(res["score"], 4), "best": state["best"]["round"]}
            )
        state_path.write_text(json.dumps(state, indent=1))
        print(
            f"round {n}: score {res['score']:.4f} (best {state['best']['score']:.4f} @ {state['best']['round']})"
        )
        if res["score"] < args.target:
            break

        prompt = PROMPT.format(
            schema=json.dumps(TwinAppearance.model_json_schema()["$defs"]["ObjectAppearance"]["properties"]),
            params=app.model_dump_json(indent=1),
            history=[(h["round"], h["score"]) for h in state["history"]],
            rois={k: round(v, 3) for k, v in res["rois"].items()},
        )
        prompt += "\nobjects_in_scene: " + ", ".join(o["name"] for o in scene["objects"])
        image = SimpleNamespace(data=_png(side_by_side(real, twin, diff_image(real, twin, ignore))))
        content = SimpleNamespace(
            parts=[SimpleNamespace(text=None, inline_data=image), SimpleNamespace(text=prompt)]
        )
        try:
            reply = client.answer(MODEL, [content])
        except RelayPending as e:
            print(f"pending relay request(s): {e.request_ids}; answer in {work / 'relay' / 'inbox'}")
            return PENDING_RELAY_EXIT_CODE
        text = reply[0].text.strip().removeprefix("```json").removesuffix("```").strip()
        edits = json.loads(text).get("edits", [])
        app, rejected = apply_edits(app, edits)
        for r in rejected:
            print("rejected edit:", r)
        cur_path.write_text(app.model_dump_json(indent=2))
        state["round"] = n + 1
        state_path.write_text(json.dumps(state, indent=1))
    print("best round", state["best"])
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("clip", help="clip derived dir (contains review/twin_scene, video.mp4)")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--frame", type=int, default=0, help="real frame index compared against")
    p.add_argument("--target", type=float, default=0.05, help="stop when the score is below this")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--device", default="gpu")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
