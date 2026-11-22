"""MuJoCo-side physics fidelity checks on the full-rate ``sim_state`` (independent of how it is rendered)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PAD_FORCE_MAX_N = 8.2
LIFT_THRESHOLD_M = 0.01
HELD_BODY = "perceived"


def _body_index(names: list[str], name: str) -> int | None:
    return names.index(name) if name in names else None


def fidelity_report(
    sim_state: Mapping,
    objects: list[str],
    table_top_z_m: float | None,
    half_heights_m: Mapping[str, float] | None = None,
    held: str = HELD_BODY,
    max_sink_m: float = 0.005,
    max_drift_m: float = 0.01,
    max_rest_speed_mps: float = 0.02,
    quat_tol: float = 1e-3,
) -> dict:
    """Check penetration proxy, rest drift, lift/release timing, pad force and finiteness.

    ``sim_state`` needs ``t`` (T,), ``body_names`` (B,), ``xpos`` (T,B,3), ``xquat`` (T,B,4 wxyz); may hold
    ``pad_force_N`` (T,P) and ``object_contact`` (T,) or (T,K) (truthy while the hand touches the object).
    """
    t = np.asarray(sim_state["t"], dtype=float)
    names = [str(n) for n in sim_state["body_names"]]
    xpos = np.asarray(sim_state["xpos"], dtype=float)
    xquat = np.asarray(sim_state["xquat"], dtype=float)
    half = dict(half_heights_m or {})
    problems: list[str] = []

    finite = bool(np.isfinite(xpos).all() and np.isfinite(xquat).all())
    if not finite:
        problems.append("non-finite xpos/xquat")
    norms = np.linalg.norm(xquat, axis=-1)
    quat_dev = float(np.nanmax(np.abs(norms - 1.0))) if norms.size else 0.0
    quat_ok = bool(finite and quat_dev <= quat_tol)
    if finite and not quat_ok:
        problems.append(f"quaternion norm deviates by {quat_dev:.2e}")

    per_object: dict[str, dict] = {}
    max_sink = 0.0
    for name in objects:
        i = _body_index(names, name)
        if i is None:
            problems.append(f"object '{name}' not in body_names")
            continue
        p = xpos[:, i]
        entry: dict = {}
        if table_top_z_m is not None:
            sink = float(np.nanmax(table_top_z_m - (p[:, 2] - half.get(name, 0.0))))
            entry["sink_m"] = max(0.0, sink)
            max_sink = max(max_sink, entry["sink_m"])
            if entry["sink_m"] > max_sink_m:
                problems.append(f"'{name}' sinks {entry['sink_m'] * 1000:.1f} mm below the table")
        if name != held:
            drift = float(np.nanmax(np.linalg.norm(p - p[0], axis=1)))
            tail = t >= t[-1] - 1.0
            speed = 0.0
            if tail.sum() >= 2:
                speed = float(np.linalg.norm(p[tail][-1] - p[tail][0]) / (t[tail][-1] - t[tail][0]))
            entry.update(drift_m=drift, final_speed_mps=speed)
            if drift > max_drift_m:
                problems.append(f"'{name}' drifts {drift * 1000:.1f} mm from its start")
            if speed > max_rest_speed_mps:
                problems.append(f"'{name}' still moves at {speed:.3f} m/s in the final second")
        per_object[name] = entry

    events: dict = {"lift_start_s": None, "lift_end_s": None, "release_s": None, "max_rise_m": None}
    hi = _body_index(names, held)
    if hi is not None:
        z = xpos[:, hi, 2]
        rise = z - z[0]
        events["max_rise_m"] = float(np.nanmax(rise))
        up = np.flatnonzero(rise > LIFT_THRESHOLD_M)
        if up.size:
            events["lift_start_s"] = float(t[up[0]])
            events["lift_end_s"] = float(t[up[-1]])
        if "object_contact" in sim_state:
            c = np.asarray(sim_state["object_contact"])
            c = c.reshape(len(t), -1).any(axis=1)
            if c.any():
                events["release_s"] = float(t[np.flatnonzero(c)[-1]])

    pad: dict = {"present": "pad_force_N" in sim_state}
    if pad["present"]:
        f = np.asarray(sim_state["pad_force_N"], dtype=float)
        pad["nan"] = bool(np.isnan(f).any())
        pad["max_N"] = float(np.nanmax(f)) if f.size and not np.isnan(f).all() else 0.0
        pad["min_N"] = float(np.nanmin(f)) if f.size and not np.isnan(f).all() else 0.0
        pad["in_range"] = bool(not pad["nan"] and pad["min_N"] >= 0.0 and pad["max_N"] <= PAD_FORCE_MAX_N)
        if pad["nan"]:
            problems.append("NaN in pad_force_N")
        elif not pad["in_range"]:
            problems.append(
                f"pad force {pad['min_N']:.2f}..{pad['max_N']:.2f} N outside 0..{PAD_FORCE_MAX_N} N"
            )

    return {
        "ok": not problems,
        "problems": problems,
        "n_records": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "finite": finite,
        "quat_norm_ok": quat_ok,
        "quat_norm_dev": quat_dev,
        "max_sink_m": float(max_sink),
        "objects": per_object,
        "events": events,
        "pad_force": pad,
    }


def _fmt(v: float | None, spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def format_markdown(report: dict, clip_id: str) -> str:
    """One table row: clip | ok | sink mm | worst drift mm | lift s | release s | pad max N | problems."""
    objs = report["objects"].values()
    drift = max((o["drift_m"] for o in objs if "drift_m" in o), default=None)
    ev, pad = report["events"], report["pad_force"]
    cells = [
        clip_id,
        "OK" if report["ok"] else "FAIL",
        _fmt(report["max_sink_m"] * 1000, ".1f"),
        _fmt(None if drift is None else drift * 1000, ".1f"),
        _fmt(ev["lift_start_s"]),
        _fmt(ev["release_s"]),
        _fmt(pad.get("max_N")),
        "; ".join(report["problems"]) or "-",
    ]
    return "| " + " | ".join(cells) + " |"


def _table_top(model_json: Path) -> float | None:
    model = json.loads(model_json.read_text())
    g = next((g for g in model["geoms"] if g["name"] == "table_top" and g["type"] == "box"), None)
    return None if g is None else float(g["pos"][2] + g["size"][2])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--tag", default="_bt")
    a = ap.parse_args(argv)
    with np.load(a.clip_dir / "review" / f"sim_state_sharpa{a.tag}.npz", allow_pickle=False) as z:
        state = {k: z[k] for k in z.files}
    model_json = a.clip_dir / "review" / "sim_model" / "sim_model.json"
    top = _table_top(model_json) if model_json.exists() else None
    names = [str(n) for n in state["body_names"]]
    objs = [n for n in names if n == HELD_BODY or n.startswith("scene_")]
    print(format_markdown(fidelity_report(state, objs, top), a.clip_dir.name))


if __name__ == "__main__":
    main()
