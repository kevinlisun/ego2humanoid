"""Tiered real2sim evaluation of one clip (Track RSI): rollout-only for physics changes, else full."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contracts.real2sim import HistoryEntry
from contracts.sim_backend import ClipReference, SimBackend
from contracts.sim_params import SimParams
from rsi.backends import get_backend
from rsi.history import append_entry, iter_dir, next_iter, read_history
from rsi.types import IterResult
from skills.params.resolve import (
    _deep_merge,
    diff_params,
    needs_plan,
    params_hash,
    plan_params_hash,
    resolve_params,
)
from skills.real2sim_report import FeedbackInputs, build_feedback, write_feedback


def real2sim_dir(clip: Path, hand: str = "sharpa") -> Path:
    return Path(clip) / f"real2sim_{hand}"


def _flat(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else k))
        return out
    return {prefix: node}


def _dump(params: SimParams) -> dict:
    return json.loads(params.model_dump_json())


def _parent_params(root: Path, parent: int | None, ref: ClipReference) -> SimParams:
    if parent is None:
        return ref.params
    f = iter_dir(root, parent) / "params.yaml"
    return SimParams.model_validate(yaml.safe_load(f.read_text())) if f.exists() else ref.params


def _refs(ref: ClipReference) -> dict:
    return {f"ref_{k}": v for k, v in ref.refs.items()}


def evaluate(
    clip_dir: Path,
    *,
    hand: str = "sharpa",
    patch: dict | None = None,
    params_files: tuple[str, ...] | list[str] = (),
    sets: tuple[str, ...] | list[str] = (),
    rationale: str = "",
    parent: int | None = None,
    mode: str = "A",
    freeze_plan: bool = False,
    offsets: str = "best",
    render: bool = False,
    force_full: bool = False,
    backend: SimBackend | None = None,
) -> IterResult:
    """Resolve parameters, run the cheapest valid tier, write `iter_NNN/` and the history entry."""
    clip = Path(clip_dir)
    root = real2sim_dir(clip, hand)
    t0 = time.perf_counter()
    backend = backend or get_backend()
    session = backend.load(root)
    ref = session.reference
    history = read_history(root)
    if parent is None and history:
        parent = history[-1].iter
    base = _parent_params(root, parent, ref)
    data = _dump(resolve_params(params_files, sets, env={})) if (params_files or sets) else _dump(base)
    if patch:
        data = _deep_merge(data, patch)
    params = SimParams.model_validate(data)

    n = next_iter(root)
    out = iter_dir(root, n)
    out.mkdir(parents=True, exist_ok=True)
    params_file = out / "params.yaml"
    params_file.write_text(yaml.safe_dump(_dump(params), sort_keys=False))

    changes = diff_params(ref.params, params)
    full = force_full or (needs_plan(changes) and not freeze_plan)
    plan_frozen = needs_plan(changes) and freeze_plan
    if full:
        backend.plan(clip, hand, params_file)
        session = backend.load(root)
        ref = session.reference

    if offsets == "grid":
        start_xy = backend.search_start(session, params)
    else:
        start_xy = ref.best_offset_m
    rec = backend.rollout(session, params, start_xy)
    inputs = FeedbackInputs(
        clip=clip.name,
        hand=ref.side,
        iter=n,
        parent_iter=parent,
        params=params,
        params_hash=params_hash(params),
        tier_run="full" if full else "rollout",
        plan_frozen=plan_frozen,
        wall_time_s=time.perf_counter() - t0,
        arrays=rec.arrays,
        probe=rec.probe,
        object_names=rec.object_names,
        held_index=0,
        static_flags=rec.static_flags,
        pick_place=rec.pick_place,
        phases_t_s=ref.phases_t_s,
        release_t_s=ref.release_t_s,
        grasp_onset_s=ref.grasp_onset_s,
        drop_point_m=ref.drop_point_m,
        demo_carry_m=ref.demo_carry_m,
        refs=_refs(ref),
        sim_pinch_contact=rec.arrays["object_robot_contact"][:, 0],
        place_buried_frac=rec.place_buried_frac,
        receptacle=rec.receptacle,
        place_retry=ref.meta.get("place_retry"),
        sized_grip_n=ref.sized_grip_n,
        scene_layout=ref.meta.get("scene_layout"),
        held_asset=ref.held_asset,
        reach=ref.reach,
    )
    fb = build_feedback(inputs)
    parent_params = _parent_params(root, parent, ref)
    changed = [(c.path, c.old, c.new) for c in diff_params(parent_params, params)]
    _, md_path = write_feedback(out, fb, changed)
    np.savez_compressed(
        out / "rollout.npz",
        **rec.arrays,
        **{f"probe_{k}": v for k, v in rec.probe.items()},
    )
    entry = HistoryEntry(
        iter=n,
        parent=parent,
        mode=mode,
        params_hash=fb.params_hash,
        plan_params_hash=plan_params_hash(params),
        patch={p: new for p, _, new in changed},
        rationale=rationale,
        tier_run=fb.tier_run,
        plan_frozen=plan_frozen,
        success=fb.qa.success,
        failure_reasons=fb.qa.failure_reasons,
        match_score=fb.match_score,
        residuals=fb.residuals,
        wall_time_s=time.perf_counter() - t0,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    append_entry(root, entry)
    if render:
        backend.rollout(session, params, start_xy, video_path=out / "sim_rsi.mp4")
    return IterResult(n, out, fb, entry, md_path.read_text(), params)


def init_clip(clip_dir: Path, hand: str = "sharpa", *, backend: SimBackend | None = None, **kw) -> IterResult:
    """Full pipeline run with the default parameters (writes the bundle), then iteration 0."""
    clip = Path(clip_dir)
    root = real2sim_dir(clip, hand)
    root.mkdir(parents=True, exist_ok=True)
    seed = root / "params_init.yaml"
    seed.write_text(yaml.safe_dump(_dump(resolve_params([], [], env={})), sort_keys=False))
    backend = backend or get_backend()
    backend.plan(clip, hand, seed)
    return evaluate(clip, hand=hand, backend=backend, **kw)
