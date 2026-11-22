"""MuJoCo implementation of `SimBackend`: the one real2sim module that touches `stages/replay`."""

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from contracts.sim_backend import ClipReference, RolloutRecord, SimSession
from contracts.sim_params import SimParams
from skills.place_check import buried_fraction_of
from stages.manipulation.bundle import load_bundle
from stages.manipulation.probes import ContactProbe
from stages.manipulation.rollout import Perturbation, run_rollout
from stages.manipulation.sim_eval import (
    SimCase,
    held_physics_kw,
    offset_grid,
    pick_place_events,
    sim_tier_extra,
    start_search,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _reference(root: Path) -> tuple[ClipReference, SimCase]:
    b = load_bundle(root)
    c = b.case
    ref = ClipReference(
        meta=b.meta,
        refs=b.refs,
        best_offset_m=b.best_offset_m,
        drop_point_m=None if c.drop_point_m is None else tuple(float(v) for v in c.drop_point_m),
        demo_carry_m=c.demo_carry_m,
        release_t_s=c.release_t_s,
        receptacle_name=c.receptacle_name,
    )
    return ref, c


def _receptacle_geometry(case: SimCase, held, extra: tuple, params: SimParams) -> dict | None:
    """Tub footprint of a hollow receptacle (`skills.receptacle_check.ReceptacleGeometry` + held half z)."""
    from stages.reconstruction.objects.place_target import container_profile, tub_profile

    obj = next((o for o in extra if o.name == case.receptacle_name), None)
    hollow, floor_z = container_profile(obj) if obj is not None else (False, 0.0)
    if not hollow:
        return None
    z, frac = tub_profile(obj, floor_z)
    half = np.asarray(obj.half_size_m[:2], float)
    margin = params.trajectory.receptacle_opening_margin_m
    return dict(
        centre_xy_m=tuple(float(v) for v in obj.pos_m[:2]),
        inner_half_xy_m=tuple(float(v) for v in frac[-1] * half + margin),
        outer_half_xy_m=tuple(float(v) for v in half + margin),
        rim_z_m=float(z[-1]),
        floor_z_m=float(z[0]),
        held_half_z_m=float(held.half_size_m[2]),
    )


def _case(session: SimSession, params: SimParams) -> tuple[SimCase, tuple]:
    case: SimCase = session.native
    extra = sim_tier_extra(case, params)
    return replace(case, mesh_kw=held_physics_kw(case, params), extra=extra), extra


class MujocoBackend:
    name = "mujoco"

    def load(self, root: Path) -> SimSession:
        ref, case = _reference(Path(root))
        return SimSession(ref, case)

    def plan(self, clip: Path, hand: str, params_file: Path) -> None:
        env = {**os.environ, "HAND": hand, "E2H_SIM_PARAMS": str(params_file)}
        script = REPO_ROOT / "scripts" / "retarget_estimated.py"
        subprocess.run([sys.executable, str(script), str(clip)], env=env, check=True)

    def rollout(
        self,
        session: SimSession,
        params: SimParams,
        start_xy_m: Sequence[float],
        *,
        seed: int = 0,
        perturb: bool = False,
        video_path: Path | None = None,
    ) -> RolloutRecord:
        case, extra = _case(session, params)
        box = case.held_at(start_xy_m)
        names = [box.name, *(o.name for o in extra)]
        holder: dict[str, ContactProbe] = {}
        recorder = None
        if video_path is not None:
            from stages.manipulation.render import VideoRecorder

            recorder = VideoRecorder(video_path, fps=50, size=(480, 640))

        def on_record(model, data):
            if "p" not in holder:
                holder["p"] = ContactProbe(model, names)
            holder["p"](model, data)
            if recorder is not None:
                recorder(model, data)

        res = run_rollout(
            case.config(params, (box, *extra)),
            case.timestamps_s,
            case.q_target_rad,
            seed=seed,
            perturbation=Perturbation() if perturb else None,
            on_record=on_record,
        )
        if recorder is not None:
            recorder.close()
        pev = pick_place_events(case, res)
        buried = None
        if case.receptacle_name in names[1:]:
            k = names.index(case.receptacle_name)
            T = res.arrays["object_T_sim"]
            buried = buried_fraction_of(box, extra[k - 1], T[-1, 0], T[-1, k])
        return RolloutRecord(
            arrays=res.arrays,
            probe=holder["p"].arrays(),
            object_names=names,
            static_flags=[box.static, *(o.static for o in extra)],
            pick_place=pev.as_dict(),
            place_buried_frac=buried,
            receptacle=_receptacle_geometry(case, box, extra, params),
            start_xy_m=(float(start_xy_m[0]), float(start_xy_m[1])),
        )

    def search_start(self, session: SimSession, params: SimParams) -> tuple[float, float]:
        case, extra = _case(session, params)
        sr = start_search(case, params, offset_grid(params.trajectory.start_search_m), extra)
        return float(sr.start_m[0]), float(sr.start_m[1])

    def start_grid(self, session: SimSession, params: SimParams) -> list[tuple[float, float]]:
        case: SimCase = session.native
        held = params.objects.get("held")
        shift = np.zeros(2) if held is None else np.asarray(held.pos_offset_m[:2], float)
        return [
            (float(case.anchor[0] + shift[0] + dx), float(case.anchor[1] + shift[1] + dy))
            for dx, dy in offset_grid(params.trajectory.start_search_m)
        ]
