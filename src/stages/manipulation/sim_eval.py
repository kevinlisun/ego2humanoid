"""Re-simulation of a planned trajectory: start-offset search and rollouts (Track RSI, RSI.6).

`SimCase` freezes everything the planner produced that a physics rollout needs, so the full
pipeline script and the rollout-only tier of the real2sim loop share one code path.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from contracts.sim_params import SimParams
from skills.params.scene_overrides import apply_scene_overrides
from stages.manipulation.events import PickPlaceEvents, detect_pick_place
from stages.manipulation.mjcf_scene import BoxObject, SceneConfig
from stages.manipulation.rollout import RolloutResult, run_rollout

HELD_NAME = "perceived"


@dataclass(frozen=True)
class SimCase:
    hand: str
    T_sim_robot_base: np.ndarray
    table_kw: dict
    timestamps_s: np.ndarray
    q_target_rad: np.ndarray
    anchor: np.ndarray  # where the pinch closes on the resting object (xy, grip z)
    half_size_m: np.ndarray
    table_z: float
    mesh_kw: dict  # held object geometry and physics as planned (mesh_*, collision_pieces, mass_kg, friction)
    extra: tuple[BoxObject, ...] = ()  # receptacle and scenery, after the scene edits of the planning run
    drop_point_m: np.ndarray | None = None
    demo_carry_m: float = 0.0
    release_t_s: float = 0.0
    base_flags: dict[str, tuple[bool, bool]] = field(
        default_factory=dict
    )  # name -> (static, robot_collision) before edits
    receptacle_name: str | None = None
    held_base: tuple[float, tuple[float, float, float]] | None = None  # held (mass, friction) before edits

    def config(self, params: SimParams, objects: Sequence[BoxObject], **kw) -> SceneConfig:
        return SceneConfig.from_params(
            params,
            T_sim_robot_base=self.T_sim_robot_base,
            objects=tuple(objects),
            hand=self.hand,
            **self.table_kw,
            **kw,
        )

    def held_at(self, xy: Sequence[float]) -> BoxObject:
        pos = (float(xy[0]), float(xy[1]), self.table_z + float(self.half_size_m[2]))
        return BoxObject(HELD_NAME, tuple(float(h) for h in self.half_size_m), pos, **self.mesh_kw)


@dataclass
class SearchResult:
    start_m: np.ndarray
    box: BoxObject
    result: RolloutResult
    events: PickPlaceEvents


def offset_grid(start_search_m: float) -> list[tuple[float, float]]:
    """The no-offset start first, then a 5 x 5 grid of start offsets within +-`start_search_m`."""
    steps = (-start_search_m, -start_search_m / 2, 0.0, start_search_m / 2, start_search_m)
    return [(0.0, 0.0)] + [(dx, dy) for dx in steps for dy in steps if (dx, dy) != (0.0, 0.0)]


def sim_tier_extra(case: SimCase, params: SimParams) -> tuple[BoxObject, ...]:
    """Scene objects with the sim-tier edits (static / robot_collision) of `params` applied.

    Pose, scale and enable edits are plan-tier: they were baked in by the planning run.
    """
    neutral = {
        k: v.model_copy(
            update=dict(
                enabled=True, pos_offset_m=(0.0, 0.0, 0.0), yaw_offset_deg=0.0, size_scale=(1.0, 1.0, 1.0)
            )
        )
        for k, v in params.objects.items()
    }
    base = []
    for o in case.extra:
        static, robot = case.base_flags.get(o.name, (o.static, o.robot_collision))
        base.append(replace(o, static=static, robot_collision=robot))
    roles = {"receptacle": case.receptacle_name} if case.receptacle_name else {}
    return tuple(apply_scene_overrides(base, params.model_copy(update=dict(objects=neutral)), roles))


def held_physics_kw(case: SimCase, params: SimParams) -> dict:
    """`mesh_kw` with the `objects.held` mass / friction of `params` (None restores the planned value)."""
    kw = dict(case.mesh_kw)
    base_mass, base_fric = case.held_base if case.held_base else (kw.get("mass_kg"), kw.get("friction"))
    held = params.objects.get("held")
    mass = held.mass_kg if held and held.mass_kg is not None else base_mass
    fric = (held.friction, 0.01, 0.001) if held and held.friction is not None else base_fric
    if mass is not None:
        kw["mass_kg"] = mass
    if fric is not None:
        kw["friction"] = fric
    return kw


def start_search(
    case: SimCase,
    params: SimParams,
    offsets_m: Sequence[tuple[float, float]],
    extra: Sequence[BoxObject] | None = None,
) -> SearchResult:
    """Roll out every candidate start offset; the best is the first success with the highest lift."""
    extra = case.extra if extra is None else tuple(extra)
    held = params.objects.get("held")
    shift = np.zeros(2) if held is None else np.asarray(held.pos_offset_m[:2], float)
    best = None
    for dx, dy in offsets_m:
        cand = np.array(
            [
                case.anchor[0] + shift[0] + dx,
                case.anchor[1] + shift[1] + dy,
                case.table_z + case.half_size_m[2],
            ]
        )
        box = case.held_at(cand[:2])
        res = run_rollout(case.config(params, (box, *extra)), case.timestamps_s, case.q_target_rad)
        ev = pick_place_events(case, res)
        score = (ev.success, ev.lift_height_m)
        if best is None or score > best[0]:
            best = (score, cand, box, res, ev)
    _, start, box, res, ev = best
    return SearchResult(start, box, res, ev)


def pick_place_events(case: SimCase, res: RolloutResult) -> PickPlaceEvents:
    return detect_pick_place(
        res.arrays["timestamps_s"],
        res.arrays["object_T_sim"][:, 0],
        place_target_pos_m=case.drop_point_m,
        expected_translation_m=case.demo_carry_m,
        hand_contact=res.arrays["object_robot_contact"][:, 0],
        release_t_s=case.release_t_s,
    )
