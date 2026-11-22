"""Simulator-neutral interface of the real2sim loop (Track RSI).

The loop (runner, robustness check, feedback, history, Mode B) only talks to a `SimBackend`. A backend
owns everything simulator-specific: how a plan is turned into a physics scene, how a rollout is
stepped, which contact data are probed and how a video is rendered. Swapping the simulator means
writing one class that satisfies this protocol and registering it in
`stages/real2sim/backends/__init__.py`; nothing in `stages/qa`, `stages/params` or the rest of
`stages/real2sim` may import a simulator package.

Conventions of the arrays in a `RolloutRecord` (all in the world frame, SI units):
  timestamps_s [T]; object_T_sim [T, N, 4, 4] (object 0 is the held one); object_robot_contact [T, N] bool;
and whatever else the feedback layer reads from `qa/` (joint targets/positions, forces), as produced by
the backend. `probe` holds per-step contact arrays (object-object / object-robot forces and depths).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from contracts.sim_params import SimParams


@dataclass(frozen=True)
class ClipReference:
    """What the plan froze about one clip, independent of the simulator that will replay it."""

    meta: dict[str, Any]  # json part of the bundle: clip, side, params, phases_t_s, best_start_xy_m, ...
    refs: dict[str, np.ndarray] = field(default_factory=dict)  # video-derived reference signals
    best_offset_m: tuple[float, float] = (0.0, 0.0)
    drop_point_m: tuple[float, float, float] | None = None
    demo_carry_m: float = 0.0
    release_t_s: float = 0.0
    receptacle_name: str | None = None

    @property
    def params(self) -> SimParams:
        return SimParams.model_validate(self.meta["params"])

    @property
    def side(self) -> str:
        return self.meta["side"]

    @property
    def phases_t_s(self) -> dict[str, tuple[float, float]]:
        return {k: (v[0], v[1]) for k, v in self.meta["phases_t_s"].items()}

    @property
    def grasp_onset_s(self) -> float | None:
        v = self.refs.get("grasp_onset_s")
        return None if v is None else float(v)

    @property
    def reach(self) -> dict | None:
        return self.meta.get("reach")

    @property
    def sized_grip_n(self) -> float | None:
        return self.meta.get("sized_grip_n")

    @property
    def held_asset(self) -> dict[str, Any] | None:
        return self.meta.get("held_asset")


@dataclass
class SimSession:
    """A loaded clip: the neutral reference plus an opaque backend handle (scene description, plan arrays)."""

    reference: ClipReference
    native: Any


@dataclass
class RolloutRecord:
    arrays: dict[str, np.ndarray]
    probe: dict[str, np.ndarray]
    object_names: list[str]  # index 0 is the held object
    static_flags: list[bool]
    pick_place: dict[str, Any]  # `PickPlaceEvents.as_dict()` of the rollout
    place_buried_frac: float | None = None
    receptacle: dict[str, Any] | None = None  # tub footprint for receptacle_check
    start_xy_m: tuple[float, float] = (0.0, 0.0)


@runtime_checkable
class SimBackend(Protocol):
    name: str

    def load(self, root: Path) -> SimSession:
        """Read the bundle of a clip (`<clip>/real2sim_<hand>/`)."""

    def plan(self, clip: Path, hand: str, params_file: Path) -> None:
        """Run the full pipeline (perception -> plan -> bundle) with these parameters."""

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
        """Replay the frozen plan once with the held object resting at `start_xy_m`."""

    def search_start(self, session: SimSession, params: SimParams) -> tuple[float, float]:
        """Best start offset over the grid of `params.trajectory.start_search_m`."""

    def start_grid(self, session: SimSession, params: SimParams) -> list[tuple[float, float]]:
        """Start positions the robustness check sweeps."""
