"""Contracts of the agentic real2sim loop (Track RSI): feedback, history and agent answers."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

REAL2SIM_FEEDBACK_VERSION = "real2sim_feedback/1"
HISTORY_VERSION = "real2sim_history/1"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PhaseSpan(_Frozen):
    name: str
    t0_s: float
    t1_s: float


class FeedbackEvent(_Frozen):
    t_s: float
    phase: str
    kind: str  # CONTACT_ON, LIFT_OFF, SLIP, EJECT, DROP, PENETRATION, ... (see qa/feedback.py)
    object: str | None = None
    detail: str = ""


class FeedbackHint(_Frozen):
    pattern: str
    param: str  # dotted SimParams path, e.g. "objects.held.friction"
    current: Any = None
    range: tuple[float, float] | None = None
    direction: Literal["up", "down", "set", "tune"] = "tune"
    affects: tuple[str, ...] = ("sim",)
    why: str = ""


class QaSummary(_Frozen):
    success: bool
    failure_reasons: tuple[str, ...] = ()
    lift_height_m: float | None = None
    translation_m: float | None = None
    expected_translation_m: float | None = None
    place_error_m: float | None = None


class Real2SimFeedback(_Frozen):
    version: str = REAL2SIM_FEEDBACK_VERSION
    clip: str
    hand: str
    iter: int = Field(ge=0)
    params_hash: str
    parent_iter: int | None = None
    tier_run: Literal["rollout", "full"] = "rollout"
    plan_frozen: bool = False
    qa: QaSummary
    residuals: dict[str, float | None] = Field(default_factory=dict)
    match_score: float
    phases: tuple[PhaseSpan, ...] = ()
    events: tuple[FeedbackEvent, ...] = ()
    hints: tuple[FeedbackHint, ...] = ()
    wall_time_s: float = 0.0


class HistoryEntry(_Frozen):
    version: str = HISTORY_VERSION
    iter: int = Field(ge=0)
    parent: int | None = None
    mode: Literal["A", "B", "human"] = "A"
    params_hash: str
    plan_params_hash: str
    patch: dict[str, Any] = Field(default_factory=dict)  # changed leaves vs the parent, dotted path -> value
    rationale: str = ""
    tier_run: Literal["rollout", "full"] = "rollout"
    plan_frozen: bool = False
    success: bool
    failure_reasons: tuple[str, ...] = ()
    match_score: float
    residuals: dict[str, float | None] = Field(default_factory=dict)
    wall_time_s: float = 0.0
    utc: str = ""


class RelayParamAnswer(_Frozen):
    """What the Copilot relay answers for a `real2sim_params` request (nested SimParams patch)."""

    patch: dict[str, Any]
    rationale: str = ""
    stop: bool = False
