"""Result of one real2sim iteration, shared by the runner and the Mode B loop."""

from dataclasses import dataclass
from pathlib import Path

from contracts.real2sim import HistoryEntry, Real2SimFeedback
from contracts.sim_params import SimParams


@dataclass(frozen=True)
class IterResult:
    iter: int
    dir: Path
    feedback: Real2SimFeedback
    entry: HistoryEntry
    feedback_md: str
    params: SimParams
