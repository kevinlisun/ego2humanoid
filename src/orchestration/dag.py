"""Minimal stage DAG: topological execution with per-stage output directories."""

from collections.abc import Callable
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path

StageFn = Callable[[Path, dict[str, Path]], None]


@dataclass
class Stage:
    name: str
    fn: StageFn
    deps: tuple[str, ...] = ()


@dataclass
class Dag:
    stages: list[Stage] = field(default_factory=list)

    def add(self, name: str, fn: StageFn, deps: tuple[str, ...] = ()) -> None:
        if any(s.name == name for s in self.stages):
            raise ValueError(f"duplicate stage {name}")
        self.stages.append(Stage(name, fn, deps))

    def order(self) -> list[str]:
        by_name = {s.name: s for s in self.stages}
        for s in self.stages:
            for d in s.deps:
                if d not in by_name:
                    raise ValueError(f"stage {s.name} depends on unknown stage {d}")
        return list(TopologicalSorter({s.name: set(s.deps) for s in self.stages}).static_order())
