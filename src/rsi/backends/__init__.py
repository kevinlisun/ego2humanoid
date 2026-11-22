"""Simulator backends of the real2sim loop. Add a simulator by registering a factory here."""

import os
from collections.abc import Callable

from contracts.sim_backend import SimBackend

DEFAULT_BACKEND = "mujoco"
ENV_VAR = "E2H_SIM_BACKEND"


def _mujoco() -> SimBackend:
    from .mujoco import MujocoBackend

    return MujocoBackend()


REGISTRY: dict[str, Callable[[], SimBackend]] = {"mujoco": _mujoco}


def get_backend(name: str | None = None) -> SimBackend:
    """Backend by name; default from `$E2H_SIM_BACKEND`, else MuJoCo. Imports lazily."""
    key = name or os.environ.get(ENV_VAR) or DEFAULT_BACKEND
    if key not in REGISTRY:
        raise ValueError(f"unknown sim backend {key!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[key]()
