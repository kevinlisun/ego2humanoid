"""Hands backend selection from ``configs/hands.yaml`` (pure python, no heavy imports)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "hands.yaml"
POSE_BACKENDS = ("fah",)
TACTILE_BACKENDS = ("none", "egotac")
TACTILE_ENV_VAR = "E2H_TACTILE_BACKEND"
DEFAULT_TACTILE_NPZ = "tactile_egotac.npz"


@dataclass(frozen=True)
class HandsConfig:
    backend: str = "fah"
    tactile_backend: str = "none"
    tactile_npz_name: str = DEFAULT_TACTILE_NPZ

    @property
    def tactile_enabled(self) -> bool:
        return self.tactile_backend != "none"


def load_hands_config(path: str | Path | None = None, tactile_override: str | None = None) -> HandsConfig:
    """Load and validate the config; ``tactile_override`` beats ``$E2H_TACTILE_BACKEND`` beats the file."""
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    raw = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    tactile = raw.get("tactile") or {}
    backend = str(raw.get("backend", "fah"))
    tactile_backend = tactile_override or os.environ.get(TACTILE_ENV_VAR) or tactile.get("backend", "none")
    tactile_backend = str(tactile_backend)
    npz_name = str(tactile.get("npz_name", DEFAULT_TACTILE_NPZ))
    if backend not in POSE_BACKENDS:
        raise ValueError(f"unknown hands backend '{backend}', expected one of {POSE_BACKENDS}")
    if tactile_backend not in TACTILE_BACKENDS:
        raise ValueError(f"unknown tactile backend '{tactile_backend}', expected one of {TACTILE_BACKENDS}")
    if Path(npz_name).name != npz_name or not npz_name.endswith(".npz"):
        raise ValueError(f"tactile.npz_name must be a bare .npz file name, got '{npz_name}'")
    return HandsConfig(backend, tactile_backend, npz_name)


def artefact_names(cfg: HandsConfig, focal_px: int) -> dict[str, str]:
    """Artefact file names (relative to the clip dir) the hands stage reads for ``cfg``."""
    names = {"pose": f"hands_fah_f{focal_px}.npz"}
    if cfg.tactile_enabled:
        names["tactile"] = cfg.tactile_npz_name
    return names
