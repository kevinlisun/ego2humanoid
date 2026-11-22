"""Stage bundle IO: ``manifest.json`` + NPZ (never pickled)."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

from contracts.arrays import BUNDLE_SPECS, validate_arrays

M = TypeVar("M", bound=BaseModel)

MANIFEST_NAME = "manifest.json"


def save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def write_manifest(out_dir: str | Path, manifest: BaseModel) -> Path:
    path = Path(out_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return path


def write_bundle(
    out_dir: str | Path,
    manifest: BaseModel,
    arrays: Mapping[str, np.ndarray],
    npz_name: str,
    bundle: str | None = None,
) -> Path:
    """Validate ``arrays`` against the bundle spec, then write NPZ and manifest."""
    bundle = bundle or str(getattr(manifest, "stage", ""))
    if bundle in BUNDLE_SPECS:
        validate_arrays(bundle, arrays)
    out_dir = Path(out_dir)
    save_npz(out_dir / npz_name, arrays)
    return write_manifest(out_dir, manifest)


def dump_json(path: str | Path, obj: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return path
