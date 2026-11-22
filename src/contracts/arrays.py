"""Machine-checkable NPZ specs for every stage bundle (docs/DATA_CONTRACTS.md).

Shape tokens: an int is an exact size; a str is a named dimension that must agree across
all arrays of one bundle (e.g. ``N`` frames, ``O`` objects). ``None`` dtype means any
numeric dtype.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.enums import EXPECTED_G1_DEX3_DOF

DIM = int | str


class ArrayContractError(ValueError):
    """Raised when an NPZ bundle violates its array contract."""


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[DIM, ...]
    kind: str = "f"  # numpy dtype kind: f float, b bool, i/u integer
    se3_axes: bool = False  # trailing (4,4) block must be a rigid transform


def _f(*shape: DIM, se3: bool = False) -> ArraySpec:
    return ArraySpec(shape, "f", se3)


HANDS_REQUIRED = {
    "timestamps_s": _f("N"),
    "frame_ids": ArraySpec(("N",), "iu"),
    "joints2d_px": _f("N", 2, 21, 2),
    "joints3d_est": _f("N", 2, 21, 3),
    "valid": ArraySpec(("N", 2), "b"),
    "confidence": _f("N", 2),
}
HANDS_OPTIONAL = {"T_world_camera": _f("N", 4, 4, se3=True), "contact_prob": _f("N", 2, "V")}

MANO_VERTEX_COUNT = 778
TACTILE_REQUIRED = {
    "force": _f("N", 2, MANO_VERTEX_COUNT),
    "contact_prob": _f("N", 2, MANO_VERTEX_COUNT),
}
TACTILE_OPTIONAL = {"timestamps_s": _f("N"), "region_of_vertex": ArraySpec((MANO_VERTEX_COUNT,), "iu")}
TACTILE_REGIONS_REQUIRED = {
    "force_region": _f("N", 2, "R"),
    "contact_prob_region": _f("N", 2, "R"),
    "tactile_contact_flags": ArraySpec(("N", 2, "R"), "b"),
}
TACTILE_REGIONS_OPTIONAL = {"timestamps_s": _f("N")}

OBJECTS_REQUIRED = {
    "timestamps_s": _f("N"),
    "T_world_object": _f("N", "O", 4, 4, se3=True),
    "valid": ArraySpec(("N", "O"), "b"),
    "visibility": _f("N", "O"),
}

ALIGNMENT_REQUIRED = {
    "T_sim_world_Rt": _f(4, 4, se3=True),
    "scale_sim_per_world": _f(),
    "T_sim_robot_base": _f(4, 4, se3=True),
    "contact_flags": ArraySpec(("N", 2), "b"),
}

D = EXPECTED_G1_DEX3_DOF
RETARGET_REQUIRED = {
    "timestamps_s": _f("M"),
    "q_target_rad": _f("M", D),
    "qvel_target_rad_s": _f("M", D),
    "wrist_T_sim": _f("M", 2, 4, 4, se3=True),
    "base_T_sim": _f("M", 4, 4, se3=True),
}
RETARGET_OPTIONAL = {"ctrl": _f("M", "U")}

SIMULATION_REQUIRED = {
    "timestamps_s": _f("K"),
    "q_command_rad": _f("K", D),
    "q_actual_rad": _f("K", D),
    "dq_actual_rad_s": _f("K", D),
    "wrist_actual_T_sim": _f("K", 2, 4, 4, se3=True),
    "object_T_sim": _f("K", "O", 4, 4, se3=True),
}

BUNDLE_SPECS: dict[str, tuple[Mapping[str, ArraySpec], Mapping[str, ArraySpec]]] = {
    "hands": (HANDS_REQUIRED, HANDS_OPTIONAL),
    "tactile": (TACTILE_REQUIRED, TACTILE_OPTIONAL),
    "tactile_regions": (TACTILE_REGIONS_REQUIRED, TACTILE_REGIONS_OPTIONAL),
    "objects": (OBJECTS_REQUIRED, {}),
    "alignment": (ALIGNMENT_REQUIRED, {}),
    "retarget": (RETARGET_REQUIRED, RETARGET_OPTIONAL),
    "simulation": (SIMULATION_REQUIRED, {}),
}


def check_se3(T: np.ndarray, *, atol: float = 1e-5) -> None:
    """Validate that the trailing (4, 4) blocks of ``T`` are proper rigid transforms."""
    T = np.asarray(T, dtype=np.float64).reshape(-1, 4, 4)
    if not np.isfinite(T).all():
        raise ArrayContractError("transform contains non-finite values")
    R = T[:, :3, :3]
    if not np.allclose(np.swapaxes(R, 1, 2) @ R, np.eye(3), atol=atol):
        raise ArrayContractError("rotation block is not orthonormal")
    if not np.allclose(np.linalg.det(R), 1.0, atol=atol):
        raise ArrayContractError("rotation block has determinant != 1")
    if not np.allclose(T[:, 3, :], [0, 0, 0, 1], atol=atol):
        raise ArrayContractError("last row of transform must be [0, 0, 0, 1]")


def _check_array(name: str, arr: np.ndarray, spec: ArraySpec, dims: dict[str, int]) -> None:
    if arr.dtype.kind not in spec.kind:
        raise ArrayContractError(f"{name}: dtype {arr.dtype} not in kinds '{spec.kind}'")
    if arr.ndim != len(spec.shape):
        raise ArrayContractError(f"{name}: expected {len(spec.shape)} dims, got shape {arr.shape}")
    for axis, (want, got) in enumerate(zip(spec.shape, arr.shape, strict=True)):
        if isinstance(want, int):
            if got != want:
                raise ArrayContractError(f"{name}: axis {axis} must be {want}, got {got}")
        elif dims.setdefault(want, got) != got:
            raise ArrayContractError(f"{name}: dim '{want}' is {got}, expected {dims[want]}")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ArrayContractError(f"{name}: contains non-finite values")
    if spec.se3_axes:
        check_se3(arr)


def validate_arrays(bundle: str, arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Validate loaded arrays against the spec of ``bundle``; returns resolved dimensions."""
    if bundle not in BUNDLE_SPECS:
        raise ArrayContractError(f"unknown bundle '{bundle}'")
    required, optional = BUNDLE_SPECS[bundle]
    missing = sorted(set(required) - set(arrays))
    if missing:
        raise ArrayContractError(f"{bundle}: missing keys {missing}")
    dims: dict[str, int] = {}
    for name, spec in {**required, **optional}.items():
        if name in arrays:
            _check_array(f"{bundle}.{name}", np.asarray(arrays[name]), spec, dims)
    for key in ("timestamps_s",):
        if key in arrays and len(arrays[key]) > 1 and np.any(np.diff(arrays[key]) <= 0):
            raise ArrayContractError(f"{bundle}.{key}: must be strictly increasing")
    if bundle in ("tactile", "tactile_regions"):
        _check_tactile(bundle, arrays)
    return dims


def _check_tactile(bundle: str, arrays: Mapping[str, np.ndarray]) -> None:
    for key in ("contact_prob", "contact_prob_region"):
        if key in arrays:
            p = np.asarray(arrays[key])
            if p.size and (p.min() < 0.0 or p.max() > 1.0):
                raise ArrayContractError(f"{bundle}.{key}: values must be in [0, 1]")


def validate_npz(bundle: str, path: str | Path) -> dict[str, int]:
    """Load an NPZ with ``allow_pickle=False`` and validate it."""
    with np.load(path, allow_pickle=False) as data:
        return validate_arrays(bundle, {k: data[k] for k in data.files})
