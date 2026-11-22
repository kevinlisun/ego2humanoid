"""Digital-twin asset package contract (Track BT, BT.2): ``asset.yaml`` schema and validator.

Layout of one asset under ``data/assets/twin/<id>/``: ``asset.yaml``, ``visual.glb`` (Z-up, origin at
the bottom centre of the bbox, metres), ``collision/*.obj`` (CoACD pieces) and ``preview.png``.
"""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TWIN_ASSET_VERSION = "twin_asset/1"
MAX_COLLISION_PIECES = 32
ID_PATTERN = r"^[a-z][a-z0-9_]*$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RestPose(_Strict):
    """A stable resting orientation: which bbox face points down and how likely it is."""

    down_face: Literal["+x", "-x", "+y", "-y", "+z", "-z"]
    probability: float = Field(gt=0, le=1)


class PhysicsPrior(_Strict):
    density_kg_m3: float | None = Field(None, gt=0, le=20000)
    mass_kg: float | None = Field(None, gt=0, le=50)
    friction: float = Field(1.0, ge=0, le=3)
    restitution: float = Field(0.0, ge=0, le=1)
    source: Literal["prior", "measured", "rsi"] = "prior"

    @model_validator(mode="after")
    def _one_of_density_or_mass(self) -> "PhysicsPrior":
        if (self.density_kg_m3 is None) == (self.mass_kg is None):
            raise ValueError("give exactly one of density_kg_m3 and mass_kg")
        return self


class Hollow(_Strict):
    """Open-topped container: the inner floor height above the asset origin."""

    inner_floor_z_m: float = Field(gt=0)


class AssetSpec(_Strict):
    version: Literal["twin_asset/1"] = TWIN_ASSET_VERSION
    id: str = Field(pattern=ID_PATTERN)
    category: str = Field(min_length=1)
    synonyms: list[str] = Field(default_factory=list)
    colour_hsv: list[tuple[float, float, float]] = Field(
        min_length=1, description="dominant colours: hue deg [0,360), saturation, value in [0,1]"
    )
    size_m: tuple[float, float, float] = Field(description="bbox x, y, z in metres; the authoritative scale")
    up_axis: Literal["+z"] = "+z"
    rest_poses: list[RestPose] = Field(min_length=1)
    physics: PhysicsPrior
    hollow: Hollow | None = None
    licence: str = Field(min_length=1)
    author: str = Field(min_length=1)
    sha256: dict[str, str] = Field(default_factory=dict, description="file name -> hex digest")

    @field_validator("size_m")
    @classmethod
    def _size_in_range(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(0.005 <= x <= 1.5 for x in v):
            raise ValueError("size_m must lie within 5 mm .. 1.5 m per axis")
        return v

    @field_validator("colour_hsv")
    @classmethod
    def _hsv_in_range(cls, v: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        for h, s, val in v:
            if not (0 <= h < 360 and 0 <= s <= 1 and 0 <= val <= 1):
                raise ValueError(f"HSV out of range: {(h, s, val)}")
        return v

    @model_validator(mode="after")
    def _rest_and_hollow(self) -> "AssetSpec":
        if abs(sum(p.probability for p in self.rest_poses) - 1.0) > 1e-3:
            raise ValueError("rest_poses probabilities must sum to 1")
        if self.hollow is not None and self.hollow.inner_floor_z_m >= self.size_m[2]:
            raise ValueError("hollow.inner_floor_z_m must lie below the asset height")
        return self


def load_asset_spec(path: str | Path) -> AssetSpec:
    return AssetSpec.model_validate(yaml.safe_load(Path(path).read_text()))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_asset_dir(root: str | Path) -> list[str]:
    """Problems found in one asset directory (empty list: valid)."""
    root = Path(root)
    spec_file = root / "asset.yaml"
    if not spec_file.exists():
        return ["asset.yaml missing"]
    try:
        spec = load_asset_spec(spec_file)
    except Exception as e:  # yaml or pydantic: report, do not raise
        return [f"asset.yaml invalid: {e}"]
    problems = []
    if spec.id != root.name:
        problems.append(f"id {spec.id!r} differs from the directory name {root.name!r}")
    if not (root / "visual.glb").exists():
        problems.append("visual.glb missing")
    pieces = list((root / "collision").glob("*.obj")) if (root / "collision").is_dir() else []
    if not pieces:
        problems.append("collision/*.obj missing")
    elif len(pieces) > MAX_COLLISION_PIECES:
        problems.append(f"{len(pieces)} collision pieces exceed {MAX_COLLISION_PIECES}")
    for name, digest in spec.sha256.items():
        f = root / name
        if not f.exists():
            problems.append(f"sha256 lists missing file {name}")
        elif sha256_file(f) != digest:
            problems.append(f"sha256 mismatch for {name}")
    return problems
