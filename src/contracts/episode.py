"""Episode input contract."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from contracts.enums import EXPECTED_G1_DEX3_DOF, ROBOT_ID, SCHEMA_VERSION


class ArtifactRef(BaseModel):
    uri: str = Field(min_length=1, description="Remote URI or sandbox-relative/local file")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class VideoSpec(BaseModel):
    video: ArtifactRef
    camera_id: str = "ego0"
    # All metadata below is optional in pure-RGB mode; ingest probes the MP4.
    fps_nominal: float | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    intrinsics_3x3: list[list[float]] | None = None
    camera_intrinsics_source: Literal["unknown", "estimated_from_rgb", "measured"] = "unknown"

    @field_validator("intrinsics_3x3")
    @classmethod
    def validate_intrinsics(cls, k):
        if k is not None and (len(k) != 3 or any(len(row) != 3 for row in k)):
            raise ValueError("intrinsics_3x3 must be a 3x3 matrix")
        return k


class RobotRequest(BaseModel):
    """Robot chosen by product; actual joint names/asset resolved by M0 probe."""

    robot_id: Literal["unitree_g1_dex3_v1"] = ROBOT_ID
    expected_joint_dof: Literal[43] = EXPECTED_G1_DEX3_DOF
    base_mode: Literal["fixed_pelvis"] = "fixed_pelvis"
    arms_actuated: Literal[True] = True
    fingers_actuated: Literal[True] = True
    robot_asset_profile: str = "protomotions_g1_dex3_review_required"


class EpisodeInput(BaseModel):
    schema_version: Literal["1.2"] = SCHEMA_VERSION
    episode_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    video: VideoSpec
    task_description: str = "grasp, lift, translate and place a single rigid object"
    robot: RobotRequest = Field(default_factory=RobotRequest)
    target_object_hint: str | None = None
    scene_seed: int = 0
    usage_class: Literal["noncommercial_research"] = "noncommercial_research"
    physics_required: Literal[True] = True
    # Optional internal model priors; no metric user annotation required.
    scale_policy: Literal["rgb_scene_prior_with_uncertainty"] = "rgb_scene_prior_with_uncertainty"
