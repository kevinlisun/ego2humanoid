"""Stage bundle contracts: hands, scene, objects, alignment, retarget."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from contracts.enums import EXPECTED_G1_DEX3_DOF, ROBOT_ID, SCHEMA_VERSION, StageStatus


class StageBundle(BaseModel):
    schema_version: Literal["1.2"] = SCHEMA_VERSION
    episode_id: str
    stage: str
    input_source: Literal["rgb_estimated", "oracle_gt", "synthetic"] = "rgb_estimated"
    manifest_uri: str
    data_uri: str
    frame_count: int = Field(ge=0)
    status: StageStatus = StageStatus.PASS
    warnings: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    source_commit: str | None = None
    image_digest: str | None = None
    upstream_model_sha256: str | None = None
    source_video_sha256: str | None = None


class HandTrackBundle(StageBundle):
    stage: Literal["hands"] = "hands"
    tracks_npz: str
    hand_mesh_type: Literal["mano", "mhr", "none"] = "mano"
    hand_3d_frame: Literal["camera", "world", "root_relative"]
    world_scale_observability: Literal["MONOCULAR_UNCERTAIN", "METRIC_GT"] = "MONOCULAR_UNCERTAIN"
    # Required npz: timestamps_s(N,), frame_ids(N,), joints2d_px(N,2,21,2),
    # joints3d_est(N,2,21,3), valid(N,2), confidence(N,2).
    # Optional: T_world_camera(N,4,4), contact_prob(N,2,V).


class TactileBundle(StageBundle):
    stage: Literal["tactile"] = "tactile"
    tactile_npz: str
    source: str
    vertex_count: Literal[778] = 778
    hand_order: tuple[Literal["left"], Literal["right"]] = ("left", "right")
    region_names: list[str] = Field(default_factory=list)
    # Required npz (validate_arrays("tactile")): force(N,2,778) >= 0, contact_prob(N,2,778) in [0,1].
    # Optional: timestamps_s(N,), region_of_vertex(778,) index into region_names (-1 = unassigned).
    # Aggregated output (validate_arrays("tactile_regions")): force_region(N,2,R),
    # contact_prob_region(N,2,R), tactile_contact_flags(N,2,R).


class SceneBundle(StageBundle):
    stage: Literal["scene"] = "scene"
    scene_descriptor_uri: str
    scene_format: Literal["omnigibson_json", "usd", "mjcf"] = "omnigibson_json"
    object_mesh_uris: dict[str, str]
    collision_enabled: bool = False
    scale_method: Literal[
        "estimated_from_rgb", "learned_prior", "manually_approved", "oracle_gt", "unknown"
    ] = "unknown"


class ObjectMotionBundle(StageBundle):
    stage: Literal["objects"] = "objects"
    tracks_npz: str
    object_ids: list[str] = Field(min_length=1)
    object_pose_method: str
    ambiguous_poses: bool = False
    # Required: timestamps_s(N,), T_world_object(N,O,4,4), valid(N,O), visibility(N,O).


class AlignedInteractionBundle(StageBundle):
    stage: Literal["alignment"] = "alignment"
    aligned_npz: str
    scale_sim_per_world: float = Field(gt=0)
    scale_method: Literal["estimated_from_rgb", "learned_prior", "manually_approved", "oracle_gt"]
    scale_range_m_per_world_unit: tuple[float, float] | None = None
    world_scale_observability: Literal["MONOCULAR_UNCERTAIN", "METRIC_GT"] = "MONOCULAR_UNCERTAIN"
    registration_rmse_px: float | None = Field(default=None, ge=0)
    reachable: bool
    # Required: T_sim_world_Rt(4,4), scale_sim_per_world scalar,
    # T_sim_robot_base(4,4), contact_flags(N,2).

    @model_validator(mode="after")
    def validate_scale_range(self):
        if self.scale_range_m_per_world_unit is not None:
            lo, hi = self.scale_range_m_per_world_unit
            if not 0 < lo <= hi:
                raise ValueError("scale range must be positive and ordered")
        return self


class RobotTrajectoryBundle(StageBundle):
    stage: Literal["retarget"] = "retarget"
    trajectory_npz: str
    robot_id: Literal["unitree_g1_dex3_v1"] = ROBOT_ID
    robot_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joint_names: list[str] = Field(min_length=43, max_length=43)
    commanded_joint_count: Literal[43] = EXPECTED_G1_DEX3_DOF
    controller_config_uri: str
    finger_mapping_policy: str
    ctrl_npz_key: str | None = "ctrl"  # actuator targets emitted by SPIDER, when present
    fk_error_m: float | None = Field(default=None, ge=0)
    # Required: timestamps_s(M,), q_target_rad(M,43),
    # qvel_target_rad_s(M,43), wrist_T_sim(M,2,4,4), base_T_sim(M,4,4).

    @model_validator(mode="after")
    def enforce_order(self):
        if len(set(self.joint_names)) != 43:
            raise ValueError("Target joint ordering must have 43 unique names")
        return self
