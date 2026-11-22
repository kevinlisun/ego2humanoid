"""Robot asset contract."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from contracts.enums import EXPECTED_G1_DEX3_DOF, ROBOT_ID


class RobotAssetSpec(BaseModel):
    """Filled by real asset inspection, not fabricated in EpisodeInput."""

    robot_id: Literal["unitree_g1_dex3_v1"] = ROBOT_ID
    sim_backend: Literal["mujoco", "omnigibson"] = "mujoco"
    source_mjcf_uri: str
    og_robot_asset_uri: str | None = None
    source_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    og_asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    joint_names: list[str] = Field(min_length=EXPECTED_G1_DEX3_DOF, max_length=EXPECTED_G1_DEX3_DOF)
    actuator_names: list[str] = Field(min_length=EXPECTED_G1_DEX3_DOF)
    joint_count: Literal[43] = EXPECTED_G1_DEX3_DOF
    joint_mapping_uri: str
    fk_parity_passed: bool

    @model_validator(mode="after")
    def check_mapping(self):
        if len(set(self.joint_names)) != self.joint_count:
            raise ValueError("G1 Dex3 joint names must be unique")
        if len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("G1 Dex3 actuator names must be unique")
        return self
