"""Loco-manipulation contracts (Track L): task, keyframes, stance goal, arrival report, failures."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from contracts.enums import SCHEMA_VERSION


class LocoManipFailure(StrEnum):
    NO_FEASIBLE_STATIC_MANIPULATION_POSE = "NO_FEASIBLE_STATIC_MANIPULATION_POSE"
    REQUIRES_MULTI_STANCE_MANIPULATION = "REQUIRES_MULTI_STANCE_MANIPULATION"
    FOV_OCCLUSION = "FOV_OCCLUSION"
    NO_NAVIGATION_PATH = "NO_NAVIGATION_PATH"
    LOCOMOTION_UNSUPPORTED = "LOCOMOTION_UNSUPPORTED"
    LOCOMOTION_FALL = "LOCOMOTION_FALL"
    LOCOMOTION_TIMEOUT = "LOCOMOTION_TIMEOUT"
    ARRIVAL_MISALIGNMENT = "ARRIVAL_MISALIGNMENT"
    ARM_IK_INFEASIBLE = "ARM_IK_INFEASIBLE"
    COLLISION = "COLLISION"
    TASK_FAILED = "TASK_FAILED"


class TaskKeyframe(BaseModel):
    """A hand pose relative to the object frame: T_OH = T_WO^-1 T_WH (4x4, row-major)."""

    phase: Literal["pre_grasp", "grasp", "lift", "place", "release"]
    time_s: float = Field(ge=0)
    T_object_hand: list[list[float]]
    hand: Literal["left", "right"] = "right"
    contact: bool = False

    @model_validator(mode="after")
    def matrix_is_4x4(self):
        if len(self.T_object_hand) != 4 or any(len(r) != 4 for r in self.T_object_hand):
            raise ValueError("T_object_hand must be 4x4")
        return self


class ManipulationTask(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scenario: str
    object_name: str
    goal_xy_on_table: tuple[float, float]
    keyframes: list[TaskKeyframe] = Field(min_length=1)
    source: Literal["manual", "rgb_estimated"] = "manual"


class PoseGoal(BaseModel):
    """Chosen static manipulation stance (ManipulationPoseGoal)."""

    base_xy_yaw: tuple[float, float, float]
    pelvis_height_m: float = Field(gt=0)
    waist_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    score: float
    visible_fraction: float = Field(ge=0, le=1)
    rejected: dict[str, int] = Field(default_factory=dict, description="Reject-reason histogram")


class ArrivalReport(BaseModel):
    arrived: bool
    pos_err_m: float = Field(ge=0)
    yaw_err_rad: float = Field(ge=0)
    time_s: float = Field(ge=0)
    failure: LocoManipFailure | None = None

    @model_validator(mode="after")
    def failure_iff_not_arrived(self):
        if self.arrived == (self.failure is not None):
            raise ValueError("failure must be set exactly when arrived is False")
        return self
