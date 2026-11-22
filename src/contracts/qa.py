"""Simulation run and QA report contracts."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from contracts.bundles import StageBundle
from contracts.enums import SCHEMA_VERSION, RealityCorrespondence, ReplayValidity, RolloutMode, StageStatus


class SimRunBundle(StageBundle):
    stage: Literal["simulation"] = "simulation"
    sim_backend: Literal["mujoco", "omnigibson"] = "mujoco"
    video_mp4: str
    actual_npz: str
    robot_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_validity: ReplayValidity
    rollout_mode: RolloutMode
    physics_step_count: int = Field(ge=0)
    object_driven_by_robot_contacts: bool
    state_injection_after_reset: bool
    rollouts_attempted: int = Field(ge=0)
    rollouts_succeeded: int = Field(ge=0)
    task_success: bool
    correspondence_to_real: RealityCorrespondence = RealityCorrespondence.UNVERIFIED_MONOCULAR
    # actual.npz: timestamps_s(K,), q_command_rad(K,43), q_actual_rad(K,43),
    # dq_actual_rad_s(K,43), wrist_actual_T_sim(K,2,4,4), object_T_sim(K,O,4,4).

    @model_validator(mode="after")
    def do_not_claim_fake_physics(self):
        if self.rollouts_succeeded > self.rollouts_attempted:
            raise ValueError("successes cannot exceed attempted rollouts")
        if self.replay_validity == ReplayValidity.PHYSICS_VALIDATED:
            valid = (
                self.rollout_mode == RolloutMode.ACTION_CONTROLLED
                and self.physics_step_count > 0
                and self.object_driven_by_robot_contacts
                and not self.state_injection_after_reset
                and self.rollouts_attempted >= 5
                and self.rollouts_succeeded >= 3
                and self.task_success
                and self.status == StageStatus.PASS
            )
            if not valid:
                raise ValueError(
                    "PHYSICS_VALIDATED requires successful, non-teleporting, "
                    "action-controlled seeded rollouts"
                )
        return self


class QAReport(BaseModel):
    schema_version: Literal["1.2"] = SCHEMA_VERSION
    episode_id: str
    input_source: Literal["rgb_estimated", "oracle_gt", "synthetic"] = "rgb_estimated"
    object_source: Literal["perceived", "grasp_proxy", "synthetic"] = "perceived"
    status: StageStatus
    replay_validity: ReplayValidity
    accepted_for_dataset: bool = False
    task_success: bool = False
    correspondence_to_real: RealityCorrespondence = RealityCorrespondence.UNVERIFIED_MONOCULAR
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    error_codes: list[str] = Field(default_factory=list)
    evidence_uris: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_acceptance(self):
        if self.accepted_for_dataset and (
            self.status != StageStatus.PASS
            or self.replay_validity != ReplayValidity.PHYSICS_VALIDATED
            or not self.task_success
            or self.error_codes
        ):
            raise ValueError(
                "Dataset acceptance requires PASS, PHYSICS_VALIDATED, task success and no errors"
            )
        if self.accepted_for_dataset and self.input_source != "rgb_estimated":
            raise ValueError("Only episodes driven by RGB-estimated hands may enter the dataset")
        if self.accepted_for_dataset and self.object_source != "perceived":
            raise ValueError("Only episodes with RGB-perceived objects may enter the dataset")
        return self


def validate_physics_acceptance(run: SimRunBundle) -> QAReport:
    """Structural acceptance guard; independent event validation lives in stages/qa.

    The event validator must additionally test lift, translation, settle, penetration,
    joint/torque and contact evidence before constructing a PHYSICS_VALIDATED run.
    """
    accepted = run.replay_validity == ReplayValidity.PHYSICS_VALIDATED and run.status == StageStatus.PASS
    return QAReport(
        episode_id=run.episode_id,
        status=StageStatus.PASS if accepted else StageStatus.FAIL,
        replay_validity=run.replay_validity,
        accepted_for_dataset=accepted,
        task_success=run.task_success if accepted else False,
        correspondence_to_real=run.correspondence_to_real,
        metrics={"rollouts_attempted": run.rollouts_attempted, "rollouts_succeeded": run.rollouts_succeeded},
        error_codes=[] if accepted else ["PHYSICS_ACCEPTANCE_GATE_FAILED"],
        evidence_uris=[run.video_mp4, run.actual_npz],
    )
