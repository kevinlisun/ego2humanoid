"""Ego2Humanoid boundary contracts (Pydantic models and Protocols)."""

from contracts.bundles import (
    AlignedInteractionBundle,
    HandTrackBundle,
    ObjectMotionBundle,
    RobotTrajectoryBundle,
    SceneBundle,
    StageBundle,
    TactileBundle,
)
from contracts.enums import (
    EXPECTED_G1_DEX3_DOF,
    ROBOT_ID,
    SCHEMA_VERSION,
    RealityCorrespondence,
    ReplayValidity,
    RolloutMode,
    StageStatus,
)
from contracts.episode import ArtifactRef, EpisodeInput, RobotRequest, VideoSpec
from contracts.protocols import (
    HandEstimator,
    ObjectMotionEstimator,
    QualityGate,
    Retargeter,
    SceneReconstructor,
    Simulator,
    WorldAligner,
)
from contracts.qa import QAReport, SimRunBundle, validate_physics_acceptance
from contracts.robot import RobotAssetSpec

__all__ = [
    "EXPECTED_G1_DEX3_DOF",
    "ROBOT_ID",
    "SCHEMA_VERSION",
    "AlignedInteractionBundle",
    "ArtifactRef",
    "EpisodeInput",
    "HandEstimator",
    "HandTrackBundle",
    "ObjectMotionBundle",
    "ObjectMotionEstimator",
    "QAReport",
    "QualityGate",
    "RealityCorrespondence",
    "ReplayValidity",
    "Retargeter",
    "RobotAssetSpec",
    "RobotRequest",
    "RobotTrajectoryBundle",
    "RolloutMode",
    "SceneBundle",
    "SceneReconstructor",
    "SimRunBundle",
    "Simulator",
    "StageBundle",
    "StageStatus",
    "TactileBundle",
    "VideoSpec",
    "WorldAligner",
    "validate_physics_acceptance",
]
