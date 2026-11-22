"""Shared enums and constants for Ego2Humanoid contracts.

T_A_B always maps coordinates from frame B into frame A. Data arrays travel as
named NPZ files; never pickle model-native classes across stage boundaries.
"""

from enum import Enum

SCHEMA_VERSION = "1.2"
ROBOT_ID = "unitree_g1_dex3_v1"
EXPECTED_G1_DEX3_DOF = 43


class StageStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"


class ReplayValidity(str, Enum):
    KINEMATIC_ONLY = "KINEMATIC_ONLY"
    PHYSICS_VALIDATED = "PHYSICS_VALIDATED"
    PHYSICS_FAILED = "PHYSICS_FAILED"


class RolloutMode(str, Enum):
    ACTION_CONTROLLED = "ACTION_CONTROLLED"
    STATE_REPLAY = "STATE_REPLAY"


class RealityCorrespondence(str, Enum):
    UNVERIFIED_MONOCULAR = "UNVERIFIED_MONOCULAR"
    METRIC_EXTERNALLY_VERIFIED = "METRIC_EXTERNALLY_VERIFIED"
