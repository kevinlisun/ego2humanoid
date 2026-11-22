"""Probe the pinned G1+Dex3 MJCF: authoritative 43-DoF joint order, limits, and spec file."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.enums import EXPECTED_G1_DEX3_DOF

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "robots" / "g1_dex3"
MJCF_PATH = ASSET_DIR / "g1_29dof_with_hand_rev_1_0.xml"
JOINT_ORDER_PATH = ASSET_DIR / "robot_joint_order.json"
FREE_BASE_JOINT = "floating_base_joint"


@dataclass(frozen=True)
class RobotProbe:
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    joint_lower_rad: np.ndarray
    joint_upper_rad: np.ndarray


def load_fixed_base_spec(mjcf_path: Path = MJCF_PATH):
    """MjSpec of the robot with the pelvis free joint removed (fixed-pelvis variant)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    for joint in spec.joints:
        if joint.name == FREE_BASE_JOINT:
            spec.delete(joint)
    return spec


def probe_robot(mjcf_path: Path = MJCF_PATH) -> RobotProbe:
    """Joint order = actuator order; each actuator drives exactly one revolute joint."""
    import mujoco

    model = load_fixed_base_spec(mjcf_path).compile()
    if model.nu != EXPECTED_G1_DEX3_DOF:
        raise ValueError(f"expected {EXPECTED_G1_DEX3_DOF} actuators, found {model.nu}")
    joint_ids = model.actuator_trnid[:, 0]
    if len(set(joint_ids.tolist())) != model.nu:
        raise ValueError("actuators must map to distinct joints")
    if not (model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE).all():
        raise ValueError("all actuated joints must be revolute (units are radians)")
    name = lambda kind, i: mujoco.mj_id2name(model, kind, i)  # noqa: E731
    return RobotProbe(
        joint_names=tuple(name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids),
        actuator_names=tuple(name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)),
        joint_lower_rad=model.jnt_range[joint_ids, 0].copy(),
        joint_upper_rad=model.jnt_range[joint_ids, 1].copy(),
    )
