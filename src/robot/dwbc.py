"""Decoupled WBC lower-body controller (NVIDIA GR00T-WholeBodyControl) on MuJoCo models.

Thin numpy/onnxruntime port of ``decoupled_wbc/control/policy/g1_gear_wbc_policy.py`` plus the
``sim2mujoco`` loop, with joint indexing by name so it runs on any G1 MJCF (including ours).

* The two ONNX policies (``Balance`` for a ~zero velocity command, ``Walk`` otherwise) output 15
  joint targets: 12 leg joints + 3 waist joints. Observations use all 29 body joints.
* The upper body (arms) is not part of the policy: the caller supplies arm joint targets, which
  are tracked by a PD law here. Fingers are held by a separate PD law.
* Policy rate 50 Hz; the physics runs at the model timestep (decimation = 0.02 s / timestep).
Frames: base quaternion is MuJoCo ``wxyz``; the base angular velocity is in the body frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DWBC_ROBOT_DIR = (
    REPO_ROOT
    / "third_party"
    / "GR00T-WholeBodyControl"
    / "decoupled_wbc"
    / "sim2mujoco"
    / "resources"
    / "robots"
    / "g1"
)
BALANCE_ONNX = "policy/GR00T-WholeBodyControl-Balance.onnx"
WALK_ONNX = "policy/GR00T-WholeBodyControl-Walk.onnx"
CONFIG_YAML = "g1_gear_wbc.yaml"

BODY_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)  # fmt: skip
NUM_POLICY_JOINTS = 15  # legs + waist
NUM_BODY_JOINTS = len(BODY_JOINT_NAMES)
SINGLE_OBS_DIM = 86
STAND_CMD_NORM = 0.05  # below this the Balance policy runs
ARM_KP = 100.0
ARM_KD = 0.5
FINGER_KP = 2.0
FINGER_KD = 0.1
CONTROL_DT_S = 0.02
DEFAULT_ARM_Q = np.zeros(NUM_BODY_JOINTS - NUM_POLICY_JOINTS)


@dataclass(frozen=True)
class DwbcConfig:
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    cmd_scale: np.ndarray
    action_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    decimation: int
    history_len: int
    num_obs: int
    height_cmd: float
    rpy_cmd: np.ndarray


def load_config(robot_dir: Path = DWBC_ROBOT_DIR) -> DwbcConfig:
    raw = yaml.safe_load((robot_dir / CONFIG_YAML).read_text())
    arr = lambda k: np.asarray(raw[k], dtype=np.float64)  # noqa: E731
    return DwbcConfig(
        kps=arr("kps"), kds=arr("kds"), default_angles=arr("default_angles"), cmd_scale=arr("cmd_scale"),
        action_scale=float(raw["action_scale"]), ang_vel_scale=float(raw["ang_vel_scale"]),
        dof_pos_scale=float(raw["dof_pos_scale"]), dof_vel_scale=float(raw["dof_vel_scale"]),
        decimation=int(raw["control_decimation"]), history_len=int(raw["obs_history_len"]),
        num_obs=int(raw["num_obs"]), height_cmd=float(raw["height_cmd"]), rpy_cmd=arr("rpy_cmd"),
    )  # fmt: skip


def gravity_in_body(quat_wxyz: np.ndarray) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame."""
    w, x, y, z = quat_wxyz
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(w * w - x * x - y * y + z * z),
    ])  # fmt: skip


class DwbcPolicy:
    """Observation history + ONNX inference; no simulator dependency."""

    def __init__(self, robot_dir: Path = DWBC_ROBOT_DIR, cfg: DwbcConfig | None = None):
        import onnxruntime as ort

        self.cfg = cfg or load_config(robot_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._balance = ort.InferenceSession(
            str(robot_dir / BALANCE_ONNX), opts, providers=["CPUExecutionProvider"]
        )
        self._walk = ort.InferenceSession(
            str(robot_dir / WALK_ONNX), opts, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self.action = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
        self._history: deque[np.ndarray] = deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * self.cfg.history_len, maxlen=self.cfg.history_len
        )

    def single_obs(self, q29, dq29, quat_wxyz, omega_body, cmd_vel, height_m, rpy_rad) -> np.ndarray:
        c = self.cfg
        defaults = np.zeros(NUM_BODY_JOINTS)
        defaults[: len(c.default_angles)] = c.default_angles
        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:3] = np.asarray(cmd_vel, dtype=np.float64) * c.cmd_scale
        obs[3] = height_m
        obs[4:7] = rpy_rad
        obs[7:10] = np.asarray(omega_body) * c.ang_vel_scale
        obs[10:13] = gravity_in_body(np.asarray(quat_wxyz))
        obs[13 : 13 + NUM_BODY_JOINTS] = (np.asarray(q29) - defaults) * c.dof_pos_scale
        obs[13 + NUM_BODY_JOINTS : 13 + 2 * NUM_BODY_JOINTS] = np.asarray(dq29) * c.dof_vel_scale
        obs[13 + 2 * NUM_BODY_JOINTS :] = self.action
        return obs

    def act(self, q29, dq29, quat_wxyz, omega_body, cmd_vel, height_m, rpy_rad) -> np.ndarray:
        """One 50 Hz policy step; returns the 15 leg + waist joint targets (rad)."""
        self._history.append(self.single_obs(q29, dq29, quat_wxyz, omega_body, cmd_vel, height_m, rpy_rad))
        obs = np.concatenate(self._history)[None].astype(np.float32)
        sess = self._balance if np.linalg.norm(cmd_vel) <= STAND_CMD_NORM else self._walk
        self.action = sess.run(None, {sess.get_inputs()[0].name: obs})[0].squeeze().astype(np.float32)
        return self.action * self.cfg.action_scale + self.cfg.default_angles


@dataclass
class DwbcCommand:
    """Operator-level command: base velocity (vx, vy, yaw rate), pelvis height and torso rpy."""

    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    height_m: float = 0.74
    rpy_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))


class DwbcRobot:
    """MuJoCo G1 driven by the Decoupled WBC lower-body policy; arm targets come from the caller."""

    def __init__(
        self, model, data, policy: DwbcPolicy | None = None, hold_joints_q: dict[str, float] | None = None
    ):
        import mujoco

        self.mj = mujoco
        self.model, self.data = model, data
        self.policy = policy or DwbcPolicy()
        self.cfg = self.policy.cfg
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in BODY_JOINT_NAMES]
        if min(jid) < 0:
            raise ValueError("model lacks the 29 G1 body joints")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        # The policy was trained with these joint dynamics; our asset ships them as zero.
        for d in self.dadr:
            if model.dof_armature[d] == 0.0:
                model.dof_armature[d], model.dof_damping[d], model.dof_frictionloss[d] = 0.01, 0.001, 0.1
        self.ctrl_idx = np.array([self._actuator_of(j) for j in jid])
        body = set(jid)
        self.finger_jids = [j for j in range(model.njnt) if j not in body and model.jnt_type[j] != 0]
        self.finger_q = np.array(
            [hold_joints_q.get(model.joint(j).name, 0.0) if hold_joints_q else 0.0 for j in self.finger_jids]
        )
        self.finger_ctrl = np.array([self._actuator_of(j) for j in self.finger_jids])
        # Position actuators (Sharpa) take the target angle itself; torque motors (Dex3) get a PD torque.
        self.finger_is_position = np.array(
            [a >= 0 and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE for a in self.finger_ctrl]
        )
        self.finger_index = {model.joint(j).name: k for k, j in enumerate(self.finger_jids)}
        self.arm_q_target = DEFAULT_ARM_Q.copy()
        self.arm_gravity_comp = False
        self.arm_kp, self.arm_kd = ARM_KP, ARM_KD
        self.leg_waist_target = self.cfg.default_angles.copy()
        self.cmd = DwbcCommand(height_m=self.cfg.height_cmd, rpy_rad=self.cfg.rpy_cmd.copy())
        self.decimation = max(1, round(CONTROL_DT_S / model.opt.timestep))
        self._substep = 0
        self.on_step = None  # optional callback(robot) after every physics step (e.g. video capture)

    def set_fingers(self, targets: dict[str, float]) -> None:
        for name, q in targets.items():
            self.finger_q[self.finger_index[name]] = q

    def _actuator_of(self, joint_id: int) -> int:
        m = self.model
        for a in range(m.nu):
            if m.actuator_trnid[a, 0] == joint_id:
                return a
        return -1

    def read_body(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self.qadr].copy(), self.data.qvel[self.dadr].copy()

    def _policy_step(self) -> None:
        q, dq = self.read_body()
        self.leg_waist_target = self.policy.act(
            q, dq, self.data.qpos[3:7], self.data.qvel[3:6], self.cmd.vel, self.cmd.height_m, self.cmd.rpy_rad
        )

    def apply_pd(self) -> None:
        d, c = self.data, self.cfg
        q, dq = self.read_body()
        n = NUM_POLICY_JOINTS
        tau = np.empty(NUM_BODY_JOINTS)
        tau[:n] = c.kps * (self.leg_waist_target - q[:n]) - c.kds * dq[:n]
        tau[n:] = self.arm_kp * (self.arm_q_target - q[n:]) - self.arm_kd * dq[n:]
        if self.arm_gravity_comp:  # the long Sharpa hand sags 5-10 cm under pure PD at kp=100
            tau[n:] += d.qfrc_bias[self.dadr[n:]]
        d.ctrl[self.ctrl_idx] = tau
        for k, j in enumerate(self.finger_jids):
            if self.finger_is_position[k]:
                lo, hi = self.model.actuator_ctrlrange[self.finger_ctrl[k]]
                d.ctrl[self.finger_ctrl[k]] = np.clip(self.finger_q[k], lo, hi)
            elif self.finger_ctrl[k] >= 0:
                qa, da = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
                d.ctrl[self.finger_ctrl[k]] = (
                    FINGER_KP * (self.finger_q[k] - d.qpos[qa]) - FINGER_KD * d.qvel[da]
                )

    def step(self) -> None:
        """One physics step; the policy runs every ``decimation`` steps."""
        if self._substep % self.decimation == 0:
            self._policy_step()
        self._substep += 1
        self.apply_pd()
        self.mj.mj_step(self.model, self.data)
        if self.on_step is not None:
            self.on_step(self)

    def reset_pose(self, base_xy_yaw=(0.0, 0.0, 0.0), pelvis_z: float = 0.793) -> None:
        d = self.data
        self.mj.mj_resetData(self.model, d)
        x, y, yaw = base_xy_yaw
        d.qpos[0:3] = (x, y, pelvis_z)
        d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        n = NUM_POLICY_JOINTS
        d.qpos[self.qadr[:n]] = self.cfg.default_angles
        d.qpos[self.qadr[n:]] = self.arm_q_target
        self.mj.mj_forward(self.model, d)
        self.policy.reset()
        self.leg_waist_target = self.cfg.default_angles.copy()
        self._substep = 0
