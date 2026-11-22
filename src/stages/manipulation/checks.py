"""Physical sanity checks over a rollout's ``actual.npz`` arrays."""

from dataclasses import dataclass, field

import numpy as np

from stages.manipulation.events import EventThresholds, detect_pick_place


@dataclass(frozen=True)
class CheckThresholds:
    joint_limit_tol_rad: float = 0.02
    max_tracking_rms_rad: float = 0.25
    max_joint_speed_rad_s: float = 30.0
    max_object_speed_m_s: float = 5.0


@dataclass
class CheckResult:
    passed: bool
    error_codes: list[str] = field(default_factory=list)
    metrics: dict[str, float | bool] = field(default_factory=dict)


def check_rollout(
    arrays: dict[str, np.ndarray],
    joint_limits_rad: np.ndarray,
    thresholds: CheckThresholds | None = None,
    event_thresholds: EventThresholds | None = None,
    object_index: int = 0,
) -> CheckResult:
    """Run finite / joint-limit / tracking / speed checks plus the pick-place event detector."""
    th = thresholds or CheckThresholds()
    codes: list[str] = []
    m: dict[str, float | bool] = {}

    finite = all(np.isfinite(a).all() for a in arrays.values())
    m["all_finite"] = bool(finite)
    if not finite:
        return CheckResult(False, ["NON_FINITE_STATE"], m)

    qa = arrays["q_actual_rad"]
    lo, hi = joint_limits_rad[:, 0], joint_limits_rad[:, 1]
    viol = np.maximum(lo - qa, 0).max(initial=0.0), np.maximum(qa - hi, 0).max(initial=0.0)
    m["max_joint_limit_violation_rad"] = float(max(viol))
    if max(viol) > th.joint_limit_tol_rad:
        codes.append("JOINT_LIMIT_VIOLATION")

    rms = float(np.sqrt(np.mean((arrays["q_command_rad"] - qa) ** 2)))
    m["tracking_rms_rad"] = rms
    if rms > th.max_tracking_rms_rad:
        codes.append("POOR_TRACKING")

    m["max_joint_speed_rad_s"] = float(np.abs(arrays["dq_actual_rad_s"]).max())
    if m["max_joint_speed_rad_s"] > th.max_joint_speed_rad_s:
        codes.append("UNSTABLE_SIMULATION")

    t = arrays["timestamps_s"]
    obj = arrays["object_T_sim"][:, object_index]
    speed = np.linalg.norm(np.diff(obj[:, :3, 3], axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)
    m["max_object_speed_m_s"] = float(speed.max(initial=0.0))
    if m["max_object_speed_m_s"] > th.max_object_speed_m_s:
        codes.append("OBJECT_TELEPORT_OR_EXPLOSION")

    ev = detect_pick_place(t, obj, event_thresholds or EventThresholds())
    m.update(
        lift_m=float(ev.lift_height_m), translation_m=float(ev.translation_m), task_success=bool(ev.success)
    )
    codes.extend(c for c in ev.failure_reasons if c not in codes)
    return CheckResult(not codes, codes, m)
