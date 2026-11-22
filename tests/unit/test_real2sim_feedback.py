from dataclasses import replace

import numpy as np

from contracts.real2sim import FeedbackEvent
from skills.feedback import EventInputs, contact_runs, detect_events, merge_and_cap, phase_at, touch_times

N = 150
T_S = np.arange(N) * 0.02
PHASES = {
    "approach": (0.0, 0.4),
    "close": (0.4, 0.8),
    "hold": (0.8, 1.2),
    "carry": (1.2, 2.2),
    "release": (2.2, 2.6),
}


def _eye(*shape: int) -> np.ndarray:
    out = np.zeros((*shape, 4, 4))
    out[..., :, :] = np.eye(4)
    return out


def make_inputs(**over) -> EventInputs:
    obj = _eye(N, 2)
    obj[:, 1, :3, 3] = [0.5, 0.5, 0.0]
    contact = np.zeros((N, 2), bool)
    contact[(T_S >= 0.4) & (T_S < 2.2), 0] = True
    probe = {
        "pen_robot_table_m": np.zeros(N),
        "pen_robot_object_m": np.zeros((N, 2)),
        "pen_object_support_m": np.zeros((N, 2)),
        "force_robot_object_n": np.zeros((N, 2)),
        "obj_speed_m_s": np.zeros((N, 2)),
        "obj_ang_speed_rad_s": np.zeros((N, 2)),
        "finite": np.ones(N, bool),
        "joint_group": np.array([0, 0, 1, 1]),
    }
    x = EventInputs(
        T_S, obj, contact, _eye(N, 2), np.zeros((N, 4)), np.zeros((N, 4)), probe, ["cup", "box"], 0,
        [False, False], 0, {"failure_reasons": ()}, PHASES, 2.2,
    )  # fmt: skip
    return replace(x, **over)


def kinds(events: list[FeedbackEvent], kind: str) -> list[FeedbackEvent]:
    return [e for e in events if e.kind == kind]


def test_phase_at_and_idle() -> None:
    assert phase_at(0.5, PHASES) == "close"
    assert phase_at(2.9, PHASES) == "idle"


def test_contact_runs_merge_short_gaps() -> None:
    t = np.arange(10) * 0.02
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], bool)
    assert contact_runs(mask, t) == [(0, 5), (8, 10)]
    assert touch_times(mask, t) == (0.0, round(0.18, 2))
    assert touch_times(np.zeros(10, bool), t) is None


def test_touch_and_contact_events_of_held_and_other_object() -> None:
    x = make_inputs()
    x.object_robot_contact[(T_S >= 1.0) & (T_S < 1.2), 0] = False  # a real gap
    x.object_robot_contact[(T_S >= 1.0) & (T_S < 1.2), 1] = True
    ev = detect_events(x)
    assert [e.t_s for e in kinds(ev, "FIRST_TOUCH")] == [0.4]
    assert kinds(ev, "FIRST_TOUCH")[0].phase == "close"
    assert [e.t_s for e in kinds(ev, "LAST_TOUCH")] == [2.18]
    assert [(e.kind, e.object) for e in ev if e.kind.startswith("CONTACT")] == [
        ("CONTACT_OFF", "cup"),
        ("CONTACT_ON", "box"),
        ("CONTACT_ON", "cup"),
        ("CONTACT_OFF", "box"),
    ]


def test_lift_off_needs_sustained_height() -> None:
    x = make_inputs()
    x.object_T_sim[(T_S >= 1.0) & (T_S < 1.04), 0, 2, 3] = 0.05  # 2 frames: jitter
    assert not kinds(detect_events(x), "LIFT_OFF")
    x.object_T_sim[T_S >= 1.2, 0, 2, 3] = 0.06
    lift = kinds(detect_events(x), "LIFT_OFF")
    assert len(lift) == 1 and lift[0].t_s == 1.2 and lift[0].phase == "carry"


def test_slip_direction_in_wrist_frame_and_rotate_in_grasp() -> None:
    x = make_inputs()
    drift = np.clip((T_S - 1.3) / 0.2, 0, 1) * 0.008
    x.object_T_sim[:, 0, 1, 3] = -drift  # sim -y == wrist -y (identity wrist)
    ev = detect_events(x)
    slip = kinds(ev, "SLIP")
    assert len(slip) == 1 and "wrist -y" in slip[0].detail and slip[0].phase == "carry"
    assert not kinds(ev, "ROTATE_IN_GRASP")

    y = make_inputs()
    ang = np.clip((T_S - 1.3) / 0.2, 0, 1) * np.radians(20)
    y.object_T_sim[:, 0, 0, 0], y.object_T_sim[:, 0, 0, 1] = np.cos(ang), -np.sin(ang)
    y.object_T_sim[:, 0, 1, 0], y.object_T_sim[:, 0, 1, 1] = np.sin(ang), np.cos(ang)
    ev = detect_events(y)
    assert kinds(ev, "ROTATE_IN_GRASP") and not kinds(ev, "SLIP")


def test_no_slip_outside_contact() -> None:
    x = make_inputs()
    x.object_robot_contact[:] = False
    x.object_T_sim[T_S >= 1.3, 0, 0, 3] = 0.05
    assert not kinds(detect_events(x), "SLIP")


def test_pick_place_events() -> None:
    x = make_inputs()
    x.object_T_sim[T_S >= 1.4, 0, 2, 3] = 0.08
    x.object_robot_contact[T_S >= 1.6, 0] = False
    pp = {
        "failure_reasons": ("DROPPED_IN_CARRY", "AIRBORNE_RELEASE", "STUCK_AFTER_RELEASE"),
        "carry_gap_s": 0.6,
        "release_height_m": 0.05,
        "contact_after_release_s": 0.0,
        "post_release_drag_m": 0.02,
    }
    ev = detect_events(replace(x, pick_place=pp))
    drop = kinds(ev, "DROP")[0]
    assert drop.t_s == 1.6 and "8 cm" in drop.detail
    assert kinds(ev, "AIRBORNE_RELEASE")[0].t_s == 2.2
    assert kinds(ev, "STUCK")

    pp = {"failure_reasons": ("EJECTED",), "ejection_speed_m_s": 1.5}
    x2 = make_inputs()
    x2.object_robot_contact[:] = False
    x2.object_T_sim[(T_S >= 0.5) & (T_S < 0.7), 0, 2, 3] = 0.05
    x2.object_T_sim[T_S >= 0.5, 0, 0, 3] = np.where(T_S[T_S >= 0.5] < 0.7, (T_S[T_S >= 0.5] - 0.5) * 2, 0.4)
    eject = kinds(detect_events(replace(x2, pick_place=pp)), "EJECT")
    assert len(eject) == 1 and eject[0].phase == "close"


def test_penetration_tracking_and_grip_force() -> None:
    x = make_inputs()
    x.probe["pen_robot_table_m"][30:36] = 0.008
    x.probe["pen_robot_table_m"][50:52] = 0.02  # too short
    x.probe["pen_robot_object_m"][60:70, 0] = 0.006
    x.probe["force_robot_object_n"][(T_S >= 0.8) & (T_S < 1.2), 0] = [2.0, 3.0] * 10
    q = x.q_actual_rad.copy()
    q[70:80, 2] = 0.2  # hand joint over 0.15 rad
    q[70:80, 0] = 0.04  # arm below its limit
    ev = detect_events(replace(x, q_actual_rad=q, sized_grip_n=4.0))
    pen = kinds(ev, "PENETRATION")
    assert [(p.object, p.detail.split()[0]) for p in pen] == [(None, "robot-table"), ("cup", "robot-object")]
    track = kinds(ev, "TRACKING_ERROR")
    assert len(track) == 1 and "hand" in track[0].detail
    grip = kinds(ev, "GRIP_FORCE")[0]
    assert "median 2.5 N" in grip.detail and "sized 4.0 N" in grip.detail and grip.phase == "hold"


def test_grasp_penetration_and_start_embedded() -> None:
    x = make_inputs()
    x.probe["pen_robot_object_m"][20:24, 0] = 0.003  # 3 mm for the close window, below PENETRATION_M
    x.probe["pen_robot_object_m"][90:100, 0] = 0.003  # after the grasp: ignored
    x.probe["pen_object_support_m"][:10, 1] = 0.012
    ev = detect_events(x)
    gp = kinds(ev, "GRASP_PENETRATION")
    assert len(gp) == 1 and gp[0].object == "cup" and "3.0 mm" in gp[0].detail
    se = kinds(ev, "START_EMBEDDED")
    assert [(e.object, e.t_s) for e in se] == [("box", 0.0)]


def test_disturbed_skips_static_and_held() -> None:
    x = make_inputs()
    x.object_T_sim[T_S >= 1.0, 1, 0, 3] += 0.05
    assert kinds(detect_events(x), "DISTURBED")[0].object == "box"
    assert not kinds(detect_events(replace(x, static_flags=[False, True])), "DISTURBED")


def test_sim_unstable_truncates_other_detectors() -> None:
    x = make_inputs()
    x.probe["finite"][100:] = False
    x.object_T_sim[100:, 0] = np.nan
    ev = detect_events(x)
    assert kinds(ev, "SIM_UNSTABLE")[0].t_s == 2.0
    assert [e.t_s for e in kinds(ev, "LAST_TOUCH")] == [1.98]
    y = make_inputs()
    y.probe["obj_speed_m_s"][40, 0] = 5.0
    assert kinds(detect_events(y), "SIM_UNSTABLE")[0].t_s == 0.8


def test_merge_dedups_and_caps_by_severity() -> None:
    e = FeedbackEvent(t_s=1.0, phase="hold", kind="SLIP", object="cup", detail="a")
    assert len(merge_and_cap([e, e])) == 1
    many = [FeedbackEvent(t_s=i * 0.1, phase="hold", kind="CONTACT_ON", object=f"o{i}") for i in range(60)]
    many.append(FeedbackEvent(t_s=9.0, phase="carry", kind="DROP", object="cup"))
    out = merge_and_cap(many)
    assert len(out) == 40 and out[-1].kind == "DROP"
    assert [x.t_s for x in out] == sorted(x.t_s for x in out)
