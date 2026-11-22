import numpy as np
import pytest

from contracts.arrays import validate_arrays
from stages.manipulation.events import detect_pick_place
from synthetic.tracks import alignment_arrays, hand_tracks, object_tracks, robot_trajectory


def test_synthetic_arrays_satisfy_contracts():
    validate_arrays("objects", object_tracks())
    validate_arrays("hands", hand_tracks())
    validate_arrays("alignment", alignment_arrays())
    validate_arrays("retarget", robot_trajectory())


def test_pick_place_detected():
    tr = object_tracks()
    ev = detect_pick_place(tr["timestamps_s"], tr["T_world_object"][:, 0])
    assert ev.success and not ev.failure_reasons
    assert ev.lift_height_m == pytest.approx(0.12, abs=1e-3)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"lift_m": 0.01}, "NO_LIFT"),
        ({"end_xy": (0.02, 0.0)}, "NO_TRANSLATION"),
    ],
)
def test_failure_reasons(kwargs, reason):
    tr = object_tracks(**kwargs)
    ev = detect_pick_place(tr["timestamps_s"], tr["T_world_object"][:, 0])
    assert not ev.success and reason in ev.failure_reasons


def test_dropped_object_not_placed_and_moving_not_settled():
    tr = object_tracks()
    T = tr["T_world_object"][:, 0].copy()
    T[-30:, 2, 3] += np.linspace(0, 0.1, 30)  # ends in the air
    ev = detect_pick_place(tr["timestamps_s"], T)
    assert "NOT_PLACED" in ev.failure_reasons and not ev.success
    T2 = tr["T_world_object"][:, 0].copy()
    T2[-30:, 0, 3] += np.linspace(0, 0.5, 30)  # still sliding
    assert "NOT_SETTLED" in detect_pick_place(tr["timestamps_s"], T2).failure_reasons


def test_invalid_trajectory():
    ev = detect_pick_place(np.arange(2.0), np.tile(np.eye(4), (2, 1, 1)))
    assert ev.failure_reasons == ("INVALID_TRAJECTORY",)


def test_translation_bar_follows_a_shorter_demonstrated_carry():
    # Clip 108: the demo carries the pen 9.4 cm, the robot reproduced 9.7 cm and failed the fixed 10 cm bar.
    tr = object_tracks(end_xy=(0.08, 0.0))
    t, T = tr["timestamps_s"], tr["T_world_object"][:, 0]
    assert "NO_TRANSLATION" in detect_pick_place(t, T).failure_reasons
    assert detect_pick_place(t, T, expected_translation_m=0.094).translated
    assert not detect_pick_place(t, T, expected_translation_m=0.2).translated


def _held(n: int, a: float, b: float) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)
    return (u >= a) & (u <= b)


@pytest.mark.parametrize(
    "contact,release_u,reason",
    [
        ((0.12, 0.86), 0.85, None),
        ("gap", 0.85, "DROPPED_IN_CARRY"),
        ((0.12, 0.5), 0.5, "AIRBORNE_RELEASE"),
        ((0.12, 1.0), 0.75, "STUCK_AFTER_RELEASE"),
        ("brush", 0.85, None),
    ],
)
def test_hand_contact_checks(contact, release_u, reason):
    tr = object_tracks()
    t = tr["timestamps_s"]
    n = len(t)
    if contact == "gap":
        c = _held(n, 0.12, 0.86) & ~_held(n, 0.45, 0.55)
    elif contact == "brush":  # a fingertip grazes the resting object on the retreat
        c = _held(n, 0.12, 0.86) | _held(n, 0.97, 0.99)
    else:
        c = _held(n, *contact)
    ev = detect_pick_place(
        t, tr["T_world_object"][:, 0], hand_contact=c, release_t_s=float(t[int(release_u * (n - 1))] - t[0])
    )
    if reason is None:
        assert ev.success and not ev.failure_reasons
    else:
        assert not ev.success and reason in ev.failure_reasons


def test_late_touch_that_drags_the_object_is_stuck():
    tr = object_tracks()
    t, T = tr["timestamps_s"], tr["T_world_object"][:, 0].copy()
    n = len(t)
    drag = _held(n, 0.92, 0.95)
    T[drag, 0, 3] += np.linspace(0.0, 0.03, drag.sum())
    T[np.flatnonzero(drag)[-1] + 1 :, 0, 3] += 0.03
    c = _held(n, 0.12, 0.86) | drag
    ev = detect_pick_place(t, T, hand_contact=c, release_t_s=float(t[int(0.85 * (n - 1))] - t[0]))
    assert "STUCK_AFTER_RELEASE" in ev.failure_reasons and ev.post_release_drag_m > 0.02


def test_thrown_object_is_ejected_even_if_it_lands_on_target():
    tr = object_tracks()
    t, T = tr["timestamps_s"], tr["T_world_object"][:, 0].copy()
    n = len(t)
    fly = _held(n, 0.4, 0.6)
    T[fly, 0, 3] += np.cumsum(fly)[fly] * 0.05  # 1.5 m/s sideways while nobody holds it
    T[np.flatnonzero(fly)[-1] + 1 :, 0, 3] += fly.sum() * 0.05
    c = _held(n, 0.12, 0.4)
    ev = detect_pick_place(t, T, hand_contact=c, release_t_s=float(t[int(0.85 * (n - 1))] - t[0]))
    assert "EJECTED" in ev.failure_reasons and ev.ejection_speed_m_s > 1.0
