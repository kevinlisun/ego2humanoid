import numpy as np

from stages.reconstruction.align.table_fit import (
    CLEARANCE_MARGIN_M,
    clearance_lift,
    near_edge_clearance,
    table_penetration,
)
from stages.retargeting.kinematic import PELVIS_HEIGHT_M
from stages.retargeting.support_clearance import CLEARANCE_M, Support, held_object_lift, lift_above_support


def test_clearance_lift_ignores_resting_contact_and_clears_penetration():
    pen = np.zeros(40)
    pen[20] = 0.001  # resting contact
    assert not clearance_lift(pen).any()
    pen[20] = 0.02
    lift = clearance_lift(pen)
    assert lift[20] >= 0.02 + CLEARANCE_MARGIN_M
    assert lift[17] > 0  # widened in time so the hand does not jump


def test_table_penetration_detects_table_above_the_resting_arms():
    q = np.zeros((1, 43))  # dex3, arms hanging: elbows at ~0.90 m, wrists at ~0.89 m
    low = Support.from_table(0.70, (0.45, 0.0), (0.35, 0.5))
    high = Support.from_table(0.95, (0.45, 0.0), (0.35, 0.5))
    assert table_penetration(q, low, "dex3", PELVIS_HEIGHT_M).max() == 0.0
    assert (table_penetration(q, high, "dex3", PELVIS_HEIGHT_M)[0, :, 0] > 0.01).all()  # both arms


def test_near_edge_clearance_moves_the_edge_away_from_the_hanging_arms_up_to_the_cap():
    q = np.zeros((1, 43))  # dex3, arms hanging over a near edge at x = 0.10
    low = Support.from_table(0.80, (0.45, 0.0), (0.35, 0.5))
    high = Support.from_table(0.85, (0.45, 0.0), (0.35, 0.5))
    assert near_edge_clearance(q, low, "dex3", PELVIS_HEIGHT_M, 0.5) == (low.lo_xy_m[0], 0.0)
    lo_x, _ = near_edge_clearance(q, high, "dex3", PELVIS_HEIGHT_M, 0.5)
    assert lo_x > high.lo_xy_m[0] + 0.02
    capped, deficit = near_edge_clearance(q, high, "dex3", PELVIS_HEIGHT_M, 0.12)
    assert capped == 0.12 and deficit > 0.0


def test_hands_are_lifted_above_an_object_on_the_table_not_only_the_table():
    table = Support.from_table(0.8, (0.45, 0.0), (0.35, 0.5))
    bowl = Support.from_points([[0.2, -0.15, 0.8], [0.26, -0.09, 0.85]], margin_m=0.02)
    lm = np.zeros((5, 21, 3))
    lm[..., :2] = (0.23, -0.12)  # resting inside the bowl
    lm[..., 2] = 0.82
    beside = lm.copy()
    beside[..., :2] = (0.5, 0.2)
    assert np.allclose(lift_above_support(lm, table)[..., 2], 0.82)  # already clear of the table
    assert np.allclose(lift_above_support(lm, (table, bowl))[..., 2], 0.85 + CLEARANCE_M)
    assert np.allclose(lift_above_support(beside, (table, bowl))[..., 2], 0.82)


def test_carried_object_is_lifted_over_a_rim_but_not_off_its_source():
    # A block carried at 4 cm from x=0 to x=0.4 through a 6 cm rim at x=0.2; it starts on a 1 cm stand.
    n = 41
    c = np.column_stack([np.linspace(0.0, 0.4, n), np.zeros(n), np.full(n, 0.04)])
    half = np.array([0.01, 0.01, 0.01])
    rim = Support(0.06, (0.19, -0.1), (0.21, 0.1))
    stand = Support(0.01, (-0.05, -0.05), (0.05, 0.05))
    active = np.zeros(n, bool)
    active[:35] = True
    lift = held_object_lift(c, half, [rim, stand], active, 0.005)
    bottom = c[:, 2] + lift - half[2]
    over_rim = np.abs(c[:, 0] - 0.2) < 0.019
    assert np.all(bottom[over_rim] >= 0.06 + 0.005 - 1e-9)
    assert lift[0] < 1e-4 and lift[-1] < 1e-4
