import numpy as np

from skills.receptacle_check import ReceptacleGeometry, check_place, check_retry

# bowl at the origin: opening 8 cm half width, wall out to 10 cm, rim 5 cm above its floor at z = 0
BOWL = ReceptacleGeometry((0.0, 0.0), (0.08, 0.08), (0.10, 0.10), rim_z_m=0.05, floor_z_m=0.0)


def _check(end, release):
    return check_place(np.array(end), np.array(release), 0.02, BOWL)


def test_inside_the_bowl_is_clean() -> None:
    c = _check((0.02, 0.0, 0.02), release=(0.0, 0.0, 0.03))
    assert c.defects() == () and c.outside_m == 0.0


def test_outside_the_wall_fires_and_reports_the_distance() -> None:
    c = _check((0.14, 0.0, 0.02), release=(0.07, 0.0, 0.03))
    assert c.defects() == ("PLACE_OUTSIDE_RECEPTACLE",)
    assert c.released_inside and abs(c.outside_m - 0.06) < 1e-9
    assert "over the opening" in c.detail()


def test_resting_on_the_wall_or_hung_on_the_rim_is_rim_blocked() -> None:
    assert _check((0.09, 0.0, 0.07), release=(0.0, 0.0, 0.03)).defects() == ("RIM_BLOCKED",)
    # clip 104: released over the opening with the underside at rim height, then pushed out
    hung = _check((0.14, 0.0, 0.02), release=(0.05, 0.0, 0.07))
    assert hung.defects() == ("PLACE_OUTSIDE_RECEPTACLE", "RIM_BLOCKED")


def test_retry_record_events() -> None:
    passes = [{"pass": 1, "frac": 1.0, "place_frac": 0.5, "bad": 0.0}]
    assert check_retry(None) is None
    assert check_retry({"outcome": "accepted", "passes": passes}) is None
    worse = check_retry({"outcome": "restored_worse", "passes": passes})
    assert worse is not None and worse.kind == "RETRY_WORSE"
    cut = check_retry({"outcome": "budget_exhausted", "budget_s": 600, "elapsed_s": 700, "passes": passes})
    assert cut is not None and cut.kind == "RETRY_BUDGET_EXHAUSTED" and "700 s of a 600 s" in cut.detail
