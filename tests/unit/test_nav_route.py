import numpy as np

from stages.locomotion.route import plan_route

EXT = ((0.0, 4.0), (0.0, 4.0))


def test_route_goes_around_wall():
    g = np.zeros((80, 80), bool)
    g[:60, 38:42] = True  # wall at x~2, open for y>3
    r = plan_route(g, EXT, 0.05, (0.5, 0.5), (3.5, 0.5, 1.0))
    assert r is not None and r[-1] == (3.5, 0.5, 1.0)
    assert max(y for _, y, _ in r) > 2.9


def test_route_none_when_blocked():
    g = np.zeros((80, 80), bool)
    g[:, 38:42] = True
    assert plan_route(g, EXT, 0.05, (0.5, 0.5), (3.5, 0.5, 0.0)) is None
