import numpy as np

from skills.place_check import box_mesh, buried_fraction


def _pose(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def test_object_on_top_is_visible_and_inside_is_buried() -> None:
    rv, rf = box_mesh((0.05, 0.1, 0.02))  # receptacle top at z = 0.02
    hv, _ = box_mesh((0.01, 0.02, 0.02))
    on_top = buried_fraction(hv, _pose(0, 0, 0.04), rv, rf, _pose())
    assert on_top == 0.0
    sunk = buried_fraction(hv, _pose(0, 0, 0.0), rv, rf, _pose())
    assert sunk == 0.5  # bottom half of the corners sit under the top face; the rest touch it


def test_beside_the_receptacle_is_not_buried() -> None:
    rv, rf = box_mesh((0.05, 0.1, 0.02))
    hv, _ = box_mesh((0.01, 0.02, 0.02))
    assert buried_fraction(hv, _pose(0.3, 0, 0.0), rv, rf, _pose()) == 0.0
