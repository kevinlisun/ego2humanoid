"""The place target: parsing it out of the task, matching it in the scene, and where to let go."""

import numpy as np
import pytest

from stages.manipulation.mjcf_scene import BoxObject
from stages.reconstruction.objects.place_target import (
    TUB_BANDS,
    WALL_SEGMENTS,
    PlaceTarget,
    drop_height,
    match_object_name,
    resolve_from_detections,
    resolve_from_scene,
)
from stages.reconstruction.objects.prompt import object_prompt, place_target_prompt
from stages.retargeting.hand_eye import blend_corrections, phase_windows


@pytest.mark.parametrize(
    ("task", "held", "target"),
    [
        (
            "pick up the white block from the table with the left hand and place it on the blue bowl.",
            "white block",
            "blue bowl",
        ),
        ("pick up the mug and put it into the sink.", "mug", "sink"),
        ("pick up the red cube from the table.", "red cube", None),
    ],
)
def test_task_description_gives_both_objects(task, held, target):
    assert object_prompt(task) == held
    assert place_target_prompt(task) == target


def test_scene_match_needs_the_head_noun():
    names = ["teal_plastic_bowl", "yellow_sponge_with_beige_top", "orange_smiley_ball"]
    assert match_object_name(names, "blue bowl") == "teal_plastic_bowl"  # blue ~ teal
    assert match_object_name(names, "blue plate") is None  # no plate was reconstructed


def _ring(inner_r: float, top_z: float, floor_z: float) -> np.ndarray:
    """Bowl-like shell: a rim annulus at the top and a disc floor lower down."""
    a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    rim = np.c_[inner_r * np.cos(a), inner_r * np.sin(a), np.full_like(a, top_z)]
    floor = np.c_[0.3 * inner_r * np.cos(a), 0.3 * inner_r * np.sin(a), np.full_like(a, floor_z)]
    return np.r_[rim, floor]


def test_drop_point_goes_inside_a_container_and_on_top_of_a_solid():
    held_half = 0.02
    v = _ring(0.08, 0.06, 0.01) - np.array([0.0, 0.0, 0.035])
    bowl = BoxObject(
        "bowl", (0.08, 0.08, 0.035), (0.4, 0.0, 0.85), mesh_vertices=v, mesh_faces=np.zeros((0, 3), int)
    )
    inside = drop_height(bowl, held_half)
    assert inside == pytest.approx(0.85 - 0.025 + held_half + 0.01, abs=1e-3)  # on the inner floor

    solid = np.random.default_rng(0).uniform(-1, 1, (500, 3)) * np.array([0.08, 0.08, 0.035])
    plate = BoxObject(
        "plate", (0.08, 0.08, 0.035), (0.4, 0.0, 0.85), mesh_vertices=solid, mesh_faces=np.zeros((0, 3), int)
    )
    assert drop_height(plate, held_half) > inside  # rests on the lid, not in it


def test_resolve_from_scene_returns_a_static_body():
    v = _ring(0.08, 0.06, 0.01) - np.array([0.0, 0.0, 0.035])
    scene = (
        BoxObject(
            "teal_plastic_bowl",
            (0.08, 0.08, 0.035),
            (0.4, -0.1, 0.85),
            mesh_vertices=v,
            mesh_faces=np.zeros((0, 3), int),
        ),
        BoxObject("orange_smiley_ball", (0.03, 0.03, 0.03), (0.2, 0.2, 0.83)),
    )
    t = resolve_from_scene(scene, "blue bowl", 0.02)
    assert isinstance(t, PlaceTarget) and t.name == "teal_plastic_bowl"
    assert t.obj is not None and t.obj.static  # a shoved receptacle is not a place target
    assert t.drop_point_m[:2] == pytest.approx([0.4, -0.1])
    assert resolve_from_scene(scene, None, 0.02) is None


def test_detection_fallback_keeps_the_stationary_candidate():
    n = 10
    cand = np.zeros((n, 2, 3))
    cand[:, 0] = np.c_[np.linspace(0.2, 0.5, n), np.zeros(n), np.linspace(0.85, 1.0, n)]  # carried
    cand[:, 1] = np.array([0.45, -0.1, 0.84])  # stationary receptacle
    extent = np.full((n, 2, 3), 0.1)
    t = resolve_from_detections(cand, extent, np.ones(n, bool), np.array([0.46, -0.1, 0.95]), 0.02)
    assert t is not None and t.drop_point_m[:2] == pytest.approx([0.45, -0.1])


def test_phase_windows_hand_over_without_double_counting():
    n = 200
    w = phase_windows(n, [np.arange(20, 40), np.arange(150, 170)])
    assert w.shape == (2, n)
    assert w.sum(axis=0).max() <= 1.0 + 1e-9
    assert w[0, 30] == pytest.approx(1.0) and w[1, 30] == pytest.approx(0.0)
    assert w[1, 160] == pytest.approx(1.0) and w[0, 160] == pytest.approx(0.0)

    grasp, place = np.array([0.0, 0.0, -0.1]), np.array([0.0, 0.2, 0.05])
    off = blend_corrections(w, [grasp, place])
    assert off[30] == pytest.approx(grasp)
    assert off[160] == pytest.approx(place)
    assert off[100] == pytest.approx(np.zeros(3), abs=1e-9)  # free carry follows the human


def test_overlapping_phases_interpolate_between_the_two_frames():
    n = 60
    w = phase_windows(n, [np.arange(20, 30), np.arange(30, 40)])
    total = w.sum(axis=0)
    assert total.max() <= 1.0 + 1e-9
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    off = blend_corrections(w, [a, b])
    assert np.all(np.linalg.norm(off, axis=1) <= 1.0 + 1e-9)  # never beyond either target


def test_a_hollow_receptacle_gets_a_tub_collision_instead_of_its_convex_hull():
    # A reconstructed bowl is an open shell: not watertight, so convex decomposition degenerates to
    # a solid dome and the carried object slides off instead of landing inside.
    v = _ring(0.08, 0.06, 0.01) - np.array([0.0, 0.0, 0.035])
    bowl = BoxObject(
        "blue_bowl", (0.08, 0.08, 0.035), (0.4, 0.0, 0.85), mesh_vertices=v, mesh_faces=np.zeros((0, 3), int)
    )
    target = resolve_from_scene((bowl,), "blue bowl", held_half_z_m=0.02)
    pieces = target.obj.collision_pieces
    assert len(pieces) == 1 + TUB_BANDS * WALL_SEGMENTS  # floor + rings of sloped wall wedges
    half = np.asarray(bowl.half_size_m)
    floor_top = max(v[:, 2].max() for v, _ in pieces[:1])
    walls_bottom = min(v[:, 2].min() for v, _ in pieces[1:])
    assert floor_top == pytest.approx(walls_bottom, abs=1e-9)  # the tub has no gap at its base
    # Nothing collides in the middle above the floor: that is the volume the object is dropped into.
    for v, _ in pieces[1:]:
        assert np.abs(v[:, :2]).max(axis=0).max() > 0.5 * half[:2].max()
    assert all(np.abs(v[:, :2]).max() <= half[:2].max() + 1e-9 for v, _ in pieces)
    # The rim follows the mesh's thin rim, not a fixed fraction: no flat ledge an object can rest on.
    top = max(v[:, 2].max() for v, _ in pieces)
    rim = np.vstack([v[np.isclose(v[:, 2], top)] for v, _ in pieces[1:]])
    assert np.linalg.norm(rim[:, :2], axis=1).min() > 0.9 * half[0] * np.cos(np.pi / WALL_SEGMENTS)
