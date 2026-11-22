"""Tabletop inventory -> sim bodies: registration, roles, footprint fitting and disturbance."""

import json

import numpy as np
import pytest

from stages.manipulation.mjcf_scene import BoxObject
from stages.reconstruction.objects.place_target import CLEARANCE_M, tub_profile
from stages.reconstruction.objects.prompt import source_prompt
from stages.reconstruction.scene.inventory import (
    Footprint,
    SceneInventory,
    _heightmap_pieces,
    _surface_samples,
    body_heightmap,
    build_scene_inventory,
    disturbance,
    fit_vertices,
    footprint,
    heightmap_open,
    load_inventory,
    orient_quarter,
)


@pytest.mark.parametrize(
    ("task", "src"),
    [
        ("pick up a white puppy plush from the blue bowl and place it in the purple scoop.", "blue bowl"),
        ("pick up the rubber duck from the gray tray with the left hand and place it.", "gray tray"),
        ("pick up a black stapler from the table and place it in the box lid.", None),
    ],
)
def test_source_prompt(task, src):
    assert source_prompt(task) == src


def _box_points(center_xy, half_xy, height, yaw=0.0, n=400, seed=0):
    """Sim-frame points of the visible surface of a box standing on z=0."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (n, 2)) * half_xy
    c, s = np.cos(yaw), np.sin(yaw)
    xy = u @ np.array([[c, s], [-s, c]]) + center_xy
    z = rng.uniform(0.0, height, n)
    z[: n // 2] = height  # top face
    return np.column_stack([xy, z])


def test_footprint_recovers_rotated_box():
    p = _box_points([0.4, 0.1], np.array([0.06, 0.02]), 0.03, yaw=np.radians(30))
    fp = footprint(p[:, :2], p[:, 2])
    assert np.allclose(fp.center_xy, [0.4, 0.1], atol=0.005)
    assert np.allclose(fp.half_xy, [0.06, 0.02], atol=0.006)
    assert abs(abs(fp.axis_xy @ [np.cos(np.radians(30)), np.sin(np.radians(30))]) - 1) < 0.01
    assert fp.top_m == pytest.approx(0.03, abs=0.002)


def test_fit_vertices_matches_footprint_and_height():
    cube = np.array([[x, y, z] for x in (0, 2) for y in (0, 1) for z in (0, 0.5)], float)
    p = _box_points([0.0, 0.0], np.array([0.05, 0.02]), 0.04, yaw=np.radians(90))
    fp = footprint(p[:, :2], p[:, 2])
    v = fit_vertices(cube, fp, 0.04)
    ext = v.max(0) - v.min(0)
    # yawed by 90 deg: the long side now runs along y
    assert ext[1] == pytest.approx(0.10, abs=0.01)
    assert ext[0] == pytest.approx(0.04, abs=0.01)
    assert ext[2] == pytest.approx(0.04, abs=1e-6)


def test_fit_vertices_puts_the_mesh_sides_on_the_footprint():
    # A near-square lid with its vertices bunched along a diagonal: the principal axis runs along
    # that diagonal, not along a side, and the stretched mesh would cut across the tray walls.
    t = np.linspace(-1, 1, 200)
    diag = np.column_stack([0.05 * t, 0.045 * t, np.full_like(t, 0.01)])
    box = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.045, 0.045) for z in (0, 0.02)])
    fp = Footprint(np.zeros(2), np.array([1.0, 0.0]), np.array([0.05, 0.045]), 0.0, 0.02)
    v = fit_vertices(np.vstack([box, diag]), fp, 0.02)
    corners = v[:8, :2]
    assert np.allclose(np.abs(corners), [0.05, 0.045], atol=1e-6)


def test_drop_in_a_bowl_clears_its_inner_wall_not_the_rim():
    bowl = BoxObject("scene_bowl", (0.09, 0.08, 0.03), (0.3, 0.0, 0.83), static=True)
    fp = Footprint(np.array([0.3, 0.0]), np.array([1.0, 0.0]), np.array([0.09, 0.08]), 0.0, 0.06)
    inv = SceneInventory(receptacle=bowl, receptacle_fp=fp, receptacle_round=True, receptacle_floor_z=0.81)
    held = np.array([0.01, 0.027, 0.018])
    xy = inv.drop_xy(np.array([0.37, 0.06]), held)  # released over the bowl's footprint corner
    z, frac = tub_profile(bowl, 0.81)
    radii = np.interp(0.81 + 0.01 + held[2], z, frac) * np.array([0.09, 0.08])
    corners = xy - [0.3, 0.0] + np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * held[:2]
    assert (np.sqrt(((corners / radii) ** 2).sum(1)) <= 1.0).all()
    assert xy[0] > 0.3 and xy[1] > 0.0  # still towards where the human let go


def _write_inventory(tmp_path, objects):
    """An inventory whose camera frame *is* the sim frame (identity transform, unit scale)."""
    d = tmp_path / "scene"
    d.mkdir()
    arrays, entries = {}, []
    for i, (name, pts, *rgb) in enumerate(objects):
        arrays[f"points_{i}"] = pts
        entries.append(
            dict(
                id=i,
                name=name,
                label=name,
                rgb_median=rgb[0] if rgb else [100, 100, 100],
                crop=f"crop_{i}.png",
                points=f"points_{i}",
            )
        )
    np.savez(d / "points.npz", depth_small=np.ones((4, 4)), **arrays)
    inv = dict(frame=0, table_normal_cam=[0, 0, 1], table_offset_cam=0.0, objects=entries)
    (d / "inventory.json").write_text(json.dumps(inv))
    return load_inventory(tmp_path)


def test_build_registers_layout_to_held_object_and_assigns_roles(tmp_path):
    inv = _write_inventory(
        tmp_path,
        [
            ("stapler", _box_points([0.40, 0.00], np.array([0.08, 0.02]), 0.04)),
            ("lid", _box_points([0.40, 0.25], np.array([0.12, 0.08]), 0.03, seed=1)),
            ("cube", _box_points([0.55, -0.15], np.array([0.03, 0.03]), 0.06, seed=2)),
            ("socket", _box_points([1.60, 0.00], np.array([0.02, 0.02]), 0.02, seed=3)),
        ],
    )
    table_z = 0.8
    out = build_scene_inventory(
        inv,
        1.0,
        np.eye(4),
        table_z,
        held_xy=np.array([0.50, 0.03]),  # grip 2 cm / 1 cm outside the perceived copy
        held_phrase="black stapler",
        place_phrase="box lid",
        source_phrase=None,
        table_near_x=0.1,
    )
    rep = out.report
    assert rep["held_match"] == "stapler"
    assert np.allclose(rep["registration_xy_m"], [0.02, 0.01], atol=0.006)
    assert rep["receptacle"] == "lid"
    assert out.receptacle is not None and out.receptacle.static
    names = [o.name for o in out.objects]
    assert names == ["scene_cube"]  # held replaced by the pipeline's, far socket dropped
    cube = out.objects[0]
    # moved with the registration, standing on the table
    assert np.allclose(cube.pos_m[:2], [0.57, -0.14], atol=0.01)
    assert cube.pos_m[2] - cube.half_size_m[2] == pytest.approx(table_z, abs=0.005)
    assert not cube.static


def test_source_container_is_welded_and_sunk_to_the_table(tmp_path):
    rng = np.random.default_rng(0)
    # A bowl: an annulus rim 6 cm high around the held plush.
    ang = rng.uniform(0, 2 * np.pi, 600)
    rad = rng.uniform(0.07, 0.09, 600)
    bowl = np.column_stack([0.4 + rad * np.cos(ang), rad * np.sin(ang), rng.uniform(0.0, 0.06, 600)])
    inv = _write_inventory(
        tmp_path,
        [("bowl", bowl), ("plush", _box_points([0.4, 0.0], np.array([0.03, 0.03]), 0.05, seed=4))],
    )
    out = build_scene_inventory(
        inv, 1.0, np.eye(4), 0.8, np.array([0.4, 0.0]), "white puppy plush", None, "blue bowl", 0.1
    )
    assert out.report["source"] == "bowl"
    (b,) = out.objects
    assert b.static and not b.robot_collision  # the hand reaches past its rim to the plush
    assert b.pos_m[2] - b.half_size_m[2] < 0.8  # sunk below the table top


@pytest.mark.parametrize(
    ("spatula_rgb", "receptacle"),
    [((90, 50, 70), "spatula"), ((200, 150, 30), "scoop")],  # purple: swap; orange-ish: no swap
)
def test_task_colour_overrides_a_confused_category(tmp_path, spatula_rgb, receptacle):
    inv = _write_inventory(
        tmp_path,
        [
            ("scoop", _box_points([0.40, 0.25], np.array([0.05, 0.03]), 0.05), (214, 178, 13)),
            ("spatula", _box_points([0.45, -0.20], np.array([0.10, 0.04]), 0.03, seed=1), spatula_rgb),
            ("fork", _box_points([0.60, 0.00], np.array([0.06, 0.01]), 0.01, seed=2), (180, 60, 50)),
        ],
    )
    out = build_scene_inventory(
        inv, 1.0, np.eye(4), 0.8, np.array([0.45, 0.0]), "white puppy plush", "purple scoop", None, 0.1
    )
    assert out.report["receptacle"] == receptacle
    assert out.report["relabelled"] == ({"spatula": "scoop"} if receptacle == "spatula" else {})
    # The plush was not detected: the fork 9 cm from the grip is not taken for it.
    assert out.report["held_match"] is None
    assert np.allclose(out.report["registration_xy_m"], 0.0)


def test_hidden_held_object_is_registered_onto_its_source_floor(tmp_path):
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 2 * np.pi, 600)
    rad = rng.uniform(0.07, 0.09, 600)
    bowl = np.column_stack([0.4 + rad * np.cos(ang), rad * np.sin(ang), rng.uniform(0.0, 0.06, 600)])
    inv = _write_inventory(tmp_path, [("bowl", bowl)])
    out = build_scene_inventory(
        inv, 1.0, np.eye(4), 0.8, np.array([0.48, 0.0]), "white puppy plush", None, "blue bowl", 0.1,
        held_half_xy=np.array([0.02, 0.02]),
    )  # fmt: skip
    # Grip 8 cm off the bowl centre, i.e. in its wall: the bowl moves until the grip is on its floor.
    (b,) = out.objects
    assert 0.48 - b.pos_m[0] < 0.07 - 0.02  # inside the inner wall, with room for the plush
    assert out.report["registration_xy_m"][0] > 0.02


def test_generic_task_places_into_the_container_the_carry_lingers_over(tmp_path):
    rng = np.random.default_rng(2)
    ang = rng.uniform(0, 2 * np.pi, 600)
    rad = rng.uniform(0.07, 0.09, 600)
    bowl = np.column_stack([0.4 + rad * np.cos(ang), rad * np.sin(ang), rng.uniform(0.0, 0.06, 600)])
    inv = _write_inventory(tmp_path, [("bowl", bowl)])
    args = (inv, 1.0, np.eye(4), 0.8, np.array([0.2, 0.2]), "", None, None, 0.1)
    assert build_scene_inventory(*args).receptacle is None
    # The pinch passes over the bowl for 6 frames, then leaves before the detected release.
    carry = np.array([[0.3, 0.1]] * 3 + [[0.41, 0.01]] * 6 + [[0.2, -0.2]] * 4)
    out = build_scene_inventory(*args, carry_xy=carry)
    assert out.report["receptacle"] == "bowl" and out.report["receptacle_from"] == "carry"
    assert build_scene_inventory(*args, carry_xy=carry[:5]).receptacle is None


def test_layout_too_far_from_the_grip_assigns_no_roles(tmp_path):
    inv = _write_inventory(
        tmp_path,
        [
            ("round table", _box_points([0.40, 0.30], np.array([0.06, 0.06]), 0.04)),
            ("square table", _box_points([0.40, 0.00], np.array([0.06, 0.06]), 0.04, seed=1)),
        ],
    )
    # The grip is 35 cm from the nearest stand: the monocular layout cannot be trusted for roles.
    out = build_scene_inventory(
        inv, 1.0, np.eye(4), 0.8, np.array([0.40, -0.35]), "blue pen", "white square table",
        "white round table", 0.1,
    )  # fmt: skip
    assert out.report["unregistered"] and out.receptacle is None and not out.objects
    assert "dropped" in out.report["warning"]


def test_stand_under_the_grip_is_the_source_when_the_named_one_is_far(tmp_path):
    inv = _write_inventory(
        tmp_path,
        [
            ("round table", _box_points([0.40, 0.30], np.array([0.06, 0.06]), 0.08)),
            ("square table", _box_points([0.40, 0.00], np.array([0.06, 0.06]), 0.08, seed=1)),
        ],
    )
    # Clip 108: the named source ("round table") is 30 cm away, a stand 8 cm tall is under the grip.
    out = build_scene_inventory(
        inv, 1.0, np.eye(4), 0.8, np.array([0.40, 0.0]), "blue pen", "white square table",
        "white round table", 0.1, held_half_xy=np.array([0.02, 0.01]),
    )  # fmt: skip
    rep = out.report
    assert not rep.get("unregistered")
    assert rep["source_from"] == "grip"
    assert rep["support"]["name"] == "square table"
    assert rep["support"]["top_m"] == pytest.approx(0.08, abs=0.005)
    assert np.linalg.norm(rep["registration_xy_m"]) < 0.15
    assert rep["source_by_name"] == "round table"


def test_source_from_name_keeps_the_named_stand_within_the_registration_cap(tmp_path):
    inv = _write_inventory(
        tmp_path,
        [
            ("round table", _box_points([0.40, 0.21], np.array([0.06, 0.06]), 0.08)),
            ("square table", _box_points([0.40, 0.00], np.array([0.06, 0.06]), 0.08, seed=1)),
        ],
    )
    args = (inv, 1.0, np.eye(4), 0.8, np.array([0.40, 0.0]), "blue pen", "white square table")
    kw = dict(held_half_xy=np.array([0.02, 0.01]), source_from="name")
    # Named source "round table" (the phrase below) lies 21 cm from the grip: over the default cap ...
    out = build_scene_inventory(*args, "white round table", 0.1, **kw)  # fmt: skip
    assert out.report["unregistered"] and out.report["source_from"] == "name"
    # ... but trusted once the cap is raised.
    out = build_scene_inventory(*args, "white round table", 0.1, max_registration_m=0.25, **kw)  # fmt: skip
    assert not out.report.get("unregistered") and out.report["source_from"] == "name"
    assert out.report["support"]["name"] == "round table"


def _open_case():
    """An open case at unit scale: a 0.3-high base over x < 0.2 and a full-height lid behind it."""
    import trimesh

    base = trimesh.creation.box((1.2, 1.6, 0.3), trimesh.transformations.translation_matrix((-0.4, 0, 0.15)))
    lid = trimesh.creation.box((0.8, 1.6, 1.0), trimesh.transformations.translation_matrix((0.6, 0, 0.5)))
    m = trimesh.util.concatenate([base, lid])
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_orient_quarter_turns_the_lid_to_where_it_was_seen():
    v, f = _open_case()
    fp = Footprint(np.zeros(2), np.array([1.0, 0.0]), np.array([0.1, 0.08]), 0.0, 0.14)
    seen = fit_vertices(v, fp, 0.14, quarter=2)  # the lid seen on the other side
    obs = _surface_samples(seen, f)[::3] + [0.0, 0.0, 0.07]
    assert orient_quarter(v, f, fp, 0.14, obs) == 2
    assert orient_quarter(v, f, fp, 0.14, _surface_samples(fit_vertices(v, fp, 0.14), f) + [0, 0, 0.07]) == 0


def test_open_mesh_collision_and_drop_follow_its_heightmap():
    v, f = _open_case()
    fp = Footprint(np.array([0.4, 0.0]), np.array([1.0, 0.0]), np.array([0.1, 0.08]), 0.0, 0.14)
    verts = fit_vertices(v, fp, 0.14)
    pos = np.array([0.4, 0.0, 0.8 + 0.07])
    hm = body_heightmap(verts, f, fp, pos)
    assert heightmap_open(hm)
    pieces = _heightmap_pieces(hm, pos)
    tops = sorted({round(float(p[0][:, 2].max() + pos[2]), 3) for p in pieces})
    assert tops[0] == pytest.approx(0.8 + 0.042, abs=0.003) and tops[-1] == pytest.approx(0.94, abs=0.003)
    rec = BoxObject("scene_case", (0.1, 0.1, 0.07), tuple(pos), static=True)
    inv = SceneInventory(receptacle=rec, receptacle_fp=fp, receptacle_hmap=hm)
    half = np.array([0.02, 0.02, 0.02])
    xy = inv.drop_xy(np.array([0.45, 0.05]), half)  # the human let go over the lid
    assert xy[0] + half[0] < 0.4 + 0.02  # on the base, in front of the lid
    assert inv.drop_z(xy, half, 0.0) == pytest.approx(0.842 + 0.02 + CLEARANCE_M, abs=0.003)


def test_heightmap_surface_is_the_collision_surface():
    import trimesh

    # A tray floor with a 3 mm bump: merged into the floor's collision boxes, so read from them too.
    parts = [
        trimesh.creation.box((2.0, 1.6, 0.1), trimesh.transformations.translation_matrix((0, 0, 0.05))),
        trimesh.creation.box((0.3, 1.6, 0.13), trimesh.transformations.translation_matrix((0.1, 0, 0.065))),
        trimesh.creation.box((0.1, 1.6, 0.6), trimesh.transformations.translation_matrix((-0.95, 0, 0.3))),
    ]
    m = trimesh.util.concatenate(parts)
    fp = Footprint(np.array([0.4, 0.0]), np.array([1.0, 0.0]), np.array([0.1, 0.08]), 0.0, 0.06)
    verts = fit_vertices(np.asarray(m.vertices), fp, 0.06)
    pos = np.array([0.4, 0.0, 0.83])
    hm = body_heightmap(verts, np.asarray(m.faces), fp, pos)
    boxes = [(p[0].min(0) + pos, p[0].max(0) + pos) for p in _heightmap_pieces(hm, pos)]
    half = np.array([0.01, 0.01])
    for xy in np.random.default_rng(0).uniform([0.33, -0.06], [0.49, 0.06], (40, 2)):
        z = hm.surface_under(xy, half)[0]
        under = [hi[2] for lo, hi in boxes if np.all(lo[:2] < xy + half) and np.all(hi[:2] > xy - half)]
        assert max(under) <= z + 1e-9


def test_contents_of_an_open_mesh_start_beside_its_walls_not_in_them():
    import trimesh

    # A floor with a divider wall across it; the perceived content footprint straddles the wall.
    parts = [
        trimesh.creation.box((2.0, 1.6, 0.1), trimesh.transformations.translation_matrix((0, 0, 0.05))),
        trimesh.creation.box((0.1, 1.6, 0.6), trimesh.transformations.translation_matrix((0.3, 0, 0.3))),
    ]
    m = trimesh.util.concatenate(parts)
    fp = Footprint(np.array([0.4, 0.0]), np.array([1.0, 0.0]), np.array([0.1, 0.08]), 0.0, 0.06)
    pos = np.array([0.4, 0.0, 0.83])
    hm = body_heightmap(fit_vertices(np.asarray(m.vertices), fp, 0.06), np.asarray(m.faces), fp, pos)
    floor = hm.surface_under(np.array([0.36, 0.0]), np.array([0.005, 0.005]))[0]
    half = np.array([0.015, 0.015])
    assert hm.surface_under(np.array([0.405, 0.0]), half)[0] > floor + 0.02
    xy = hm.free_spot(np.array([0.405, 0.0]), half, floor)
    assert hm.surface_under(xy, half)[0] <= floor + 0.005
    assert np.linalg.norm(xy - [0.405, 0.0]) <= 0.015
    # Deep in the wall it is not moved into the room around it (where the held object goes).
    assert np.allclose(hm.free_spot(np.array([0.43, 0.0]), half, floor), [0.43, 0.0])


def test_container_around_the_grip_is_not_the_undetected_held_object(tmp_path):
    rng = np.random.default_rng(3)
    u = rng.uniform(-1, 1, (800, 2)) * [0.12, 0.07]
    rim = np.abs(u).max(1) > 0.9 * 0.07
    tray = np.column_stack([u + [0.33, -0.2], np.where(rim, 0.06, 0.005)])
    inv = _write_inventory(
        tmp_path, [("tupperware", tray), ("scooper", _box_points([0.5, 0.25], np.array([0.1, 0.04]), 0.04))]
    )
    out = build_scene_inventory(
        inv, 1.0, np.eye(4), 0.8, np.array([0.35, -0.18]), "wooden block", "purple scooper",
        "silver tray", 0.1, held_half_xy=np.array([0.013, 0.03]),
    )  # fmt: skip
    assert out.report["held_match"] is None
    assert out.report["source"] == "tupperware"
    assert any(o.name == "scene_tupperware" and o.static for o in out.objects)


def test_disturbance_flags_moved_free_objects():
    T = np.tile(np.eye(4), (5, 3, 1, 1))
    T[:, 1, 0, 3] = np.linspace(0, 0.05, 5)  # pushed 5 cm
    T[:, 2, 1, 3] = np.linspace(0, 0.01, 5)  # nudged 1 cm
    d = disturbance(T, ["held", "cube", "pen"], {"held"})
    assert d["disturbed"] == ["cube"]
    assert "held" not in d["per_object"]


def test_separate_pushes_overlaps_apart_but_keeps_stacks():
    from stages.reconstruction.scene.inventory import Footprint, _penetration, separate

    def fp(x, y, hx, hy, yaw=0.0):
        return Footprint(
            np.array([x, y]), np.array([np.cos(yaw), np.sin(yaw)]), np.array([hx, hy]), 0.0, 0.02
        )

    fps = {
        0: fp(0.0, 0.0, 0.05, 0.05),  # fixed (the held object's keep-out)
        1: fp(0.08, 0.0, 0.05, 0.02, yaw=0.3),  # overlaps it by ~2 cm
        2: fp(0.30, 0.0, 0.10, 0.10),  # a lid ...
        3: fp(0.30, 0.02, 0.02, 0.02),  # ... with a cube in it
    }
    shift = separate(fps, {1, 3})
    assert np.allclose(shift[0], 0) and np.allclose(shift[2], 0)
    moved = {i: Footprint(f.center_xy + shift[i], f.axis_xy, f.half_xy, 0.0, 0.02) for i, f in fps.items()}
    assert _penetration(moved[0], moved[1]) is None
    assert np.allclose(shift[3], 0)
    # The held object's keep-out over a receptacle's centre is no stack: the receptacle is pushed clear.
    held, rec = fp(0.31, 0.0, 0.03, 0.03), fps[2]
    assert np.allclose(separate({0: held, 2: rec}, {2})[2], 0)
    shift = separate({0: held, 2: rec}, {2}, never_stacked=frozenset({frozenset((0, 2))}))
    assert (
        _penetration(held, Footprint(rec.center_xy + shift[2], rec.axis_xy, rec.half_xy, 0.0, 0.02)) is None
    )
