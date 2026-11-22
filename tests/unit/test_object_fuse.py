import numpy as np

from stages.reconstruction.objects.fuse import hand_depth_scale, select_held_object


def test_depth_scale_recovers_ratio():
    depth = np.full((3, 10, 20), 0.5, dtype=np.float16)
    s = hand_depth_scale(depth, np.full((3, 2), 100.0), np.full(3, 0.75), (160, 80))
    assert abs(s - 1.5) < 1e-3


def test_select_held_object_follows_pinch_not_distractor():
    n = 6
    cand = np.full((n, 2, 3), np.nan)
    cand[:, 0] = [0.4, 0.3, 0.85]  # static distractor, far from the pinch
    cand[:, 1] = [0.2, 0.0, 0.82]
    ext = np.full((n, 2, 3), 0.05)
    mid = np.tile([0.21, 0.0, 0.83], (n, 1))
    cand[3:, 1, 2] = 0.90  # carried up from the 4th detection on
    mid[3:, 2] = 0.91
    held = select_held_object(cand, ext, mid, np.ones(n, bool), 0.80)
    assert held is not None and np.allclose(held.start_xy_m, [0.21, 0.0]) and held.first_idx == 3
    assert select_held_object(cand, ext, mid, np.zeros(n, bool), 0.80) is None
    # A tray-sized box below the pinch (the whole tray segmented) must not lower the rest height.
    cand[:2, 1] = np.nan
    cand[:2, 0] = [0.21, 0.0, 0.76]
    ext[:2, 0] = [0.7, 0.15, 0.3]
    held = select_held_object(cand, ext, mid, np.ones(n, bool), 0.80)
    assert held is not None and held.first_idx == 3


def test_object_prompt_from_task_description():
    from stages.reconstruction.objects.prompt import object_prompt

    assert object_prompt("pick up a white puppy plush from the blue bowl and place it in the scoop.") == (
        "white puppy plush"
    )
    assert (
        object_prompt("pick up the blue tongs from the table with the left hand and place it.")
        == "blue tongs"
    )
    assert object_prompt("pick up an object and place it somewhere else.") == "object"
    assert object_prompt("wipe the table") == "object"


def test_prepare_mesh_rests_flat_and_aligns_thin_axis():
    import trimesh

    from stages.reconstruction.objects.mesh_asset import prepare_mesh

    box = trimesh.creation.box(extents=[0.3, 1.0, 2.0])  # arbitrary pose: long side is z
    v, f = prepare_mesh(box.vertices, box.faces, max_extent_m=0.2, pinch_axis_xy=np.array([1.0, 0.0]))
    ext = v.max(0) - v.min(0)
    assert np.isclose(ext.max(), 0.2)
    assert ext[2] < ext[:2].max()  # lies on a broad face, not standing on its end
    assert ext[0] <= ext[1] + 1e-9  # thin horizontal side along the pinch axis (x)
    assert np.allclose((v.max(0) + v.min(0)) / 2, 0, atol=1e-9)


def test_prepare_mesh_enforces_min_height_and_texture_view_fit():
    import trimesh

    from stages.reconstruction.objects.mesh_asset import prepare_mesh

    sheet = trimesh.creation.box(extents=[1.0, 0.6, 0.001])
    v, _ = prepare_mesh(sheet.vertices, sheet.faces, max_extent_m=0.2)
    assert np.isclose(v[:, 2].max() - v[:, 2].min(), 0.015)


def test_prism_hull_has_straight_sides_under_a_tapered_top():
    import trimesh

    from stages.reconstruction.objects.mesh_asset import prism_hull

    dome = trimesh.creation.icosphere(subdivisions=2).vertices * [0.045, 0.025, 0.02]
    pv, pf = prism_hull(dome)
    hull = trimesh.Trimesh(pv, pf)
    assert hull.is_watertight
    widths = [np.ptp(hull.section([0, 0, 1], [0, 0, z]).vertices[:, 1]) for z in (-0.015, 0.0, 0.015)]
    assert np.allclose(widths, 0.05, atol=2e-3)  # the dome is 0.033 wide at z=0.015


def test_slab_prism_gives_a_tapered_stick_parallel_pinched_faces():
    import trimesh

    from stages.reconstruction.objects.mesh_asset import slab_prism

    # A stick along y, 0.019 wide at y=-0.027 and 0.011 at y=+0.027, pinched across x.
    ends = ((-0.027, 0.019), (0.027, 0.011))
    wedge = np.array([[sx * w / 2, y, z] for y, w in ends for sx in (-1, 1) for z in (0, 0.04)])
    pv, pf = slab_prism(wedge, [1.0, 0.0], 0.019)
    slab = trimesh.Trimesh(pv, pf)
    assert slab.is_watertight
    for y in (-0.02, 0.0, 0.02):
        assert np.isclose(np.ptp(slab.section([0, 1, 0], [0, y, 0]).vertices[:, 0]), 0.019)
    assert np.isclose(np.ptp(pv[:, 1]), 0.054) and np.isclose(np.ptp(pv[:, 2]), 0.04)


def test_footprint_chord_width_measures_a_rhombus_at_its_centre():
    from stages.reconstruction.objects.mesh_asset import footprint_chord_width

    # Rhombus 0.16 long (x), 0.06 wide (y) at its centre.
    rhombus = np.array([[0.08, 0, 0], [0, 0.03, 0], [-0.08, 0, 0], [0, -0.03, 0], [0, 0, 0.02]])
    assert np.isclose(footprint_chord_width(rhombus, [0.0, 1.0], band_m=0.0), 0.06)
    # Off-centre pinch point: the width there, not the extent.
    assert np.isclose(footprint_chord_width(rhombus, [0.0, 1.0], [0.04, 0.0], band_m=0.0), 0.03)
    assert np.isclose(footprint_chord_width(rhombus, [1.0, 0.0], band_m=0.01), 0.16)


def test_fit_view_recovers_in_plane_rotation():
    import cv2
    import trimesh

    from stages.reconstruction.objects.mesh_texture import fit_view

    box = trimesh.creation.box(extents=[1.0, 0.3, 0.2])
    mask = np.zeros((128, 128), np.uint8)
    cv2.fillPoly(mask, [np.array([[20, 40], [100, 20], [108, 40], [28, 60]], np.int32)], 255)  # rotated bar
    rgba = np.dstack([np.zeros((128, 128, 3), np.uint8), mask])
    _, iou = fit_view(box.sample(5000), rgba)
    assert iou > 0.8


def test_physics_params_parse_and_heuristic():
    from stages.reconstruction.objects.physics_params import heuristic_physics, parse_physics_json

    assert parse_physics_json('Sure: {"mass": 0.4, "friction": 0.6}') == {"mass": 0.4, "friction": 0.6}
    assert parse_physics_json('{"mass": 50, "friction": 0.6}')["mass"] == 1.0  # clipped to liftable range
    with np.testing.assert_raises(ValueError):
        parse_physics_json('{"mass": -1, "friction": 0.6}')
    plush = heuristic_physics("white puppy plush", 1e-3)
    stapler = heuristic_physics("black stapler", 1e-3)
    assert plush["mass"] < stapler["mass"] and plush["friction"] > stapler["friction"]


def test_arm_collision_checker_rest_pose_is_clear():
    import numpy as np

    from robot.collision import ArmCollisionChecker
    from stages.manipulation.mjcf_scene import SceneConfig, build_scene

    checker = ArmCollisionChecker(build_scene(SceneConfig()))
    assert checker.colliding_frames(np.zeros((3, 43))).tolist() == [False, False, False]


def test_antipodal_axis_turns_a_corner_pinch_onto_parallel_faces():
    from stages.reconstruction.objects.mesh_asset import antipodal_axis_xy

    # A square standing on its corner: the x axis runs corner to corner (45 deg wedges).
    diamond = np.array([[0.03, 0, 0], [0, 0.03, 0], [-0.03, 0, 0], [0, -0.03, 0], [0, 0, 0.02]])
    axis, tan_alpha, width = antipodal_axis_xy(diamond, [1.0, 0.0], max_width_m=0.07)
    assert abs(abs(axis[0]) - np.sqrt(0.5)) < 0.02 and tan_alpha < 0.05
    assert np.isclose(width, 0.03 * np.sqrt(2), atol=1e-3)


def test_object_contact_overrides_the_finger_parameters():
    from stages.manipulation.mjcf_scene import BoxObject, SceneConfig, build_scene

    box = BoxObject("obj", (0.02, 0.02, 0.02), (0.4, 0.0, 0.02), friction=(0.5, 0.0, 0.0))
    m = build_scene(SceneConfig(objects=(box,)))
    g = m.body("obj").geomadr[0]
    assert m.geom_priority[g] == 1 and m.geom_condim[g] == 4
    np.testing.assert_allclose(m.geom_solref[g], (0.01, 1.0))
    assert m.geom_friction[g][0] == 1.0 and m.opt.noslip_iterations == 5
    legacy = build_scene(SceneConfig(objects=(box,), object_contact=None, noslip_iterations=0))
    g = legacy.body("obj").geomadr[0]
    assert legacy.geom_priority[g] == 0 and legacy.geom_condim[g] == 3 and legacy.geom_friction[g][0] == 0.5
