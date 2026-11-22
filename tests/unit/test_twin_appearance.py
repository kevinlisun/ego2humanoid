from contracts.twin_appearance import ObjectAppearance, TwinAppearance, apply_edits


def test_edits_are_clamped_and_round_trip():
    app, bad = apply_edits(
        TwinAppearance(),
        [
            {"path": "objects.scene_box.shape", "value": "tray"},
            {"path": "objects.scene_box.size_m", "value": [0.25, 0.28, 9.0]},
            {"path": "objects.scene_box.color_rgb", "value": [1.4, 0.5, -1]},
            {"path": "light.sun_energy", "value": 3.5},
        ],
    )
    assert bad == []
    box = app.objects["scene_box"]
    assert box.shape == "tray" and box.size_m == (0.25, 0.28, 0.6) and box.color_rgb == (1.0, 0.5, 0.0)
    assert TwinAppearance.model_validate(app.model_dump()) == app
    assert ObjectAppearance().shape == "mesh"


def test_unknown_paths_and_bad_values_are_rejected_without_changing_anything():
    app, bad = apply_edits(
        TwinAppearance(),
        [
            {"path": "objects.scene_box.mass_kg", "value": 1},
            {"path": "robot.color", "value": 1},
            {"path": "objects.scene_box.shape", "value": "sphere"},
            {"path": "table.color_rgb", "value": "red"},
        ],
    )
    assert len(bad) == 4 and app == TwinAppearance()
