import mujoco
import numpy as np
import trimesh

from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.reconstruction.objects.twin_asset import asset_mass_kg, load_twin_asset
from stages.reconstruction.objects.twin_import import import_twin, load_mesh, mesh_report, normalise_mesh
from stages.reconstruction.objects.twin_schema import load_asset_spec, validate_asset_dir


def _cup_glb(path):
    """A Y-up open cup (annulus wall + floor), off-origin, in centimetres: what a raw export looks like."""
    wall = trimesh.creation.annulus(r_min=3.5, r_max=4.0, height=9.0)
    floor = trimesh.creation.cylinder(
        radius=4.0, height=0.5, transform=trimesh.transformations.translation_matrix([0, 0, -4.25])
    )
    cup = trimesh.util.concatenate([wall, floor])
    cup.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # Z-up -> Y-up
    cup.apply_translation([10, 20, 30])
    cup.export(path)


def _import(tmp_path, **kw):
    src = tmp_path / "cup.glb"
    _cup_glb(src)
    args = dict(
        asset_id="cup",
        category="cup",
        size_m=(0.08, 0.08, 0.09),
        colour_hsv=[(200.0, 0.3, 0.8)],
        physics={"density_kg_m3": 600.0, "friction": 0.8},
        licence="CC0",
        author="test",
        hollow_inner_floor_z_m=0.005,
        max_pieces=8,
    )
    return import_twin(src, tmp_path / "cup", **{**args, **kw})


def test_normalise_gives_z_up_metres_and_bottom_centre(tmp_path):
    src = tmp_path / "cup.glb"
    _cup_glb(src)
    mesh = normalise_mesh(load_mesh(src), (0.08, 0.08, 0.09))
    lo, hi = mesh.bounds
    assert np.allclose(hi - lo, (0.08, 0.08, 0.09), atol=1e-6)
    assert np.allclose([lo[0] + hi[0], lo[1] + hi[1], lo[2]], 0, atol=1e-6)


def test_import_writes_a_valid_package(tmp_path):
    report = _import(tmp_path)
    assert report["validation"] == []
    assert 1 <= report["collision_pieces"] <= 8
    assert validate_asset_dir(tmp_path / "cup") == []
    assert load_asset_spec(tmp_path / "cup" / "asset.yaml").hollow.inner_floor_z_m == 0.005


def test_open_mesh_reports_hull_fallback():
    open_cup = trimesh.creation.annulus(r_min=0.03, r_max=0.04, height=0.09)
    open_cup.update_faces(np.arange(len(open_cup.faces)) % 7 != 0)
    report = mesh_report(open_cup)
    assert not report["watertight"] and any("convex hull" in w for w in report["warnings"])


def test_asset_becomes_a_simulated_object_with_density_mass(tmp_path):
    _import(tmp_path)
    obj = load_twin_asset(tmp_path / "cup", "cup", (0.4, 0.0, 0.8))
    mesh = trimesh.load(tmp_path / "cup" / "visual.glb", force="mesh")
    assert obj.mass_kg == asset_mass_kg(load_asset_spec(tmp_path / "cup" / "asset.yaml"), mesh)
    assert 0.02 < obj.mass_kg < 0.2 and obj.friction[0] == 0.8
    assert load_twin_asset(tmp_path / "cup", "cup", (0, 0, 0), friction=1.2, mass_kg=0.3).mass_kg == 0.3

    model = build_scene(SceneConfig(objects=(obj,)))
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_texture_uv_v_runs_from_the_image_top():
    from PIL import Image

    from stages.reconstruction.objects.twin_asset import _texture

    quad = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    img = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
    quad.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [0, 1]], material=material)
    uv, tex = _texture(quad)
    assert np.allclose(uv, [[0, 1], [1, 1], [0, 0]]) and tex.shape == (4, 4, 3)
