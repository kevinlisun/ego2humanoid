import copy

import pytest
import yaml

from stages.reconstruction.objects.twin_schema import AssetSpec, sha256_file, validate_asset_dir

SPEC = dict(
    id="mug_white",
    category="mug",
    synonyms=["cup"],
    colour_hsv=[[40.0, 0.05, 0.95]],
    size_m=[0.08, 0.08, 0.095],
    rest_poses=[dict(down_face="-z", probability=1.0)],
    physics=dict(density_kg_m3=1800),
    hollow=dict(inner_floor_z_m=0.01),
    licence="CC0",
    author="owner",
)


def _write(root, spec, collision=1):
    root.mkdir(parents=True)
    (root / "asset.yaml").write_text(yaml.safe_dump(spec))
    (root / "visual.glb").write_bytes(b"glb")
    (root / "collision").mkdir()
    for i in range(collision):
        (root / "collision" / f"p{i}.obj").write_text("o")


def test_valid_spec_and_directory(tmp_path):
    AssetSpec.model_validate(SPEC)
    _write(tmp_path / "mug_white", SPEC)
    assert validate_asset_dir(tmp_path / "mug_white") == []


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s["physics"].update(mass_kg=0.3),  # density and mass together
        lambda s: s.update(size_m=[0.08, 0.08, 3.0]),
        lambda s: s.update(colour_hsv=[[400.0, 0.5, 0.5]]),
        lambda s: s["rest_poses"][0].update(probability=0.5),
        lambda s: s["hollow"].update(inner_floor_z_m=0.2),
        lambda s: s.update(unknown_field=1),
    ],
)
def test_invalid_specs_rejected(edit):
    spec = copy.deepcopy(SPEC)
    edit(spec)
    with pytest.raises(ValueError):
        AssetSpec.model_validate(spec)


def test_directory_problems(tmp_path):
    root = tmp_path / "other_name"
    _write(root, SPEC, collision=33)
    spec = dict(SPEC, sha256={"visual.glb": sha256_file(root / "visual.glb")[::-1]})
    (root / "asset.yaml").write_text(yaml.safe_dump(spec))
    problems = " | ".join(validate_asset_dir(root))
    assert "differs from the directory name" in problems
    assert "exceed" in problems
    assert "sha256 mismatch" in problems
