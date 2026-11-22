import json

import numpy as np
import pytest

from skills.asset_provenance import FALLBACK_BOX, FROM_SCENE, provenance_events
from stages.retargeting.held_mesh import HeldMeshMissing, resolve_held_asset, scene_held_mesh


def _clip(tmp_path, centroid=(0.1, 0.0, 1.0), with_mesh=True):
    scene = tmp_path / "scene"
    scene.mkdir()
    np.savez(
        tmp_path / "objects_sam3.npz",
        score=np.array([[0.2, 0.9]]),
        centroid_cam=np.array([[[5.0, 5.0, 1.0], centroid]]),
    )
    (tmp_path / "intrinsics_est.json").write_text(
        json.dumps({"focal_px": 1000.0, "width": 1000, "height": 800})
    )
    boxes = {0: [0, 0, 100, 100], 1: [550, 380, 700, 450], 2: [400, 300, 900, 500]}  # 1 is inside 2
    objs = [
        {"id": i, "box_xyxy": b, "mesh": f"mesh_{i}.glb", "crop": f"crop_{i}.png"} for i, b in boxes.items()
    ]
    (scene / "inventory.json").write_text(json.dumps({"objects": objs}))
    for i in boxes if with_mesh else ():
        (scene / f"mesh_{i}.glb").write_bytes(b"x")
    return tmp_path


def test_scene_mesh_is_the_smallest_box_around_the_projected_sam3_centroid(tmp_path) -> None:
    mesh, crop = scene_held_mesh(_clip(tmp_path))  # (0.1, 0, 1) -> pixel (600, 400)
    assert mesh.name == "mesh_1.glb" and crop is None


def test_no_scene_match_when_centroid_misses_every_box(tmp_path) -> None:
    assert scene_held_mesh(_clip(tmp_path, centroid=(-0.45, 0.35, 1.0))) is None


def test_resolve_prefers_crop_then_scene_and_fails_loudly_without_fallback(tmp_path) -> None:
    clip = _clip(tmp_path, with_mesh=False)
    with pytest.raises(HeldMeshMissing, match="HELD_MESH_MISSING|allow_held_box_fallback"):
        resolve_held_asset(clip, allow_box=False)
    assert HeldMeshMissing.code == "HELD_MESH_MISSING"
    assert resolve_held_asset(clip, allow_box=True) == (None, None, {"source": "box"})
    (clip / "scene" / "mesh_1.glb").write_bytes(b"x")
    assert resolve_held_asset(clip, False)[2] == {"source": "scene", "mesh": "mesh_1.glb"}
    (clip / "object_mesh_raw.glb").write_bytes(b"x")
    assert resolve_held_asset(clip, False)[0] == clip / "object_mesh_raw.glb"


def test_provenance_events() -> None:
    assert [e.kind for e in provenance_events({"source": "box"}, "h")] == [FALLBACK_BOX]
    assert [e.kind for e in provenance_events({"source": "scene", "mesh": "m.glb"}, "h")] == [FROM_SCENE]
    assert provenance_events({"source": "crop"}, "h") == provenance_events(None, "h") == []


def test_hf_weights_preflight_names_the_variable(tmp_path) -> None:
    from stages.reconstruction.objects.hf_weights import cached_model, require_model

    env = {"HF_HOME": str(tmp_path)}
    with pytest.raises(SystemExit, match="set HF_HOME"):
        require_model("facebook/sam3", "SAM3", env)
    (tmp_path / "hub" / "models--facebook--sam3" / "snapshots" / "abc").mkdir(parents=True)
    assert cached_model("facebook/sam3", env)
    assert not cached_model("facebook/sam3", {"HF_HOME": str(tmp_path), "HF_HUB_CACHE": str(tmp_path / "x")})
