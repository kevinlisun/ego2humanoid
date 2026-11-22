import json

import mujoco
import numpy as np
import trimesh

from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.reconstruction.scene.export_mjcf import load_scene_objects


def test_multi_object_scene_loads_and_compiles(tmp_path):
    trimesh.creation.box((0.06, 0.04, 0.10)).export(tmp_path / "a.glb")
    trimesh.creation.cylinder(0.03, 0.12).export(tmp_path / "b.glb")
    T_a = np.eye(4)
    T_a[:3, 3] = [0.4, 0.1, 0.85]
    T_b = np.eye(4)
    T_b[:3, 3] = [0.5, -0.1, 0.86]
    (tmp_path / "objects.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"name": "a", "glb": "a.glb", "T_sim_object": T_a.tolist(), "mass_kg": 0.3},
                    {"name": "b", "glb": "b.glb", "T_sim_object": T_b.tolist()},
                ]
            }
        )
    )
    objs = load_scene_objects(tmp_path)
    assert [o.name for o in objs] == ["a", "b"]
    assert np.allclose(objs[0].pos_m, [0.4, 0.1, 0.85], atol=1e-6)
    assert np.allclose(objs[0].half_size_m, [0.03, 0.02, 0.05], atol=1e-6)
    model = build_scene(SceneConfig(table_top_z_m=0.8, objects=objs))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b") >= 0


def test_concave_object_gets_convex_pieces_and_keeps_mass(tmp_path):
    # An L-shaped (concave) solid: two boxes fused.
    a = trimesh.creation.box((0.10, 0.03, 0.03))
    b = trimesh.creation.box((0.03, 0.03, 0.10))
    b.apply_translation([-0.035, 0, 0.035])
    trimesh.util.concatenate([a, b]).export(tmp_path / "l.glb")
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.86]
    (tmp_path / "objects.json").write_text(
        json.dumps({"objects": [{"name": "l", "glb": "l.glb", "T_sim_object": T.tolist(), "mass_kg": 0.25}]})
    )
    (obj,) = load_scene_objects(tmp_path, decompose=True)
    assert len(obj.collision_pieces) >= 1
    model = build_scene(SceneConfig(table_top_z_m=0.8, objects=(obj,)))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l")
    assert abs(model.body_mass[body] - 0.25) < 1e-3


def test_textured_obj_keeps_uv_and_texture(tmp_path):
    from PIL import Image

    box = trimesh.creation.box((0.05, 0.05, 0.05))
    box.visual = trimesh.visual.TextureVisuals(
        uv=np.random.default_rng(0).random((len(box.vertices), 2)),
        image=Image.new("RGB", (8, 8), (200, 30, 30)),
    )
    box.export(tmp_path / "t.glb")
    (tmp_path / "objects.json").write_text(
        json.dumps({"objects": [{"name": "t", "glb": "t.glb", "T_sim_object": np.eye(4).tolist()}]})
    )
    (obj,) = load_scene_objects(tmp_path)
    assert obj.mesh_texture.shape == (8, 8, 3)
    assert obj.mesh_uv.shape == (len(obj.mesh_vertices), 2)
    build_scene(SceneConfig(table_top_z_m=0.8, objects=(obj,)))
