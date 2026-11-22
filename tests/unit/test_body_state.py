import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from skills.body_state import BodyStateRecorder, export_sim_model
from stages.manipulation.mjcf_scene import BoxObject, SceneConfig, build_scene


def _model():
    box = BoxObject("cube", (0.02, 0.02, 0.02), (0.4, 0.0, 0.5), mass_kg=0.05)
    return build_scene(SceneConfig(hand="sharpa", objects=(box,), table_top_z_m=0.3))


def test_geom_world_pose_is_body_pose_times_local_pose(tmp_path):
    model = _model()
    data = mujoco.MjData(model)
    rec = BodyStateRecorder()
    for i in range(30):
        mujoco.mj_step(model, data)
        if i % 10 == 9:
            rec(model, data)
    arrays = rec.arrays()
    geoms = json.loads(export_sim_model(model, tmp_path).read_text())["geoms"]
    assert arrays["xpos"].shape == (3, model.nbody, 3)
    assert len(arrays["camera_names"]) >= 1
    names = list(arrays["body_names"])
    for g in geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g["name"])
        if gid < 0:
            continue
        b = names.index(g["body"])
        q = arrays["xquat"][-1, b]
        Rb = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        pos = Rb @ g["pos"] + arrays["xpos"][-1, b]
        assert np.allclose(pos, data.geom_xpos[gid], atol=1e-4), g["name"]


def test_export_has_robot_meshes_without_collision_proxies(tmp_path):
    model = _model()
    export_sim_model(model, tmp_path)
    info = json.loads((tmp_path / "sim_model.json").read_text())
    meshes = np.load(tmp_path / "sim_model_meshes.npz")
    mesh_geoms = [g for g in info["geoms"] if g["type"] == "mesh"]
    assert len(mesh_geoms) > 50
    assert all(f"{g['mesh']}_v" in meshes and f"{g['mesh']}_f" in meshes for g in mesh_geoms)
    assert len(info["geoms"]) < model.ngeom  # group-3 collision proxies are left out


def _run(model, steps=30, every=10):
    data = mujoco.MjData(model)
    rec = BodyStateRecorder()
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i % every == every - 1:
            rec(model, data)
    return rec.arrays()


def test_v2_pad_arrays_shapes_and_names(tmp_path):
    model = _model()
    arrays = _run(model)
    p = len(arrays["pad_geom_names"])
    assert str(arrays["version"]) == "sim_state/2"
    assert p > 0
    assert arrays["pad_force_N"].shape == (3, p) and arrays["pad_force_N"].dtype == np.float32
    assert arrays["object_contact"].shape == (3,) and arrays["object_contact"].dtype == bool
    for n in arrays["pad_geom_names"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(n)) >= 0
    pads = json.loads(export_sim_model(model, tmp_path).read_text())["pad_geoms"]
    assert [d["name"] for d in pads] == list(arrays["pad_geom_names"])


def test_pad_force_and_object_contact_on_touch():
    model = _model()
    data = mujoco.MjData(model)
    rec = BodyStateRecorder()
    mujoco.mj_forward(model, data)
    rec(model, data)  # builds the pad map
    left = next(
        g
        for g in rec._pad_ids
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g).startswith("left_index")
    )
    cube_geom = int(
        np.where(model.geom_bodyid == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"))[0][0]
    )
    # Place the cube on the pad geom so they interpenetrate, then step: a real contact force appears.
    adr = int(model.jnt_qposadr[model.body_jntadr[model.geom_bodyid[cube_geom]]])
    data.qpos[adr : adr + 3] = data.geom_xpos[left]
    mujoco.mj_forward(model, data)
    assert any({int(c.geom1), int(c.geom2)} >= {left, cube_geom} for c in data.contact[: data.ncon])
    rec(model, data)
    a = rec.arrays()
    assert a["object_contact"][-1] and not a["object_contact"][0]
    assert a["pad_force_N"][-1, rec._pad_ids.index(left)] > 0
    assert a["pad_force_N"][0].sum() == 0
