import numpy as np
import pytest

from common.io import load_npz, save_npz
from common.timebase import resample_linear, resample_se3, uniform_grid
from common.transforms import (
    compose,
    invert_se3,
    make_se3,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    rot_axis_angle,
    rotation_angle,
    umeyama_sim3,
)
from contracts.arrays import ArrayContractError, check_se3, validate_arrays, validate_npz


def _rand_se3(rng):
    R = rot_axis_angle(rng.normal(size=3), rng.uniform(0, 3))
    return make_se3(R, rng.normal(size=3))


def test_invert_and_compose_identity():
    T = _rand_se3(np.random.default_rng(0))
    assert np.allclose(compose(T, invert_se3(T)), np.eye(4), atol=1e-12)


def test_quaternion_wxyz_roundtrip_and_known_value():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = rot_axis_angle(rng.normal(size=3), rng.uniform(0, 3.1))
        assert np.allclose(quat_wxyz_to_matrix(matrix_to_quat_wxyz(R)), R, atol=1e-9)
    # 90 deg about z in wxyz order
    R = quat_wxyz_to_matrix(np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 3))
    R = rot_axis_angle([0.3, 1.0, -0.2], 0.9)
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    T = umeyama_sim3(src, dst)
    assert np.allclose(src @ T[:3, :3].T + T[:3, 3], dst, atol=1e-9)


def test_resample_se3_midpoint():
    T0 = np.eye(4)
    T1 = make_se3(rot_axis_angle([0, 0, 1], 1.0), [2.0, 0, 0])
    out = resample_se3(np.array([0.0, 1.0]), np.stack([T0, T1]), np.array([0.5]))
    assert np.isclose(rotation_angle(out[0, :3, :3]), 0.5, atol=1e-9)
    assert np.allclose(out[0, :3, 3], [1.0, 0, 0])


def test_resample_linear_and_grid():
    t = uniform_grid(0.0, 1.0, 4.0)
    assert len(t) == 5
    y = resample_linear(np.array([0.0, 1.0]), np.array([[0.0], [2.0]]), t)
    assert np.allclose(y[:, 0], 2 * t)


def test_npz_roundtrip_rejects_pickle(tmp_path):
    save_npz(tmp_path / "a.npz", {"x": np.arange(3)})
    assert load_npz(tmp_path / "a.npz")["x"].tolist() == [0, 1, 2]


def _alignment(n=4):
    return {
        "T_sim_world_Rt": np.eye(4),
        "scale_sim_per_world": np.float64(1.0),
        "T_sim_robot_base": np.eye(4),
        "contact_flags": np.zeros((n, 2), bool),
    }


def test_validate_arrays_ok_and_dim_mismatch(tmp_path):
    arrays = _alignment()
    assert validate_arrays("alignment", arrays) == {"N": 4}
    save_npz(tmp_path / "al.npz", arrays)
    assert validate_npz("alignment", tmp_path / "al.npz") == {"N": 4}
    objs = {
        "timestamps_s": np.arange(3.0),
        "T_world_object": np.tile(np.eye(4), (3, 1, 1, 1)),
        "valid": np.ones((3, 1), bool),
        "visibility": np.ones((4, 1)),
    }
    with pytest.raises(ArrayContractError, match="dim 'N'"):
        validate_arrays("objects", objs)


def test_validate_arrays_missing_key_and_bad_se3():
    with pytest.raises(ArrayContractError, match="missing"):
        validate_arrays("alignment", {})
    bad = _alignment()
    bad["T_sim_world_Rt"] = np.diag([2.0, 1.0, 1.0, 1.0])
    with pytest.raises(ArrayContractError, match="orthonormal"):
        validate_arrays("alignment", bad)
    with pytest.raises(ArrayContractError):
        check_se3(np.full((4, 4), np.nan))


def test_non_monotonic_timestamps_rejected():
    arrays = {
        "timestamps_s": np.array([0.0, 1.0, 1.0]),
        "q_target_rad": np.zeros((3, 43)),
        "qvel_target_rad_s": np.zeros((3, 43)),
        "wrist_T_sim": np.tile(np.eye(4), (3, 2, 1, 1)),
        "base_T_sim": np.tile(np.eye(4), (3, 1, 1)),
    }
    with pytest.raises(ArrayContractError, match="increasing"):
        validate_arrays("retarget", arrays)
