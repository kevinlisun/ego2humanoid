import numpy as np
import pytest

from contracts import TactileBundle
from contracts.arrays import ArrayContractError, validate_arrays
from stages.reconstruction.hands import tactile as tc
from stages.reconstruction.hands.backend import HandsConfig, artefact_names, load_hands_config

V = 778


def _synthetic(n=12):
    rng = np.random.default_rng(0)
    region = np.arange(V) % tc.N_REGIONS
    force = rng.random((n, 2, V))
    prob = np.zeros((n, 2, V))
    prob[:, 1, region == 1] = 0.9
    return force, prob, region


def test_defaults_and_validation(tmp_path, monkeypatch):
    monkeypatch.delenv("E2H_TACTILE_BACKEND", raising=False)
    cfg = load_hands_config()
    assert cfg == HandsConfig("fah", "none", "tactile_egotac.npz")
    assert artefact_names(cfg, 700) == {"pose": "hands_fah_f700.npz"}
    monkeypatch.setenv("E2H_TACTILE_BACKEND", "egotac")
    assert artefact_names(load_hands_config(), 700)["tactile"] == "tactile_egotac.npz"
    assert load_hands_config(tactile_override="none").tactile_enabled is False
    bad = tmp_path / "h.yaml"
    bad.write_text("backend: fah\ntactile:\n  backend: nope\n")
    with pytest.raises(ValueError):
        load_hands_config(bad, tactile_override="nope")
    bad.write_text("backend: other\n")
    with pytest.raises(ValueError):
        load_hands_config(bad)
    bad.write_text("tactile:\n  npz_name: ../x.npz\n")
    with pytest.raises(ValueError):
        load_hands_config(bad)


def test_tactile_contract():
    f, p, _ = _synthetic()
    validate_arrays("tactile", {"force": f, "contact_prob": p})
    with pytest.raises(ArrayContractError):
        validate_arrays("tactile", {"force": f, "contact_prob": p + 2.0})
    with pytest.raises(ArrayContractError):
        validate_arrays("tactile", {"force": f[..., :10], "contact_prob": p[..., :10]})
    with pytest.raises(ValueError):
        TactileBundle(
            episode_id="e",
            manifest_uri="m",
            data_uri="d",
            frame_count=1,
            tactile_npz="t",
            source="egotac",
            vertex_count=100,
        )


def test_aggregate_sums_and_modes():
    f, p, region = _synthetic()
    fr, pr = tc.aggregate_regions(f, p, region)
    assert np.allclose(fr[:, 0, 3], f[:, 0, region == 3].sum(-1))
    assert np.allclose(pr[:, 1, 1], 0.9) and np.allclose(pr[:, 0], 0)
    _, pm = tc.aggregate_regions(f, np.full_like(p, 0.2), region, "mean")
    assert np.allclose(pm, 0.2)
    region2 = region.copy()
    region2[region2 == 3] = -1
    fr2, _ = tc.aggregate_regions(f, p, region2)
    assert np.all(fr2[..., 3] == 0)
    with pytest.raises(ValueError):
        tc.aggregate_regions(f, p, np.zeros(10, int))


def test_skinning_regions():
    w = np.zeros((V, 16))
    w[:, 0] = 1.0
    w[100:200, 13] = 1.0
    w[100:200, 0] = 0.0
    w[300:320, 10] = 1.0
    w[300:320, 0] = 0.0
    r = tc.region_of_vertex_from_skinning(w)
    assert r[150] == tc.REGION_NAMES.index("thumb_pad")
    assert r[310] == tc.REGION_NAMES.index("ring_pad")
    assert r[0] == tc.REGION_NAMES.index("palm")
    assert [r[v] for v in tc.FINGERTIP_VERTEX_IDS] == [0, 1, 2, 3, 4]


def test_smoothing_flags_and_fusion():
    x = np.zeros((9, 2, 1))
    x[4] = 1.0  # single-frame spike
    assert float(tc.smooth_time(x, "median", 3).max()) == 0.0
    ema = tc.smooth_time(x, "ema", alpha=0.5)
    assert ema[4, 0, 0] == 0.5 and ema[5, 0, 0] == 0.25
    with pytest.raises(ValueError):
        tc.smooth_time(x, "median", 4)
    f, p, region = _synthetic()
    out = tc.compute_tactile_regions(f, p, region, window=3, threshold=0.5)
    flags = out["tactile_contact_flags"]
    assert flags.shape == (12, 2, tc.N_REGIONS) and flags.dtype == bool
    assert flags[:, 1, 1].all() and not flags[:, 0].any()
    geo = np.zeros((12, 2), bool)
    geo[0, 0] = True
    assert tc.fuse_with_geometric(flags, geo, "or")[0, 0]
    assert not tc.fuse_with_geometric(flags, geo, "and").any()
    assert tc.fuse_with_geometric(flags, geo, "tactile_any")[:, 1].all()


def test_write_roundtrip(tmp_path):
    f, p, region = _synthetic()
    path = tc.write_tactile_regions(
        tmp_path / "tactile_regions.npz", tc.compute_tactile_regions(f, p, region)
    )
    with np.load(path, allow_pickle=False) as d:
        assert list(d["region_names"]) == list(tc.REGION_NAMES)
        validate_arrays("tactile_regions", {k: d[k] for k in d.files if k != "region_names"})
