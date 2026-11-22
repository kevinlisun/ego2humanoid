from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from egodex.clip_reader import HAND_JOINTS_21, list_clips, load_clip

SAMPLE = Path(__file__).resolve().parents[2] / "data/raw/egodex/test/basic_pick_place/0.hdf5"


def _fake(path: Path, n=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["camera/intrinsic"] = np.array([[700.0, 0, 960], [0, 700.0, 540], [0, 0, 1]])
        names = ["camera"] + [s + j for s in ("left", "right") for j in HAND_JOINTS_21]
        for name in names:
            T = np.tile(np.eye(4), (n, 1, 1))
            T[:, :3, 3] = rng.normal(size=(n, 3))
            f["transforms/" + name] = T
        f["confidences/rightHand"] = np.ones(n)
        f.attrs["environment"] = "table:metal, hand:right"
    path.with_suffix(".mp4").write_bytes(b"x")


def test_load_synthetic_clip_shapes(tmp_path):
    _fake(tmp_path / "task" / "0.hdf5")
    clip = load_clip(next(iter(list_clips(tmp_path))))
    assert clip.joints_world.shape == (10, 2, 21, 3)
    assert clip.T_world_wrist.shape == (10, 2, 4, 4)
    assert clip.confidence.shape == (10, 2) and clip.confidence[:, 0].min() == 1.0
    assert clip.active_hand() == "right" and clip.clip_id == "task_0"
    assert clip.timestamps_s[1] == pytest.approx(1 / 30)


def test_joints_camera_inverts_camera_pose(tmp_path):
    _fake(tmp_path / "t" / "0.hdf5")
    clip = load_clip(tmp_path / "t" / "0.hdf5")
    back = np.einsum(
        "nij,nhkj->nhki",
        clip.T_world_camera,
        np.concatenate([clip.joints_camera(), np.ones((*clip.joints_camera().shape[:-1], 1))], axis=-1),
    )[..., :3]
    assert np.allclose(back, clip.joints_world)


@pytest.mark.skipif(not SAMPLE.exists(), reason="EgoDex sample not downloaded")
def test_real_sample_projects_into_image():
    clip = load_clip(SAMPLE)
    assert clip.video_path.exists() and clip.n_frames == 126
    J = clip.joints_camera()[60, 1]  # right hand, OpenCV camera convention (verified by overlay)
    K = clip.intrinsics
    u = K[0, 0] * J[:, 0] / J[:, 2] + K[0, 2]
    v = K[1, 1] * J[:, 1] / J[:, 2] + K[1, 2]
    assert (J[:, 2] > 0).all() and ((u > 0) & (u < 1920) & (v > 0) & (v < 1080)).all()
    assert np.isfinite(clip.joints_world).all()


def test_export_clip_layout(tmp_path):
    import pytest

    from contracts import EpisodeInput
    from egodex.export import export_clip

    src = Path("data/raw/egodex/test/basic_pick_place/0.hdf5")
    if not src.exists():
        pytest.skip("EgoDex sample not present")
    out = export_clip(load_clip(src), tmp_path)
    ep = EpisodeInput.model_validate_json((out / "episode.json").read_text())
    assert ep.video.video.sha256 and (out / "video.mp4").exists()
    gt = np.load(out / "gt" / "egodex_gt.npz")
    assert gt["joints_camera"].shape[2:] == (21, 3)


def test_select_accepts_single_active_hand_sample():
    from egodex.select import select_clips

    root = Path("data/raw/egodex")
    if not list(root.rglob("0.hdf5")):
        pytest.skip("EgoDex sample not present")
    best = select_clips(root)[0]
    assert best.accepted and best.active == "right" and best.other_travel_m < 0.15
