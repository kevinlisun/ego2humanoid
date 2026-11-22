import numpy as np
import pytest

from stages.reconstruction.align.frames import anchor_to_table, gravity_aligned_rotation
from stages.retargeting.kinematic import RIGHT_Y_FLIP, finger_curls, hand_frame, retarget_hand


def _open_right_hand_palm_down() -> np.ndarray:
    """21 landmarks of a flat right hand, fingers along +x, palm facing down (-z), index on +y."""
    lm = np.zeros((21, 3))
    base_y = {1: 0.03, 5: 0.03, 9: 0.0, 13: -0.02, 17: -0.04}
    for b, y in base_y.items():
        for k in range(4):
            lm[b + k] = [0.06 + 0.03 * k, y, 0.0]
    lm[1:5, 1] += 0.03  # thumb splayed outwards
    return lm


def test_gravity_alignment_maps_up_to_z_and_is_orthonormal():
    up_cam = np.array([0.0, -0.9, -0.4])
    R = gravity_aligned_rotation(up_cam)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    assert np.allclose(R @ (up_cam / np.linalg.norm(up_cam)), [0, 0, 1], atol=1e-9)
    assert (R @ np.array([0.0, 0.0, 1.0]))[0] > 0  # camera forward stays forward


def test_anchor_puts_fingertips_on_table_and_wrist_at_target():
    rng = np.random.default_rng(0)
    lm = rng.normal(0, 0.02, (30, 21, 3)) + [0.1, 0.2, 0.5]  # a hand jittering in place
    R = gravity_aligned_rotation(np.array([0.0, -1.0, 0.0]))
    T = anchor_to_table(R, lm, "right", table_top_z_m=0.8, target_xy_m=(0.3, 0.1), tip_clearance_m=0.0)
    sim = lm @ T[:3, :3].T + T[:3, 3]
    assert np.isclose(np.percentile(sim[:, [4, 8, 12, 16, 20], 2], 2), 0.8)
    assert np.allclose(sim[:, 0, :2].mean(0), [0.3, -0.1])  # right hand is on the robot's right (-y)


def test_anchor_ignores_a_hand_resting_below_the_table_in_the_lap():
    """Clip 108: the hand starts and ends 20 cm below the table; the work sets the table height."""
    R = gravity_aligned_rotation(np.array([0.0, -1.0, 0.0]))
    sim = np.zeros((100, 21, 3))
    sim[:, :, :2] = [0.0, 0.2]
    sim[:, :, 2] = 0.3  # resting in the lap
    sim[20:80, :, :2] = [0.3, 0.0]
    sim[20:80, :, 2] = 0.5  # working at the table
    lm = sim @ R  # sim -> camera
    T = anchor_to_table(R, lm, "right", table_top_z_m=0.8, tip_clearance_m=0.0)
    assert np.isclose((lm @ T[:3, :3].T + T[:3, 3])[50, 8, 2], 0.8)


def test_hand_frame_follows_the_chirality_of_the_robot_hand():
    """The two human hands are mirror images, so the raw index x pinky cross product flips sign.

    Both robot hands are mounted so that their two wrist frames share one convention (the G1 arms are
    not mirrored), so that flip must be preserved: for the same palm-down pose the left and right
    wrist targets have the same finger axis and opposite palm normals, and Dex3 and Sharpa agree.
    """
    right = _open_right_hand_palm_down()
    left = right * np.array([1.0, -1.0, 1.0])  # mirror in y -> left hand, same palm-down pose
    for hand in ("dex3", "sharpa"):
        r, lf = hand_frame(right, "right", hand), hand_frame(left, "left", hand)
        assert np.allclose(r[:, 0], lf[:, 0], atol=0.05)  # same finger axis
        assert np.allclose(r[:, 1], -lf[:, 1], atol=0.05)  # opposite palm normal
    assert np.allclose(hand_frame(right, "right", "sharpa"), hand_frame(right, "right", "dex3"), atol=0.05)


@pytest.mark.parametrize("hand", ["dex3", "sharpa"])
def test_right_y_flip_matches_the_compiled_hand(hand):
    """Re-derive RIGHT_Y_FLIP from the model: run hand_frame's formula on the robot's own geometry."""
    mujoco = pytest.importorskip("mujoco")
    from stages.manipulation.mjcf_scene import SceneConfig, build_scene

    mcp = {  # wrist, middle MCP, index MCP, and the most ulnar finger the hand has
        "sharpa": ("{s}_wrist_yaw_link", "{s}_middle_PP", "{s}_index_PP", "{s}_pinky_PP"),
        "dex3": (
            "{s}_wrist_yaw_link",
            "{s}_hand_middle_0_link",
            "{s}_hand_index_0_link",
            "{s}_hand_thumb_0_link",
        ),
    }[hand]
    model = build_scene(SceneConfig(hand=hand))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    def y_local(side):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n.format(s=side)) for n in mcp]
        w, mid, idx, ulnar = (data.xpos[i] for i in ids)
        x = (mid - w) / np.linalg.norm(mid - w)
        y = np.cross(idx - w, ulnar - w)
        y -= (y @ x) * x
        return (data.xmat[ids[0]].reshape(3, 3).T @ (y / np.linalg.norm(y)))[1]

    assert y_local("left") > 0.9  # the left hand always defines the convention
    assert RIGHT_Y_FLIP[hand] == (y_local("right") < 0)


def test_hand_frame_and_curl_on_flat_hand():
    lm = _open_right_hand_palm_down()
    R = hand_frame(lm, "right", "sharpa")
    assert np.allclose(R[:, 0], [1, 0, 0], atol=0.05)  # x along fingers
    assert R[2, 1] < -0.9  # y comes out of the right palm, so it faces down with the palm
    assert np.allclose(R, hand_frame(lm, "right", "dex3"), atol=0.05)  # both hands share a convention
    assert np.linalg.det(R) > 0
    assert finger_curls(lm[None])[0, 1:].max() < 0.2
    fist = lm.copy()
    for chain in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        for k, idx in enumerate(chain):
            ang = 0.9 * k
            fist[idx] = fist[chain[0]] + 0.03 * k * np.array([np.cos(ang), 0, -np.sin(ang)])
    assert finger_curls(fist[None])[0, 1:].min() > 0.4


def test_retarget_reaches_reachable_wrist_and_orients_palm_down():
    lm = _open_right_hand_palm_down()
    lm = lm + [0.30, -0.15, 0.85] - lm[0]
    res = retarget_hand(np.stack([lm, lm + [0.0, 0.02, 0.0]]), "right")
    assert res.pos_err_m.max() < 1e-2 and res.rot_err_rad.max() < 0.05
    assert np.isfinite(res.q_target_rad).all() and res.q_target_rad.shape == (2, 43)


def test_pinch_leveling_is_scaled_back_where_it_would_leave_the_workspace(monkeypatch):
    from scipy.spatial.transform import Rotation

    import stages.retargeting.kinematic as kin

    # A leveling turn that swings the wrist out of reach (Dex3 clip 0: 47% of frames unreachable).
    monkeypatch.setattr(kin, "turn_rotation", lambda *a: Rotation.from_euler("y", 2.0).as_matrix())
    lm = _open_right_hand_palm_down()
    lm = lm + [0.30, -0.15, 0.85] - lm[0]
    res = retarget_hand(lm[None], "right", pinch_gap_m=np.array([0.03]))
    assert res.pos_err_m.max() < 0.01


def test_turn_share_ramps_into_an_unreachable_dip_instead_of_flipping():
    from stages.retargeting.kinematic import rate_limited_share

    # Per-frame choice on Dex3 clip 101 flipped 1 -> 0 -> 0.25 and swung the wrist 15 cm in two frames.
    share = np.ones(41)
    share[20], share[21] = 0.0, 0.25
    s = rate_limited_share(share, 0.05)
    assert np.all(s <= share + 1e-12)
    assert np.abs(np.diff(s)).max() <= 0.05 + 1e-12
    assert s[20] == 0.0 and s[0] == 1.0 and s[-1] == 1.0


def test_left_retarget_only_drives_left_arm_and_hand():
    lm = _open_right_hand_palm_down() * [1, -1, 1]  # mirrored: a left hand
    lm = lm + [0.30, 0.15, 0.85] - lm[0]
    q = retarget_hand(lm[None], "left").q_target_rad[0]
    assert np.abs(q[15:29]).max() > 0.1  # left arm + hand
    assert np.abs(q[29:]).max() == 0.0 and np.abs(q[:15]).max() == 0.0


def test_camera_view_never_buys_a_standing_spot_with_unreachable_frames():
    from stages.locomotion.base_placement import Placement, select_placement

    T = np.eye(4)
    reach_all = Placement(T, (0.25, 0.1), 0.001, 0.0)
    # 7 mm mean error is within the score slack, but 6% of the frames (the grasp) are out of reach.
    nicer_view = Placement(T, (0.30, 0.1), 0.007, 0.06)
    pick = select_placement([(0.001, 0.5, reach_all), (0.007, 0.1, nicer_view)])
    assert pick is reach_all
    slightly_worse = Placement(T, (0.25, 0.05), 0.004, 0.0)
    assert select_placement([(0.001, 0.5, reach_all), (0.004, 0.1, slightly_worse)]) is slightly_worse


@pytest.mark.parametrize("axis", [(1.0, 0.0, 1.0), (0.3, -0.5, -0.4), (-0.2, 0.7, 0.05)])
def test_level_rotation_turns_a_tilted_pinch_axis_horizontal_keeping_its_heading(axis):
    from stages.retargeting.kinematic import level_rotation

    v = np.array(axis)
    R = level_rotation(v)
    w = R @ v
    assert np.allclose(R @ R.T, np.eye(3))
    assert abs(w[2]) < 1e-9
    assert np.allclose(w[:2] / np.linalg.norm(w[:2]), v[:2] / np.linalg.norm(v[:2]))
    half = level_rotation(v, 0.5) @ v  # partial weight: half the elevation
    elev = np.arctan2(v[2], np.linalg.norm(v[:2]))
    assert np.isclose(np.arctan2(half[2], np.linalg.norm(half[:2])), elev / 2)


def test_a_grip_run_whose_roll_deepened_the_table_contact_is_reverted():
    from stages.retargeting.clip import _worse_roll_runs

    grip = np.array([0, 1, 1, 1, 0, 1, 1, 0], bool)
    prev_roll = np.zeros(8)
    roll = np.where(grip, 0.5, 0.0)
    prev_pen = np.array([0, 0.012, 0.012, 0.012, 0, 0.03, 0.03, 0])
    pen = np.array([0, 0.03, 0.025, 0.0, 0, 0.0, 0.01, 0])  # first run: thumb base sank; second: palm cleared
    assert _worse_roll_runs(grip, roll, prev_roll, pen, prev_pen).tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert not _worse_roll_runs(grip, prev_roll, prev_roll, pen, prev_pen).any()  # unchanged roll is kept


def test_retarget_cache_resumes_bit_identically_after_a_changed_frame():
    n = 14
    base = np.tile(_open_right_hand_palm_down(), (n, 1, 1)) + [0.35, -0.1, 0.9]
    base += np.linspace(0, 0.05, n)[:, None, None] * [0.0, 1.0, 0.0]
    gaps = np.full(n, np.nan)
    gaps[4:9] = 0.03
    changed = base.copy()
    changed[8:] += [0.0, 0.0, 0.02]  # only the frames from 8 on differ
    cache: dict = {}
    retarget_hand(base, "right", pinch_gap_m=gaps, cache=cache)
    resumed = retarget_hand(changed, "right", pinch_gap_m=gaps, cache=cache)
    fresh = retarget_hand(changed, "right", pinch_gap_m=gaps)
    assert np.array_equal(resumed.q_target_rad, fresh.q_target_rad)
    assert np.array_equal(resumed.pos_err_m, fresh.pos_err_m)
    again = retarget_hand(changed, "right", pinch_gap_m=gaps, cache=cache)  # identical input: full reuse
    assert np.array_equal(again.q_target_rad, fresh.q_target_rad)
