import mujoco

from robot.floating_hand import WRIST_DOF_NAMES, write_floating_hand_xml


def test_floating_hand_xml_round_trips(tmp_path):
    for side in ("left", "right"):
        model = mujoco.MjModel.from_xml_path(str(write_floating_hand_xml(side, tmp_path)))
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)}
        assert {f"{side}_float_{d}" for d in WRIST_DOF_NAMES} <= names
        assert f"{side}_hand_thumb_0_joint" in names and f"{side}_wrist_yaw_joint" not in names
        assert model.nu == model.njnt == 13  # 6 wrist + 7 Dex3 finger joints
        for site in ("thumb_tip", "index_tip", "pinky_tip", "palm"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{site}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_{side}_index_tip") >= 0
