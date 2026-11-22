from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import (
    ROBOT_ID,
    EpisodeInput,
    QAReport,
    ReplayValidity,
    RobotAssetSpec,
    RolloutMode,
    SimRunBundle,
    StageStatus,
    validate_physics_acceptance,
)


def test_rgb_only_episode_template():
    path = Path(__file__).resolve().parents[2] / "examples" / "episode.template.json"
    ep = EpisodeInput.model_validate_json(path.read_text())
    assert ep.robot.robot_id == ROBOT_ID
    assert ep.robot.expected_joint_dof == 43
    assert ep.video.width is None and ep.video.intrinsics_3x3 is None
    assert ep.usage_class == "noncommercial_research" and ep.physics_required is True


def test_no_extra_input_robot_types_or_fake_robot_dof():
    ep = EpisodeInput.model_validate(
        {"episode_id": "rgb_only_0001", "video": {"video": {"uri": "example.mp4"}}}
    )
    assert ep.robot.expected_joint_dof == 43
    with pytest.raises(ValidationError):
        EpisodeInput.model_validate(
            {
                "episode_id": "wrong",
                "video": {"video": {"uri": "x.mp4"}},
                "robot": {"robot_id": "unitree_g1", "expected_joint_dof": 29},
            }
        )


def test_asset_joint_mapping_43_unique():
    payload = dict(
        source_mjcf_uri="g1_dex3.xml",
        og_robot_asset_uri="g1_dex3.usd",
        source_asset_sha256="0" * 64,
        og_asset_sha256="1" * 64,
        joint_names=[f"joint{i}" for i in range(43)],
        actuator_names=[f"act{i}" for i in range(43)],
        joint_mapping_uri="robot_joint_order.json",
        fk_parity_passed=True,
    )
    model = RobotAssetSpec.model_validate(payload)
    assert model.joint_count == 43
    with pytest.raises(ValidationError):
        RobotAssetSpec.model_validate({**payload, "joint_names": ["joint0"] * 43})


def make_sim(**kw):
    p = dict(
        episode_id="e",
        manifest_uri="manifest.json",
        data_uri=".",
        frame_count=120,
        video_mp4="render.mp4",
        actual_npz="actual.npz",
        robot_asset_sha256="a" * 64,
        replay_validity=ReplayValidity.PHYSICS_VALIDATED,
        rollout_mode=RolloutMode.ACTION_CONTROLLED,
        physics_step_count=12000,
        object_driven_by_robot_contacts=True,
        state_injection_after_reset=False,
        rollouts_attempted=5,
        rollouts_succeeded=3,
        task_success=True,
        status=StageStatus.PASS,
    )
    p.update(kw)
    return SimRunBundle(**p)


def test_fake_kinematic_and_object_teleport_cannot_be_physics_validated():
    for mutation in (
        {"rollout_mode": RolloutMode.STATE_REPLAY},
        {"object_driven_by_robot_contacts": False},
        {"state_injection_after_reset": True},
        {"rollouts_succeeded": 2},
        {"physics_step_count": 0},
        {"task_success": False},
    ):
        with pytest.raises(ValidationError):
            make_sim(**mutation)


def test_successful_structural_gate_and_dataset_acceptance():
    sim = make_sim()
    report = validate_physics_acceptance(sim)
    assert report.accepted_for_dataset
    assert report.metrics["rollouts_succeeded"] == 3
    with pytest.raises(ValidationError):
        QAReport(
            episode_id="e",
            status=StageStatus.PASS,
            replay_validity=ReplayValidity.KINEMATIC_ONLY,
            accepted_for_dataset=True,
            task_success=True,
        )


def test_failed_sim_not_accepted():
    sim = make_sim(
        replay_validity=ReplayValidity.PHYSICS_FAILED,
        rollouts_succeeded=1,
        task_success=False,
        status=StageStatus.FAIL,
    )
    assert not validate_physics_acceptance(sim).accepted_for_dataset


def test_v12_oracle_bundle_fields():
    from contracts import ObjectMotionBundle, SceneBundle

    common = dict(episode_id="e", manifest_uri="m", data_uri="d", frame_count=3)
    obj = ObjectMotionBundle(
        **common, tracks_npz="o.npz", object_ids=["a"], object_pose_method="gt", input_source="oracle_gt"
    )
    assert obj.input_source == "oracle_gt"
    scene = SceneBundle(
        **common,
        scene_descriptor_uri="s.xml",
        scene_format="mjcf",
        object_mesh_uris={},
        scale_method="oracle_gt",
    )
    assert scene.input_source == "rgb_estimated"
    with pytest.raises(ValidationError):
        ObjectMotionBundle(
            **common, tracks_npz="o.npz", object_ids=["a"], object_pose_method="gt", input_source="x"
        )


def test_non_rgb_sources_cannot_enter_dataset():
    from pydantic import ValidationError

    from contracts import QAReport
    from contracts.enums import ReplayValidity, StageStatus

    kw = dict(
        episode_id="e",
        status=StageStatus.PASS,
        replay_validity=ReplayValidity.PHYSICS_VALIDATED,
        accepted_for_dataset=True,
        task_success=True,
    )
    QAReport(input_source="rgb_estimated", **kw)
    for src in ("oracle_gt", "synthetic"):
        with pytest.raises(ValidationError):
            QAReport(input_source=src, **kw)
    for obj_src in ("grasp_proxy", "synthetic"):
        with pytest.raises(ValidationError):
            QAReport(input_source="rgb_estimated", object_source=obj_src, **kw)
