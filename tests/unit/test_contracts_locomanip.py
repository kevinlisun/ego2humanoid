import pytest
from pydantic import ValidationError

from contracts.locomanip import ArrivalReport, LocoManipFailure, ManipulationTask, TaskKeyframe

I4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def test_arrival_report_requires_failure_when_not_arrived():
    ArrivalReport(arrived=True, pos_err_m=0.02, yaw_err_rad=0.01, time_s=9)
    ArrivalReport(
        arrived=False, pos_err_m=0.5, yaw_err_rad=0.1, time_s=60, failure=LocoManipFailure.LOCOMOTION_TIMEOUT
    )
    with pytest.raises(ValidationError):
        ArrivalReport(arrived=False, pos_err_m=0.5, yaw_err_rad=0.1, time_s=60)


def test_keyframe_matrix_shape_and_task_roundtrip():
    with pytest.raises(ValidationError):
        TaskKeyframe(phase="grasp", time_s=0, T_object_hand=[[1.0]])
    t = ManipulationTask(
        scenario="tabletop_v1",
        object_name="box",
        goal_xy_on_table=(0.1, 0.25),
        keyframes=[TaskKeyframe(phase="grasp", time_s=1.0, T_object_hand=I4, contact=True)],
    )
    assert ManipulationTask.model_validate_json(t.model_dump_json()) == t
