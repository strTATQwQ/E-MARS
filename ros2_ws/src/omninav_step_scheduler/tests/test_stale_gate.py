from omninav_step_scheduler.schemas import OmniNavAction, StepConstraints, StepPlan
from omninav_step_scheduler.stale_gate import (
    DEFAULT_CONFIG,
    is_omninav_action_valid,
    is_step_plan_valid,
)


def _plan(created_at=100.0, modality="multimodal"):
    return StepPlan(
        request_id="p1",
        timestamp_request=created_at - 1.0,
        timestamp_response=created_at,
        multimodal=modality == "multimodal",
        pose_at_request=[0.0, 0.0, 0.0],
        navila_or_omninav_instruction="go to the door",
        subgoal="door",
        success_condition="arrived",
        constraints=StepConstraints(),
        replan_triggers=[],
        recommended_pending_mode="stop",
        confidence=0.9,
    )


def test_multimodal_step_plan_accepts_small_pose_drift():
    ok = is_step_plan_valid(_plan(), [0.1, 0.0, 5.0], 101.0, DEFAULT_CONFIG)

    assert ok


def test_multimodal_step_plan_rejects_large_pose_drift():
    ok = is_step_plan_valid(_plan(), [0.5, 0.0, 5.0], 101.0, DEFAULT_CONFIG)

    assert not ok


def test_text_step_plan_allows_larger_pose_drift():
    ok = is_step_plan_valid(_plan(modality="text"), [0.25, 0.0, 5.0], 101.0, DEFAULT_CONFIG)

    assert ok


def test_omninav_action_rejects_stale_candidate():
    action = OmniNavAction(
        request_id="a1",
        timestamp_request=99.9,
        timestamp_response=100.0,
        frame_timestamp=100.0,
        pose_at_snapshot=[0.0, 0.0, 0.0],
        primitive="move_forward",
        distance_m=0.25,
        yaw_deg=0.0,
        confidence=0.9,
    )

    ok = is_omninav_action_valid(action, [0.0, 0.0, 0.0], 101.0, DEFAULT_CONFIG)

    assert not ok
