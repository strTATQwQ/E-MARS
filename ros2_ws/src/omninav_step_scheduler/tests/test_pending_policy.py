from omninav_step_scheduler.schemas import SchedulerState, parse_safety_status
from omninav_step_scheduler.step_pending_policy_node import choose_pending_policy_mode, pending_wait_exceeded


def test_intersection_blocks_move_while_step_thinks():
    mode = choose_pending_policy_mode(
        SchedulerState.STEP_THINK_MOVE.value,
        parse_safety_status({"near_intersection": True, "local_costmap_clear": True}),
        {"type": "move_forward", "interruptible": True},
        False,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        {"pending_policy": {"allow_move_while_step": True}},
    )

    assert mode == "stop"


def test_clear_interruptible_primitive_can_move_slow():
    mode = choose_pending_policy_mode(
        SchedulerState.STEP_THINK_MOVE.value,
        parse_safety_status({"local_costmap_clear": True, "human_distance_m": 9.0}),
        {"type": "move_forward", "interruptible": True},
        False,
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 1.0],
        {"pending_policy": {"allow_move_while_step": True}},
    )

    assert mode == "move_slow"


def test_multimodal_step_request_defaults_to_stop_when_configured():
    mode = choose_pending_policy_mode(
        SchedulerState.STEP_THINK_MOVE.value,
        parse_safety_status({"local_costmap_clear": True, "human_distance_m": 9.0}),
        {"type": "move_forward", "interruptible": True},
        True,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        {"pending_policy": {"allow_move_while_step": True, "stop_if_step_multimodal": True}},
    )

    assert mode == "stop"


def test_near_human_blocks_pending_motion():
    mode = choose_pending_policy_mode(
        SchedulerState.STEP_THINK_MOVE.value,
        parse_safety_status({"local_costmap_clear": True, "human_distance_m": 1.5}),
        {"type": "move_forward", "interruptible": True},
        False,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        {"pending_policy": {"allow_move_while_step": True}},
    )

    assert mode == "stop"


def test_pending_wait_threshold_allows_full_step_inference_then_holds():
    assert not pending_wait_exceeded(100.0, 105.0, 15.0)
    assert pending_wait_exceeded(100.0, 115.0, 15.0)
