from omninav_step_scheduler.internnav_bridge import (
    InternNavProgressMonitor,
    action_entropy,
    normalize_action_name,
    primitive_for_action,
)


def test_internnav_action_mapping_unknown_stops():
    assert primitive_for_action("forward") == "move_forward"
    assert primitive_for_action("left") == "turn_left"
    assert primitive_for_action("right") == "turn_right"
    assert primitive_for_action("unknown") == "stop"
    assert primitive_for_action(999) == "stop"
    assert normalize_action_name("move_forward") == "forward"


def test_progress_monitor_detects_forward_bias_without_goal_improvement():
    monitor = InternNavProgressMonitor(
        window_sec=8.0,
        min_progress_m=0.5,
        max_forward_without_goal_update=3,
        max_same_action_ratio=0.85,
    )
    decision = None
    for i in range(6):
        decision = monitor.update(
            timestamp=float(i),
            pose=[0.2 * i, 0.0, 0.0],
            target_distance_m=5.0,
            model_action="forward",
            applied_action="forward",
        )
        if decision.trigger:
            break
    assert decision is not None
    assert decision.trigger in {"forward_bias", "no_progress"}
    assert decision.forward_ratio == 1.0
    assert decision.recovery_primitive == "move_forward"


def test_action_entropy_for_mixed_actions():
    assert action_entropy(["forward", "forward", "forward"]) == 0.0
    assert action_entropy(["forward", "left"]) == 1.0
