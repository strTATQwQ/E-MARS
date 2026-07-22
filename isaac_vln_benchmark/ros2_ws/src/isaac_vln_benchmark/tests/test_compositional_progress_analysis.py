from isaac_vln_benchmark.compositional_progress_analysis import classify_subgoal_failure


def base_row(**overrides):
    row = {
        "completed": False,
        "start_t": 0.0,
        "subgoal_type": "pass",
        "start_distance_m": 2.3,
        "min_distance_m": 2.0,
        "end_distance_m": 2.1,
        "turn_action_ratio": 0.2,
        "forward_action_ratio": 0.7,
        "turn_direction_switches": 0,
        "crossed_completion_plane": False,
        "min_cross_track_after_plane_m": None,
        "recovery_completed_events": 0,
        "distance_decrease_step_ratio": 0.8,
    }
    row.update(overrides)
    return row


def test_classifies_verify_without_approach_as_benchmark_gap():
    row = base_row(subgoal_type="verify", start_distance_m=2.44)
    assert classify_subgoal_failure(row) == "verify_without_approach_gap"


def test_classifies_turn_oscillation_before_generic_no_progress():
    row = base_row(
        turn_action_ratio=0.86,
        forward_action_ratio=0.04,
        turn_direction_switches=17,
    )
    assert classify_subgoal_failure(row) == "turn_oscillation_no_progress"


def test_classifies_near_pass_overshoot_as_completion_geometry_miss():
    row = base_row(min_distance_m=1.57, end_distance_m=5.0)
    assert classify_subgoal_failure(row) == "completion_geometry_miss_or_overshoot"


def test_prior_subgoal_failure_marks_later_subgoal_not_reached():
    row = base_row(start_t=None)
    assert classify_subgoal_failure(row) == "not_reached_due_to_prior_subgoal"
