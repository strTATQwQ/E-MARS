from isaac_vln_benchmark.v8_coverage_utils import analyze_route_episode, evaluate_route_coverage


def test_v8_route_coverage_chain_detects_bridge_primitive_safe_mux_and_branch():
    task = {
        "task_id": "turn_001",
        "task_type": "turn_choice",
        "instruction": "At the intersection, turn left toward the red sign.",
        "target_object": "exit_sign_1",
    }
    scene = {
        "robot_start_pose": [0.0, 0.0, 0.0],
        "objects": [{"id": "exit_sign_1", "pose": [6.0, 2.0, 1.5]}],
    }
    events = [
        {"t": 0.9, "event": "metrics_event_jsonl", "details": {"event_type": "route_oracle_coverage", "near_intersection_detected": True, "route_oracle_triggered": True, "route_oracle_json_published": True, "route_oracle_choice": "left"}},
        {"t": 1.0, "event": "oracle_route_choice_json", "details": {"route_choice": "left", "source": "forced_oracle"}},
        {"t": 1.1, "event": "metrics_event_jsonl", "details": {"event_type": "route_choice_bridge", "result": "primitive_published", "decision": {"route_choice": "left", "source": "forced_oracle"}, "primitive": {"primitive": "follow_waypoint", "yaw_deg": 35}}},
        {"t": 1.2, "event": "metrics_event_jsonl", "details": {"event_type": "primitive_command", "action_type": "follow_waypoint", "primitive": {"primitive": "follow_waypoint"}, "cmd_vel": {"linear": {"x": 0.1}, "angular": {"z": 0.35}}}},
        {"t": 1.3, "event": "metrics_event_jsonl", "details": {"event_type": "safe_cmd_mux", "result": "accepted", "cmd_vel": {"linear": {"x": 0.1}, "angular": {"z": 0.35}}}},
    ]
    trajectory = [{"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0}, {"t": 5.0, "x": 4.0, "y": 0.8, "yaw": 0.5}]
    episode = {"episode_id": "ep0", "mode": "omninav_forced_route_stop_oracle_v8", "task_id": "turn_001", "task_type": "turn_choice"}

    row = analyze_route_episode(episode, task, scene, events, trajectory)

    assert row["failure_stage"] is None
    assert row["route_stop_bridge_received_route"] is True
    assert row["safe_cmd_vel_turn_nonzero"] is True
    assert row["entered_correct_branch"] is True


def test_v8_route_coverage_gate_requires_four_correct_branches():
    rows = []
    for index in range(6):
        rows.append(
            {
                "near_intersection_detected": True,
                "route_oracle_triggered": True,
                "route_oracle_json_published": True,
                "route_stop_bridge_received_route": True,
                "turn_primitive_generated": True,
                "safe_cmd_vel_turn_nonzero": True,
                "entered_correct_branch": index < 4,
            }
        )

    assert evaluate_route_coverage(rows)["pass"] is True
    rows[-1]["safe_cmd_vel_turn_nonzero"] = False
    assert evaluate_route_coverage(rows)["pass"] is False
