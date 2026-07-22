from isaac_vln_benchmark.v8_coverage_utils import analyze_stop_episode, evaluate_stop_coverage


def test_v8_semantic_stop_chain_records_latency_and_success_conditions():
    task = {
        "task_id": "semantic_001",
        "task_type": "semantic_target",
        "target_object": "fire_extinguisher_1",
        "success": {"distance_to_target_m": 2.5, "target_visible": True, "stop_required": True},
    }
    scene = {"objects": [{"id": "fire_extinguisher_1", "pose": [5.0, 0.0, 0.5]}]}
    zero = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
    events = [
        {"t": 0.5, "event": "isaac_episode_status", "details": {"target_visible": True, "distance_to_target": 2.4, "done": False}},
        {"t": 1.0, "event": "metrics_event_jsonl", "details": {"event_type": "semantic_stop_coverage", "target_visible": True, "distance_to_target": 2.3, "semantic_stop_oracle_triggered": True}},
        {"t": 1.1, "event": "oracle_semantic_stop_json", "details": {"stop": True, "target_visible": True, "distance_to_target": 2.3}},
        {"t": 1.2, "event": "metrics_event_jsonl", "details": {"event_type": "semantic_stop_bridge", "result": "primitive_published", "primitive": {"primitive": "stop"}}},
        {"t": 1.3, "event": "metrics_event_jsonl", "details": {"event_type": "primitive_command", "action_type": "stop", "primitive": {"primitive": "stop"}, "cmd_vel": zero}},
        {"t": 1.4, "event": "metrics_event_jsonl", "details": {"event_type": "safe_cmd_mux", "result": "accepted", "cmd_vel": zero}},
    ]
    episode = {"episode_id": "ep0", "mode": "omninav_semantic_stop_coverage_v8", "task_id": "semantic_001", "task_type": "semantic_target", "success": True}

    row = analyze_stop_episode(episode, task, scene, events, [])

    assert row["failure_stage"] is None
    assert row["visible_to_stop_latency"] == 0.9
    assert row["success_judge_distance_condition"] is True
    assert row["semantic_target_success"] is True


def test_v8_semantic_stop_gate_allows_initial_one_of_three_success():
    rows = []
    for index in range(3):
        rows.append(
            {
                "target_visible_first_time": 0.5,
                "semantic_stop_oracle_triggered": True,
                "bridge_received_stop": True,
                "safe_cmd_vel_zero_time": 1.0,
                "semantic_target_success": index == 0,
                "visible_to_stop_latency": 1.0,
            }
        )

    assert evaluate_stop_coverage(rows)["pass"] is True
    rows[0]["semantic_target_success"] = False
    assert evaluate_stop_coverage(rows)["pass"] is False
