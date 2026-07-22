from types import SimpleNamespace

from isaac_vln_benchmark.bag_replay_utils import evaluate_bag_records


def string(data):
    import json

    return SimpleNamespace(data=json.dumps(data))


def test_offline_bag_replay_reconstructs_success_and_safety_chain():
    episode_id = "mode_turn_seed0_0000"
    twist = SimpleNamespace(
        linear=SimpleNamespace(x=0.2),
        angular=SimpleNamespace(z=0.3),
    )
    records = [
        ("/benchmark/mode_json", string({"episode_id": episode_id})),
        ("/isaac/episode_status", string({"episode_id": episode_id, "success": True, "done": True})),
        ("/safe_cmd_vel", twist),
        ("/metrics/event_jsonl", string({"event_type": "route_choice_bridge", "result": "primitive_published"})),
        ("/step/route_choice_json", string({"episode_id": episode_id, "route_choice": "left"})),
        ("/primitive/command_json", string({"episode_id": episode_id, "primitive": "follow_waypoint"})),
    ]
    metrics = {"episodes": [{"episode_id": episode_id, "task_type": "turn_choice", "success": True}]}

    result = evaluate_bag_records(records, metrics)

    assert result["pass"] is True
    assert result["safe_linear_max_mps"] == 0.2


def test_offline_bag_replay_rejects_missing_role_message():
    episode_id = "mode_semantic_seed0_0000"
    records = [
        ("/benchmark/mode_json", string({"episode_id": episode_id})),
        ("/isaac/episode_status", string({"episode_id": episode_id, "success": True})),
        ("/safe_cmd_vel", {"linear": {"x": 0.0}, "angular": {"z": 0.0}}),
        ("/metrics/event_jsonl", string({"event_type": "ok"})),
        ("/primitive/command_json", string({"primitive": "stop"})),
    ]
    metrics = {"episodes": [{"episode_id": episode_id, "task_type": "semantic_target", "success": True}]}

    result = evaluate_bag_records(records, metrics)

    assert result["pass"] is False
    assert result["checks"]["semantic_stop_count_matches"] is False
