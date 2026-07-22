from isaac_vln_benchmark.decision_replay_utils import evaluate_decision_replay


def test_decision_replay_requires_discard_attribution_and_no_motion():
    records = [
        ("/benchmark/mode_json", {"episode_id": "e"}),
        ("/step/route_choice_json", {"request_id": "accepted"}),
        ("/step/request_json", {"request_id": "dropped_http"}),
        ("/primitive/command_json", {"request_id": "accepted"}),
        ("/cmd_vel_candidate", {"linear": {"x": 0.0}, "angular": {"z": 0.0}}),
        ("/scheduler/state", "STEP_THINK_STOP"),
    ]
    for attribution in (
        "duplicate_response",
        "out_of_order",
        "episode_mismatch",
        "timebase_error",
        "old_response_after_reset",
    ):
        records.append(("/metrics/event_jsonl", {"event_type": "discard", "attribution": attribution}))
    expected = [{"profile": name, "pass": True} for name in (
        "delay", "drop", "duplicate", "out_of_order", "reset",
        "episode_mismatch", "timestamp_mismatch", "horizontal_flip",
    )]
    assert evaluate_decision_replay(records, expected)["pass"] is True


def test_decision_replay_rejects_blind_forward():
    records = [
        ("/scheduler/state", "STEP_THINK_STOP"),
        ("/cmd_vel_candidate", {"linear": {"x": 0.1}}),
    ]
    result = evaluate_decision_replay(records, [])
    assert result["pass"] is False
    assert result["checks"]["pending_or_timeout_never_blind_forward"] is False
