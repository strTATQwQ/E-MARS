from isaac_vln_benchmark.v8_coverage_utils import analyze_stale_attribution_v8


def test_v8_stale_attribution_splits_reset_cleanup_from_runtime():
    events = [
        {"event": "metrics_event_jsonl", "details": {"event_type": "omninav_stale", "result": "discarded", "attribution": "old_response_after_reset"}},
        {"event": "metrics_event_jsonl", "details": {"event_type": "route_choice_bridge_stale", "result": "discarded", "attribution": "episode_mismatch"}},
        {"event": "metrics_event_jsonl", "details": {"event_type": "safe_cmd_mux", "result": "accepted"}},
    ]

    stale = analyze_stale_attribution_v8(events)

    assert stale["total_stale_discards"] == 2
    assert stale["reset_cleanup_discards"] == 1
    assert stale["runtime_stale_discards"] == 1
    assert stale["old_response_after_reset"] == 1
    assert stale["episode_mismatch"] == 1


def test_v9_early_episode_mismatch_is_reset_cleanup_not_runtime():
    events = [
        {
            "t": 0.1,
            "event": "metrics_event_jsonl",
            "details": {
                "event_type": "primitive_command_stale",
                "result": "discarded",
                "attribution": "episode_mismatch",
            },
        }
    ]

    stale = analyze_stale_attribution_v8(events)

    assert stale["episode_mismatch"] == 1
    assert stale["reset_cleanup_discards"] == 1
    assert stale["runtime_stale_discards"] == 0
