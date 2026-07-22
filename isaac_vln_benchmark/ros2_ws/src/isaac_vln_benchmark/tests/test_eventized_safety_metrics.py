from isaac_vln_benchmark.metrics import SafetyRecoveryEventCounter, eventized_safety_recovery_metrics


def test_continuous_blocked_ticks_are_one_safety_event():
    counter = SafetyRecoveryEventCounter()
    for tick in range(100):
        counter.update_safety(True, tick * 0.05)
    counter.update_safety(False, 5.0)

    result = counter.finalize()

    assert result["safety_block_ticks"] == 100
    assert result["safety_intervention_events"] == 1
    assert result["max_continuous_safety_block_sec"] > 4.0


def test_event_stream_counts_recovery_ticks_and_events():
    events = [
        {"t": 0.0, "event": "metrics_event_jsonl", "details": {"event_type": "internnav_primitive", "recovery_override": True}},
        {"t": 0.5, "event": "metrics_event_jsonl", "details": {"event_type": "internnav_primitive", "recovery_override": True}},
        {"t": 1.0, "event": "metrics_event_jsonl", "details": {"event_type": "internnav_primitive", "recovery_override": False}},
        {"t": 2.0, "event": "metrics_event_jsonl", "details": {"event_type": "internnav_primitive", "recovery_override": True}},
    ]

    result = eventized_safety_recovery_metrics(events)

    assert result["recovery_override_ticks"] == 3
    assert result["recovery_override_events"] == 2
    assert result["episodes_with_recovery"] == 1
