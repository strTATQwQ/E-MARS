from isaac_vln_benchmark.v8_coverage_utils import evaluate_sim2real_v8


def test_v8_sim2real_gate_stays_not_ready_when_real_speed_gate_fails():
    metrics = {
        "episodes": [{"success": True, "clean_success": True, "num_collisions": 0, "failure_reason": None}],
        "v8_route_coverage": {"correct_branch_rate": 1.0},
        "v8_semantic_stop_coverage": {"semantic_stop_accuracy": 1.0, "visible_to_stop_latency_max_sec": 1.0},
        "v8_stale_attribution": {"runtime_stale_discards": 0, "timebase_error": 0},
        "_events": [
            {
                "event": "metrics_event_jsonl",
                "details": {
                    "event_type": "safe_cmd_mux",
                    "result": "accepted",
                    "cmd_vel": {"linear": {"x": 0.45}, "angular": {"z": 0.0}},
                },
            }
        ],
    }
    gate = {"required": {"max_linear_x_mps_real_gate": 0.20}}

    result = evaluate_sim2real_v8(metrics, gate)

    assert result["ready"] is False
    assert result["status"] == "NOT READY FOR REAL ROBOT AUTONOMY"
    assert any("max_linear_x_mps" in item for item in result["failures"])
