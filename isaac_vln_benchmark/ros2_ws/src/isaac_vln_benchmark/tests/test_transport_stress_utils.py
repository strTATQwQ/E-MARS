import json

from isaac_vln_benchmark.transport_stress_utils import evaluate_transport_profile, evaluate_transport_stress


def write_run(tmp_path, events, *, success=True):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"episodes": [{"episode_id": "ep1", "task_type": "turn_choice", "success": success, "num_collisions": 0, "stale_action_rate": 0.0}]}),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")


def event(name, details):
    return {"event": name, "details": details}


def test_latency_profile_requires_audited_delivery_and_success(tmp_path):
    write_run(
        tmp_path,
        [
            event("stress_raw_step_route_choice_json", {"request_id": "r1"}),
            event("benchmark_delay_audit_json", {"request_id": "r1", "delay_sec": 0.21, "dropped": False}),
            event("primitive_command_json", {"request_id": "r1"}),
        ],
    )

    result = evaluate_transport_profile(tmp_path, profile="latency", requested_delay_sec=0.2, requested_drop_rate=0.0)

    assert result["pass"] is True


def test_drop_profile_rejects_dropped_request_becoming_primitive(tmp_path):
    write_run(
        tmp_path,
        [
            event("stress_raw_step_route_choice_json", {"request_id": "r2"}),
            event("benchmark_delay_audit_json", {"request_id": "r2", "delay_sec": None, "dropped": True}),
            event("primitive_command_json", {"request_id": "r2"}),
        ],
        success=False,
    )

    result = evaluate_transport_profile(tmp_path, profile="packet_loss", requested_delay_sec=0.0, requested_drop_rate=1.0)

    assert result["pass"] is False
    assert result["forbidden_primitive_request_ids"] == ["r2"]


def test_reset_profile_requires_old_response_discard(tmp_path):
    write_run(
        tmp_path,
        [
            event("stress_raw_step_route_choice_json", {"request_id": "r3"}),
            event("benchmark_delay_audit_json", {"request_id": "r3", "delay_sec": 1.01, "dropped": False}),
            event("benchmark_reset_stress_json", {"request_id": "r3"}),
            event(
                "metrics_event_jsonl",
                {"request_id": "r3", "event_type": "route_choice_bridge_stale", "result": "discarded", "attribution": "old_response_after_reset"},
            ),
        ],
        success=False,
    )

    profile = evaluate_transport_profile(tmp_path, profile="reset", requested_delay_sec=1.0, requested_drop_rate=0.0, reset_expected=True)
    overall = evaluate_transport_stress([
        {"profile": "latency", "pass": True},
        {"profile": "packet_loss", "pass": True},
        profile,
    ])

    assert profile["pass"] is True
    assert overall["pass"] is True
