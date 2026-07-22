import json

from isaac_vln_benchmark.sensor_only_audit import audit_run, audit_runtime_events, nested_forbidden_fields


def valid_events():
    return [
        {
            "event_type": "grounded_sam_request",
            "image_source": "actual_isaac_viewport",
            "model": "groundingdino_b_sam2_1_hiera_large",
        },
        {"event_type": "sensor_target_track", "track": {"distance_m": 2.2, "bearing_rad": 0.1}},
        {"event_type": "sensor_planner_primitive", "primitive": {"phase": "approach"}},
        {"event_type": "primitive_command", "result": "accepted"},
    ]


def test_sensor_only_audit_accepts_actual_sensor_chain():
    result = audit_runtime_events(valid_events())
    assert result["pass"] is True
    assert result["oracle_context_leakage"] == 0


def test_sensor_only_audit_rejects_nested_truth_and_synthetic_source():
    events = valid_events()
    events[1]["track"]["target_pose"] = [1.0, 2.0, 0.0]
    events[2]["source"] = "synthetic_camera_fallback"
    result = audit_runtime_events(events)
    assert result["pass"] is False
    assert "oracle_context_leakage" in result["failures"]
    assert "mock_fallback_or_synthetic_source" in result["failures"]
    assert nested_forbidden_fields(events[1]) == ["track.target_pose"]


def test_sensor_only_audit_reads_metrics_event_wrapper():
    wrapped = [
        {"event": "metrics_event_jsonl", "details": event, "timestamp": index}
        for index, event in enumerate(valid_events())
    ]
    result = audit_runtime_events(wrapped)
    assert result["pass"] is True
    assert result["required_event_counts"] == {
        "grounded_sam_request": 1,
        "sensor_target_track": 1,
        "sensor_planner_primitive": 1,
        "primitive_command": 1,
    }


def test_run_audit_does_not_double_count_remote_artifact_copies(tmp_path):
    lines = "".join(json.dumps(event) + "\n" for event in valid_events())
    (tmp_path / "events.jsonl").write_text(lines, encoding="utf-8")
    duplicate = tmp_path / "_remote" / "out"
    duplicate.mkdir(parents=True)
    (duplicate / "events.jsonl").write_text(lines, encoding="utf-8")
    result = audit_run(tmp_path)
    assert result["controller_event_count"] == 4
    assert len(result["event_files"]) == 1
