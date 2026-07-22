import json

from isaac_vln_benchmark.reporting import write_summary


def test_report_generation_writes_clean_recovered_and_csvs(tmp_path):
    metrics = [
        {
            "mode": "internnav_only",
            "task_id": "simple_001",
            "task_type": "simple_navigation",
            "success": True,
            "clean_success": True,
            "recovered_success": False,
            "success_class": "clean_success",
            "failure_reason": None,
            "mission_time_sec": 10.0,
            "path_length_m": 2.0,
            "num_step_calls": 0,
            "num_omninav_calls": 0,
            "num_internnav_calls": 10,
            "internnav_mean_latency_s": 0.04,
            "recovery_override_count": 0,
            "recovery_override_rate": 0.0,
            "safety_block_ticks": 0,
            "safety_intervention_events": 0,
            "recovery_override_ticks": 0,
            "recovery_override_events": 0,
            "episodes_with_recovery": 0,
            "forward_ratio": 0.8,
            "left_ratio": 0.2,
            "right_ratio": 0.0,
            "stop_ratio": 0.0,
            "action_entropy": 0.72,
        },
        {
            "mode": "step_internnav_event",
            "task_id": "simple_002",
            "task_type": "simple_navigation",
            "success": True,
            "clean_success": False,
            "recovered_success": True,
            "success_class": "recovered_success",
            "failure_reason": None,
            "mission_time_sec": 11.0,
            "path_length_m": 2.2,
            "num_step_calls": 1,
            "num_omninav_calls": 0,
            "num_internnav_calls": 10,
            "internnav_mean_latency_s": 0.04,
            "recovery_override_count": 1,
            "recovery_override_rate": 0.1,
            "safety_block_ticks": 4,
            "safety_intervention_events": 1,
            "recovery_override_ticks": 1,
            "recovery_override_events": 1,
            "episodes_with_recovery": 1,
            "forward_ratio": 0.7,
            "left_ratio": 0.3,
            "right_ratio": 0.0,
            "stop_ratio": 0.0,
            "action_entropy": 0.88,
        },
    ]
    path = write_summary(tmp_path, metrics, {"latency_model": {}, "oracle_semantics": True})
    text = path.read_text(encoding="utf-8")
    assert "clean_success_rate" in text
    assert "recovered_success_rate" in text
    assert (tmp_path / "mode_table.csv").exists()
    assert (tmp_path / "failure_table.csv").exists()
    assert (tmp_path / "action_distribution.csv").exists()
    assert (tmp_path / "safety_recovery_events.csv").exists()
