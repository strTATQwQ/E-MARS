import json
import subprocess
import sys
from pathlib import Path

from isaac_vln_benchmark.v5_timebase_utils import (
    analyze_stale_events_v5,
    classify_attribution,
    evaluate_sim2real_readiness_v5,
    evaluate_timebase_probe,
    make_timebase_probe_rows,
    write_v5_run_artifacts,
)


ROOT = Path(__file__).resolve().parents[4]


def test_sim2real_gate_v5_fails_on_timebase_error():
    result = evaluate_sim2real_readiness_v5(
        {
            "timebase_error_count": 1,
            "episode_mismatch_count": 0,
            "missing_timestamp_count": 0,
            "stale_discard_count": 0,
            "route_choice_correct_branch_rate": 0.8,
            "semantic_stop_accuracy": 0.8,
            "visible_to_stop_latency_sec": 1.0,
            "collision_count": 0,
            "stale_action_executed": 0,
            "parse_error": 0,
            "max_linear_x_mps": 0.2,
            "actions_through_safe_mux": True,
        }
    )

    assert result["ready"] is False
    assert result["status"] == "NOT READY FOR REAL ROBOT AUTONOMY"
    assert any("timebase_error_count" in failure for failure in result["failures"])


def test_timebase_probe_artifacts_are_complete(tmp_path):
    rows = make_timebase_probe_rows(3)
    metrics = evaluate_timebase_probe(rows)
    write_v5_run_artifacts(tmp_path, title="stale_gate_timebase_probe", metrics=metrics, timebase_rows=rows)

    for name in (
        "summary.md",
        "metrics.json",
        "events.jsonl",
        "stale_attribution.csv",
        "timebase_table.csv",
        "mode_table.csv",
        "failure_table.csv",
        "trajectory.csv",
    ):
        assert (tmp_path / name).exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["pass"] is True


def test_timebase_fields_do_not_imply_timebase_error():
    event = {
        "event_type": "safe_cmd_mux",
        "result": "accepted",
        "clock_domain": "ros_system",
        "ros_now_sec": 123.0,
        "wall_now_sec": 123.0,
        "clock_msg_sec": 123.0,
        "header_stamp_sec": 123.0,
        "source_stamp_sec": 123.0,
        "created_ros_time_sec": 123.0,
        "created_wall_time_sec": 123.0,
    }

    assert classify_attribution(event) == "valid"
    analysis = analyze_stale_events_v5([event])
    assert analysis["counts"]["timebase_error"] == 0
    assert analysis["total_discards"] == 0


def test_accepted_primitive_ttl_does_not_imply_stale_due_to_age():
    event = {
        "event_type": "primitive_command",
        "result": "accepted",
        "primitive": {"ttl_sec": 1.0},
        "clock_domain": "ros_system",
        "episode_id": "episode_a",
    }

    assert classify_attribution(event) == "valid"
    analysis = analyze_stale_events_v5([event])
    assert analysis["counts"]["stale_due_to_age"] == 0
    assert analysis["total_discards"] == 0


def test_sim2real_gate_v5_reports_missing_metrics_without_sentinel_values():
    result = evaluate_sim2real_readiness_v5(
        {
            "timebase_error_count": 0,
            "episode_mismatch_count": 0,
            "missing_timestamp_count": 0,
            "stale_discard_count": 20,
            "collision_count": 0,
            "parse_errors": 0,
            "max_linear_x_mps": 0.0,
            "safe_mux_results": {"accepted": 1},
        }
    )

    assert result["ready"] is False
    assert "actions_through_safe_mux=missing/unverified" not in result["failures"]
    assert any(failure == "stale_action_executed=missing/unverified" for failure in result["failures"])
    assert any(failure == "route_choice_correct_branch_rate=missing/unverified" for failure in result["failures"])
    assert not any("999" in failure or "-1.0" in failure for failure in result["failures"])


def test_analyze_stale_discards_accepts_run_root_with_out_dir(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "stale_gate", "discard_reason": "old_response_after_reset"}) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps({"stale_discard_count": 1}), encoding="utf-8")

    script = ROOT / "scripts" / "analyze_stale_discards.py"
    result = subprocess.run(
        [sys.executable, str(script), "--v5", "--input", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["total_stale_discards"] == 1
    assert (run_dir / "stale_discard_analysis.md").exists()
    assert (run_dir / "stale_attribution.csv").exists()
