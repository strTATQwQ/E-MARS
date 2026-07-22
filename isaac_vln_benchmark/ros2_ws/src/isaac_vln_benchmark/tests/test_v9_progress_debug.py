import json

from isaac_vln_benchmark.v8_coverage_utils import analyze_stale_attribution_v8
from isaac_vln_benchmark.v9_progress_utils import (
    analyze_route_reachability_episode,
    evaluate_progress_watchdog,
    evaluate_route_injection_sweep,
    evaluate_route_reachability,
    evaluate_semantic_approach,
    postprocess_route_reachability_run,
)


def test_v9_route_reachability_aggregates_progress_to_trigger():
    scene = {"semantic_zones": [{"class": "intersection", "center": [4.0, 0.0]}]}
    trajectory = [
        {"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0},
        {"t": 4.0, "x": 2.1, "y": 0.0, "yaw": 0.0},
        {"t": 7.0, "x": 3.25, "y": 0.04, "yaw": 0.0},
    ]
    events = [{"t": 1.0, "event": "safe_cmd_vel", "details": {"linear": {"x": 0.2}, "angular": {"z": 0.0}}}]
    episode = {"episode_id": "ep0", "mode": "v9_forced_forward_until_trigger", "task_id": "turn_001_seed0", "task_type": "turn_choice"}

    row = analyze_route_reachability_episode(episode, scene, events, trajectory)

    assert row["reached_injection_window"] is True
    assert row["time_to_x_2m"] == 4.0
    assert row["time_to_trigger_window"] == 7.0
    assert row["failure_stage"] is None


def test_v9_route_reachability_gate_requires_five_of_six_and_clean_stale():
    rows = []
    for index in range(6):
        rows.append(
            {
                "mode": "v9_forced_forward_until_trigger",
                "reached_injection_window": index < 5,
                "safe_cmd_vel_linear_nonzero_count": 3,
            }
        )

    result = evaluate_route_reachability(rows, {"episodes": [{} for _ in range(6)]}, {"runtime_stale_discards": 0, "timebase_error": 0})

    assert result["pass"] is True
    rows[0]["reached_injection_window"] = False
    assert evaluate_route_reachability(rows, {"episodes": [{} for _ in range(6)]}, {"runtime_stale_discards": 0, "timebase_error": 0})["pass"] is False


def test_v9_injection_window_selects_earliest_effective_window():
    rows = []
    for window in ["x>=1.5", "x>=2.0"]:
        for index in range(6):
            rows.append(
                {
                    "trigger_window": window,
                    "route_oracle_triggered": True,
                    "route_oracle_json_published": True,
                    "route_stop_bridge_received_route": True,
                    "turn_primitive_generated": True,
                    "safe_cmd_vel_turn_nonzero": True,
                    "entered_correct_branch": window == "x>=2.0" and index < 4,
                }
            )

    result = evaluate_route_injection_sweep(rows, {"episodes": [{} for _ in rows]}, {"runtime_stale_discards": 0, "timebase_error": 0})

    assert result["pass"] is True
    assert result["earliest_effective_window"] == "x>=2.0"


def test_v9_semantic_forced_approach_requires_threshold_and_two_successes():
    rows = []
    for index in range(3):
        rows.append(
            {
                "mode": "v9_semantic_forced_approach",
                "reached_stop_threshold": True,
                "stop_oracle_triggered": True,
                "bridge_received_stop": True,
                "safe_cmd_vel_zero_time": 1.2,
                "semantic_target_success": index < 2,
            }
        )

    result = evaluate_semantic_approach(rows, {"episodes": [{} for _ in range(3)]}, {"runtime_stale_discards": 0, "timebase_error": 0})

    assert result["pass"] is True
    rows[0]["reached_stop_threshold"] = False
    assert evaluate_semantic_approach(rows, {"episodes": [{} for _ in range(3)]}, {"runtime_stale_discards": 0, "timebase_error": 0})["pass"] is False


def test_v9_watchdog_reports_assist_unlock():
    rows = [
        {
            "assist_triggered": True,
            "assist_reason": "elapsed_gt_20_x_lt_2",
            "reached_trigger_after_assist": True,
            "success_after_assist": True,
        }
    ]

    result = evaluate_progress_watchdog(rows, {"episodes": [{}]}, {"runtime_stale_discards": 0, "timebase_error": 0})

    assert result["pass"] is True
    assert result["assist_reasons"] == ["elapsed_gt_20_x_lt_2"]


def test_v9_report_generation_writes_required_artifacts(tmp_path):
    out = tmp_path / "run"
    episode_dir = out / "v9_forced_forward_until_trigger" / "turn_001_seed0"
    episode_dir.mkdir(parents=True)
    metrics = {
        "episodes": [
            {
                "episode_id": "ep0",
                "mode": "v9_forced_forward_until_trigger",
                "task_id": "turn_001_seed0",
                "task_type": "turn_choice",
                "num_collisions": 0,
            }
        ]
    }
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out / "events.jsonl").write_text("", encoding="utf-8")
    (episode_dir / "events.jsonl").write_text(
        json.dumps({"t": 1.0, "event": "safe_cmd_vel", "details": {"linear": {"x": 0.2}, "angular": {"z": 0.0}}}) + "\n",
        encoding="utf-8",
    )
    (episode_dir / "scene.yaml").write_text(json.dumps({"semantic_zones": [{"class": "intersection", "center": [4.0, 0.0]}]}), encoding="utf-8")
    (episode_dir / "trajectory.csv").write_text("t,x,y,yaw\n0,0,0,0\n1,3.3,0,0\n", encoding="utf-8")

    result = postprocess_route_reachability_run(out)

    assert "v9_route_trigger_reachability" in result
    for name in [
        "summary.md",
        "coverage_chain.csv",
        "route_trigger_reachability.csv",
        "route_injection_sweep.csv",
        "semantic_approach_probe.csv",
        "progress_watchdog.csv",
        "stale_attribution.csv",
        "trajectory.csv",
        "visual/viewport.png",
    ]:
        assert (out / name).exists()


def test_v9_stale_runtime_timebase_split_is_preserved():
    events = [
        {"event": "metrics_event_jsonl", "details": {"event_type": "omninav_stale", "result": "discarded", "attribution": "old_response_after_reset"}},
        {"event": "metrics_event_jsonl", "details": {"event_type": "route_choice_bridge_stale", "result": "discarded", "attribution": "timebase_error"}},
    ]

    stale = analyze_stale_attribution_v8(events)

    assert stale["reset_cleanup_discards"] == 1
    assert stale["runtime_stale_discards"] == 1
    assert stale["timebase_error"] == 1
