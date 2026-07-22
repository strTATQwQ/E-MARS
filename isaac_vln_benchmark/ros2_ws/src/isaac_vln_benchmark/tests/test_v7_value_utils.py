import importlib.util
import json
from pathlib import Path

from isaac_vln_benchmark.v7_value_utils import (
    evaluate_forced_oracle,
    evaluate_golden_smoke,
    evaluate_route_stop,
    evaluate_sim2real_v7,
    max_linear_x_from_events,
    postprocess_v7_run,
    primitive_counts_from_events,
)


ROOT = Path(__file__).resolve().parents[4]


def _episode(mode, task_id, task_type, success=True, clean=True):
    return {
        "mode": mode,
        "task_id": task_id,
        "task_type": task_type,
        "success": success,
        "clean_success": clean,
        "path_length_m": 3.0,
        "num_collisions": 0,
        "num_stale_step_results": 0,
        "num_stale_omninav_actions": 0,
        "failure_reason": None if success else "timeout",
    }


def test_v7_event_helpers_accept_live_safe_mux_and_primitive_formats():
    events = [
        {
            "event": "metrics_event_jsonl",
            "details": {
                "event_type": "safe_cmd_mux",
                "cmd_vel": {"linear": {"x": 0.45}, "angular": {"z": 0.0}},
            },
        },
        {
            "event": "metrics_event_jsonl",
            "details": {
                "event_type": "primitive_command",
                "primitive": {"primitive": "move_forward"},
            },
        },
    ]

    assert max_linear_x_from_events(events) == 0.45
    assert primitive_counts_from_events(events)["move_forward_count"] == 1


def test_golden_smoke_requires_forward_motion_and_clean_timebase():
    metrics = {
        "episodes": [_episode("omninav_only_v6_golden", "simple_001_seed0", "simple_navigation")],
        "v7_summary": {
            "move_forward_count": 1,
            "max_linear_x_mps": 0.45,
            "timebase_error_count": 0,
            "collision_count": 0,
            "stale_action_executed": 0,
        },
    }

    assert evaluate_golden_smoke(metrics)["pass"] is True
    metrics["v7_summary"]["move_forward_count"] = 0
    failed = evaluate_golden_smoke(metrics)
    assert failed["pass"] is False
    assert "move_forward_count <= 0" in failed["failures"]


def test_forced_oracle_gate_requires_nonworse_baseline_and_task_signal():
    episodes = []
    for idx in range(6):
        episodes.append(_episode("omninav_only_v6_golden", f"simple_{idx}", "simple_navigation", success=True, clean=idx < 4))
    for idx in range(6):
        episodes.append(_episode("omninav_forced_route_stop_oracle_v7", f"simple_{idx}", "simple_navigation", success=True, clean=idx < 4))
    episodes.append(_episode("omninav_forced_route_stop_oracle_v7", "turn_001", "turn_choice", success=True, clean=True))
    metrics = {
        "episodes": episodes,
        "v7_summary": {"stale_discard_count": 0, "collision_count": 0, "stale_action_executed": 0},
    }

    result = evaluate_forced_oracle(metrics)
    assert result["pass"] is True
    assert result["oracle_turn_choice_success"] == 1


def test_route_stop_gate_matches_v7_value_claim_thresholds():
    episodes = []
    for idx in range(8):
        task_type = "turn_choice" if idx == 0 else "semantic_target" if idx == 1 else "simple_navigation"
        episodes.append(_episode("omninav_step_route_stop_v7", f"task_{idx}", task_type, success=True, clean=idx < 6))
    metrics = {"episodes": episodes, "v7_summary": {"collision_count": 0, "stale_action_executed": 0}}

    assert evaluate_route_stop(metrics)["pass"] is True


def test_sim2real_gate_v7_uses_nested_required_config_and_stays_conservative():
    metrics = {
        "v7_summary": {
            "max_linear_x_mps": 0.45,
            "route_choice_correct_branch_rate": 1.0,
            "stop_decision_accuracy": 1.0,
            "visible_to_stop_latency_sec": 0.5,
            "collision_count": 0,
            "stale_action_executed": 0,
            "timebase_error_count": 0,
            "parse_error_count": 0,
            "stale_discard_count": 0,
        }
    }
    gate = {"required": {"max_linear_x_mps_real_gate": 0.20, "stale_discard_count_max": 0}}

    result = evaluate_sim2real_v7(metrics, gate)
    assert result["ready"] is False
    assert result["status"] == "NOT READY FOR REAL ROBOT AUTONOMY"
    assert any("max_linear_x_mps" in failure for failure in result["failures"])


def test_postprocess_writes_v7_artifacts(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"episodes": [_episode("omninav_step_route_only_v7", "turn_001", "turn_choice")]}),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "safe_cmd_vel", "details": {"linear": {"x": 0.2}}}),
                json.dumps({"event": "metrics_event_jsonl", "details": {"event_type": "primitive_command", "primitive": {"primitive": "move_forward"}}}),
                json.dumps({"event": "metrics_event_jsonl", "details": {"event_type": "step_route_choice_json", "output": {"route_choice": "left", "confidence": 0.8, "visible_in_view": "left"}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = postprocess_v7_run(tmp_path, run_kind="unit")
    assert metrics["v7_summary"]["move_forward_count"] == 1
    assert (tmp_path / "step_route_decisions.csv").exists()
    assert (tmp_path / "stale_attribution.csv").exists()


def test_run_live_success_v7_mode_routing_keeps_forced_oracle_baseline_pair():
    script = ROOT / "scripts" / "run_live_success_v7.py"
    spec = importlib.util.spec_from_file_location("run_live_success_v7", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    forced = module.MODE_SPECS["forced_oracle"]
    assert forced["modes"] == ["omninav_only_v6_golden", "omninav_forced_route_stop_oracle_v7"]
    assert module.ALIASES["omninav_step_route_stop_v7"] == "step_route_stop"
