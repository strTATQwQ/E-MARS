import importlib.util
from pathlib import Path
from types import SimpleNamespace

from isaac_vln_benchmark.v6_recovery_utils import (
    classify_first10_root_cause,
    config_diff_rows,
    evaluate_action_parser_records,
    evaluate_safe_mux_clamp_records,
    evaluate_sim2real_readiness_v6,
)


def _load_live_success_benchmark_script():
    script = Path(__file__).resolve().parents[4] / "scripts" / "run_live_success_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_live_success_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_action_parser_probe_passes_forward_and_no_unknown():
    metrics = evaluate_action_parser_records(
        [
            {
                "parsed_primitive": "move_forward",
                "fallback_used": False,
                "cmd_vel_candidate": {"linear_x": 0.2, "angular_z": 0.0},
            },
            {
                "parsed_primitive": "turn_left",
                "fallback_used": False,
                "cmd_vel_candidate": {"linear_x": 0.0, "angular_z": 0.2},
            },
        ]
    )

    assert metrics["pass"] is True
    assert metrics["move_forward_count"] == 1
    assert metrics["unknown_count"] == 0


def test_safe_mux_clamp_probe_fails_when_safe_linear_zero():
    metrics = evaluate_safe_mux_clamp_records(
        [{"cmd_vel_candidate": {"linear_x": 0.15}, "safe_cmd_vel": {"linear_x": 0.0}}],
        path_m=0.4,
    )

    assert metrics["pass"] is False
    assert any("safe_cmd_vel" in failure for failure in metrics["failures"])


def test_config_diff_marks_zero_max_linear_x_high_risk():
    rows = config_diff_rows(
        {
            "v3": {"primitive": {"max_linear_x": 0.05}},
            "v5": {"primitive": {"max_linear_x": 0.05}},
            "v6": {"primitive": {"max_linear_x": 0.0}},
        },
        fields=[("primitive.max_linear_x", "max_linear_x")],
    )

    assert rows[0]["risk"].startswith("high")


def test_first10_trace_classifies_parser_forward_to_turn():
    root = classify_first10_root_cause(
        [
            {
                "raw_waypoint": [0.4, 0.0],
                "raw_action": "move_forward",
                "parsed_primitive": "turn_left",
                "cmd_vel_candidate": {"linear_x": 0.0},
                "safe_cmd_vel": {"linear_x": 0.0},
                "pose_delta": 0.0,
            }
        ]
    )

    assert root == "parser_maps_forward_to_turn"


def test_first10_trace_classifies_safe_mux_clamp():
    root = classify_first10_root_cause(
        [
            {
                "raw_waypoint": [0.4, 0.0],
                "parsed_primitive": "move_forward",
                "cmd_vel_candidate": {"linear_x": 0.2},
                "safe_cmd_vel": {"linear_x": 0.0},
                "pose_delta": 0.0,
            }
        ]
    )

    assert root == "safe_mux_clamps_linear_x"


def test_sim2real_gate_v6_requires_omninav_baseline_restored():
    result = evaluate_sim2real_readiness_v6(
        {
            "success_count": 0,
            "mean_path_m": 0.139,
            "move_forward_count": 0,
            "max_linear_x_mps": 0.0,
            "stale_discard_count": 20,
            "timebase_error_count": 0,
            "episode_mismatch_count": 0,
            "missing_timestamp_count": 0,
            "collision_count": 0,
            "stale_action_executed": 0,
        }
    )

    assert result["ready"] is False
    assert result["status"] == "NOT READY FOR REAL ROBOT AUTONOMY"
    assert any("success_count" in failure for failure in result["failures"])


def test_mock_omninav_live_run_stops_remote_real_client(monkeypatch):
    module = _load_live_success_benchmark_script()
    calls = []
    args = SimpleNamespace(real_omninav=False, dgx_user="railgun", dgx_password="spark")

    monkeypatch.setattr(module.shutil, "which", lambda name: "plink.exe" if name == "plink" else None)
    monkeypatch.setattr(module.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setenv("DGX_HOST", "10.100.100.128")

    module.stop_remote_omninav_client_for_mock(args, ["omninav_only"])

    assert len(calls) == 1
    assert "railgun@10.100.100.128" in calls[0]
    assert "pkill -f '[o]mninav_model_client_node' 2>/dev/null || true" in calls[0]


def test_real_omninav_live_run_keeps_remote_real_client(monkeypatch):
    module = _load_live_success_benchmark_script()
    calls = []
    args = SimpleNamespace(real_omninav=True, dgx_user="railgun", dgx_password="spark")

    monkeypatch.setattr(module.shutil, "which", lambda name: "plink.exe")
    monkeypatch.setattr(module.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    module.stop_remote_omninav_client_for_mock(args, ["omninav_only"])

    assert calls == []


def test_non_omninav_mock_live_run_keeps_remote_client(monkeypatch):
    module = _load_live_success_benchmark_script()
    calls = []
    args = SimpleNamespace(real_omninav=False, dgx_user="railgun", dgx_password="spark")

    monkeypatch.setattr(module.shutil, "which", lambda name: "plink.exe")
    monkeypatch.setattr(module.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    module.stop_remote_omninav_client_for_mock(args, ["internnav_only"])

    assert calls == []
