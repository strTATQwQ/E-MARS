from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_t5_evaluation_starvation.py"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_rows(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _decision(
    *,
    sequence: int,
    stop: bool = False,
    model_stop: bool = False,
    action_source: int = 1,
) -> dict[str, object]:
    return {
        "episode_id": "a::7",
        "reset_generation": 0,
        "sequence_id": sequence,
        "request_id": f"a::7:0:{sequence}",
        "discrete_action": 0 if model_stop else -1,
        "model_discrete_action": 0 if model_stop else 1,
        "model_stop": model_stop,
        "stop": stop,
        "action_source": action_source,
        "status_message": "ok",
    }


def _gate_terminal(sequence: int) -> dict[str, object]:
    return {
        "episode_id": "a::7",
        "reset_generation": 0,
        "sequence_id": sequence,
        "request_id": f"a::7:0:{sequence}",
        "discrete_action": -1,
        "model_discrete_action": 1,
        "model_stop": False,
        "stop": True,
        "motion_observation_gate_only": True,
        "motion_observation_gate_decision": {
            "kind": "hold_post_stop",
            "permits_model_step": False,
        },
        "status_message": "waiting for post-stop sensors",
    }


def _fixture(
    root: Path,
    client_rows: list[dict[str, object]],
    *,
    trajectory_id: str = "99_7",
    desired_linear: float = 0.2,
) -> None:
    evaluator = root / "remote/x86/evaluator/attempt"
    _write(
        evaluator / "per_episode.json",
        {
            "completed_episode_count": 1,
            "episodes": [
                {
                    "ordinal": 1,
                    "trajectory_id": trajectory_id,
                    "duration_sec": 10.0,
                    "step_count": 100,
                    "termination_reason": (
                        "success" if any(row.get("model_stop") for row in client_rows) else "not_reach_goal"
                    ),
                    "success": any(row.get("model_stop") for row in client_rows),
                    "TL": 1.0,
                    "shortest_path_length": 2.0,
                    "official_metrics": {
                        "ne_m": 1.0,
                        "os": 1,
                        "sr": int(any(row.get("model_stop") for row in client_rows)),
                        "spl": 0.5,
                    },
                }
            ],
        },
    )
    _write_rows(evaluator / "go2_runtime_audit.jsonl", [{"physics_hz": 20.0}])
    _write_rows(root / "remote/dgx/client/client_records.jsonl", client_rows)
    _write_rows(
        root / "remote/dgx/onboard/controller_records.jsonl",
        [
            {
                "episode_id": "a::7",
                "reset_generation": 0,
                "state_only": False,
                "identity_valid": True,
                "motion_enabled": True,
                "command_fresh": True,
                "desired_linear_x": desired_linear,
                "desired_angular_z": 0.5,
                "nan_detected": False,
                "emergency_stop": False,
                "physical_collision": False,
            }
        ],
    )
    _write(
        root / "remote/dgx/onboard/controller_summary.json",
        {
            "camera_frame": "camera",
            "camera_hfov_deg": 70.0,
            "camera_translation_from_base_m": [0.2, 0.0, 0.2],
            "stale_motion_execution_count": 0,
            "direct_motion_bypass_count": 0,
            "cmd_vel_quantization_count": 0,
            "nan_count": 0,
            "physical_collision_count": 0,
        },
    )
    _write(
        root / "remote/dgx/model/model_weight_audit.json",
        {
            "model_revision": "model",
            "checkpoint_revision": "checkpoint",
            "inventory_sha256": "inventory",
        },
    )
    _write(
        root / "remote/x86/ordered_episode_manifest.json",
        {"dataset_sha256": "dataset"},
    )


def _run(
    tmp_path: Path,
    specifications: list[str],
    *extra: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    output = tmp_path / "analysis.json"
    command = [sys.executable, str(ANALYZER)]
    for specification in specifications:
        command.extend(["--run", specification])
    command.extend([*extra, "--output", str(output)])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_safe_stop_terminal_conflation_is_primary_hard_failure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=index) for index in range(4)] + [_gate_terminal(3)])
    evaluator_log = tmp_path / "eval.log"
    evaluator_log.write_text(
        "INTERNVLA_MODEL_ACTION_OK discrete_action: -1\n"
        "now action [{'action': [0], 'ideal_flag': True}]\n"
        "[1/1] finish: [trajectory_id:99_7][result:not_reach_goal]\n",
        encoding="utf-8",
    )

    completed, payload = _run(
        tmp_path,
        [f"candidate={run}"],
        "--evaluator-log",
        f"candidate={evaluator_log}",
    )

    assert completed.returncode == 2
    assert payload["status"] == "EVALUATION_INVALID"
    aggregate = payload["runs"][0]["aggregate"]
    assert aggregate["evaluator_step_count"] == 100
    assert aggregate["true_model_decisions"] == 4
    assert aggregate["gate_holds"] == {
        "total": 1,
        "by_kind": {"hold_post_stop": 1},
    }
    assert aggregate["terminal_stop_without_model_stop_count"] == 1
    assert aggregate["safe_stop_terminal_conflation_count"] == 1
    assert aggregate["false_termination_count"] == 1
    episode = payload["runs"][0]["episodes"][0]
    assert episode["failures"][:2] == [
        "safe_stop_terminal_conflation",
        "terminal_stop_without_model_stop",
    ]
    assert episode["termination"]["last_response"]["discrete_action"] == -1
    assert episode["termination"]["last_response"]["model_stop"] is False
    assert episode["termination"]["terminal_cause"] == "safe_stop_terminal_conflation"
    assert episode["termination"]["ipc_step_error_termination"] is False


def test_gate_only_client_stop_without_evaluator_chain_is_not_conflation(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=0), _gate_terminal(0)])

    completed, payload = _run(tmp_path, [f"candidate={run}"])

    assert completed.returncode == 2
    episode = payload["runs"][0]["episodes"][0]
    assert episode["termination"]["terminal_cause"] == (
        "client_stop_without_model_stop"
    )
    assert episode["termination"]["safe_stop_terminal_conflation"] is False
    assert episode["failures"][0] == "terminal_stop_without_model_stop"


def test_natural_model_stop_satisfies_decision_budget(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _fixture(
        run,
        [_decision(sequence=0), _decision(sequence=1, stop=True, model_stop=True)],
    )

    completed, payload = _run(tmp_path, [f"candidate={run}"])

    assert completed.returncode == 0
    episode = payload["runs"][0]["episodes"][0]
    assert episode["true_model_decisions"] == 2
    assert episode["natural_model_stop"] is True
    assert episode["evaluation_valid"] is True


def test_fifty_decisions_without_false_stop_satisfy_budget(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=index) for index in range(50)])

    completed, payload = _run(tmp_path, [f"candidate={run}"])

    assert completed.returncode == 0
    episode = payload["runs"][0]["episodes"][0]
    assert episode["true_model_decisions"] == 50
    assert episode["natural_model_stop"] is False
    assert episode["termination"]["false_termination"] is False


def test_velocity_boundary_remains_a_hard_gate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _fixture(
        run,
        [_decision(sequence=0, stop=True, model_stop=True)],
        desired_linear=0.3,
    )

    completed, payload = _run(tmp_path, [f"candidate={run}"])

    assert completed.returncode == 2
    episode = payload["runs"][0]["episodes"][0]
    assert "safety_or_velocity_boundary_failure" in episode["failures"]
    assert episode["safety"]["checks"]["linear_velocity_within_bound"] is False


def test_paired_runs_require_identical_manifest_model_and_camera(tmp_path: Path) -> None:
    lane_a = tmp_path / "lane_a"
    lane_b = tmp_path / "lane_b"
    rows = [_decision(sequence=0, stop=True, model_stop=True)]
    _fixture(lane_a, rows, trajectory_id="99_7")
    _fixture(lane_b, rows, trajectory_id="100_7")

    completed, payload = _run(
        tmp_path, [f"lane_a={lane_a}", f"lane_b={lane_b}"]
    )

    assert completed.returncode == 2
    assert payload["paired_gate"]["status"] == "FAIL"
    assert payload["paired_gate"]["same_manifest_model_camera"] is False
    assert payload["paired_gate"]["all_runs_evaluation_valid"] is True


def test_optional_evaluator_log_corroborates_without_embedding_content(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=0), _gate_terminal(0)])
    evaluator_log = tmp_path / "eval.log"
    evaluator_log.write_text(
        "INTERNVLA_MODEL_ACTION_OK discrete_action: -1\n"
        "now action [{'action': [0], 'ideal_flag': True}]\n"
        "[1/1] finish: [trajectory_id:99_7][result:not_reach_goal]\n",
        encoding="utf-8",
    )

    completed, payload = _run(
        tmp_path,
        [f"candidate={run}"],
        "--evaluator-log",
        f"candidate={evaluator_log}",
    )

    assert completed.returncode == 2
    evidence = payload["runs"][0]["evaluator_log_corroboration"]
    assert evidence["available"] is True
    assert (
        evidence[
            "discrete_minus_one_then_action_zero_then_not_reach_goal_event_count"
        ]
        == 1
    )
    assert evidence["raw_log_content_embedded"] is False
    assert evidence["ipc_step_error_marker_count"] == 0
    assert evidence["episode_terminal_events"][0]["terminal_cause"] == (
        "gate_response_chain"
    )
    episode = payload["runs"][0]["episodes"][0]
    assert episode["termination"]["terminal_cause"] == (
        "safe_stop_terminal_conflation"
    )


def test_ipc_error_has_priority_over_final_gate_response_adjacency(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=0), _gate_terminal(0)])
    evaluator_log = tmp_path / "eval.log"
    evaluator_log.write_text(
        "INTERNVLA_MODEL_ACTION_OK discrete_action: -1\n"
        'INTERNVLA_LOCAL_IPC_STEP_ERROR {"episode_ordinal": 0, '
        '"error": "RuntimeError: System 1 queue has no identity-bound absolute '
        'target; fresh trajectory required", "safe_stop": true}\n'
        "now action [{'action': [0], 'ideal_flag': True}]\n"
        "[1/1] finish: [trajectory_id:99_7][result:not_reach_goal]\n",
        encoding="utf-8",
    )

    completed, payload = _run(
        tmp_path,
        [f"candidate={run}"],
        "--evaluator-log",
        f"candidate={evaluator_log}",
    )

    assert completed.returncode == 2
    aggregate = payload["runs"][0]["aggregate"]
    assert aggregate["ipc_step_error_termination_count"] == 1
    assert aggregate["safe_stop_terminal_conflation_count"] == 0
    assert aggregate["terminal_causes"] == {"ipc_step_error": 1}
    episode = payload["runs"][0]["episodes"][0]
    assert episode["termination"]["terminal_cause"] == "ipc_step_error"
    assert episode["termination"]["ipc_step_error_code"] == (
        "system1_queue_missing_identity_bound_target"
    )
    assert episode["termination"]["direct_gate_response_chain"] is False
    assert episode["failures"][:2] == [
        "ipc_step_error_termination",
        "terminal_stop_without_model_stop",
    ]


def test_ipc_error_classifies_evaluator_stop_without_client_stop(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=index) for index in range(50)])
    evaluator_log = tmp_path / "eval.log"
    evaluator_log.write_text(
        'INTERNVLA_LOCAL_IPC_STEP_ERROR {"episode_ordinal": 0, '
        '"error": "RuntimeError: fresh recovery trajectory timed out", '
        '"safe_stop": true}\n'
        "now action [{'action': [0], 'ideal_flag': True}]\n"
        "[1/1] finish: [trajectory_id:99_7][result:success]\n",
        encoding="utf-8",
    )

    completed, payload = _run(
        tmp_path,
        [f"candidate={run}"],
        "--evaluator-log",
        f"candidate={evaluator_log}",
    )

    assert completed.returncode == 2
    episode = payload["runs"][0]["episodes"][0]
    assert episode["termination"]["client_terminal_stop_without_model_stop"] is False
    assert episode["termination"]["terminal_stop_without_model_stop"] is True
    assert episode["termination"]["terminal_cause"] == "ipc_step_error"
    assert episode["termination"]["ipc_step_error_code"] == (
        "fresh_recovery_trajectory_timeout"
    )
    assert episode["termination"]["safe_stop_terminal_conflation"] is False
    assert episode["termination"]["false_termination"] is True


def test_historical_inputs_are_descriptive_not_retroactive_gates(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _fixture(run, [_decision(sequence=0, stop=True, model_stop=True)])
    t0 = tmp_path / "t0.json"
    _write(
        t0,
        {
            "protocol_requests_parsed": 70,
            "action_distribution": {"stop_0": 1},
            "metrics": {"TL": 2.0, "NE": 1.0, "OS": 1.0, "SR": 1.0, "SPL": 0.5},
            "episodes": [
                {
                    "episode_key": "1_1",
                    "steps": 70,
                    "wall_time_seconds": 10.0,
                    "termination_reason": "success",
                    "NE": 1.0,
                    "OS": 1.0,
                    "SR": 1.0,
                    "SPL": 0.5,
                }
            ],
        },
    )
    t2 = tmp_path / "t2"
    _write(
        t2 / "per_episode.json",
        {"episodes": [{"ordinal": 1, "trajectory_id": "1_1", "step_count": 70, "duration_sec": 10.0, "success": True, "termination_reason": "success"}]},
    )
    _write_rows(
        t2 / "active_records.jsonl",
        [{"reset_generation": 0, "action_source": 1, "model_action": 0}],
    )
    _write(t2 / "result.json", {"val_unseen": {"SR": 1.0, "NE": 1.0}})
    t3 = tmp_path / "t3"
    _write(
        t3 / "per_episode.json",
        {"episodes": [{"ordinal": 1, "trajectory_id": "1_1", "step_count": 200, "duration_sec": 2.0, "success": True, "termination_reason": "success"}]},
    )
    _write_rows(
        t3 / "client_records.jsonl",
        [{"reset_generation": 0, "action_source": 1, "model_stop": True}],
    )
    _write_rows(t3 / "go2_runtime_audit.jsonl", [{"physics_hz": 200.0}])
    _write(t3 / "result.json", {"val_unseen": {"SR": 1.0, "NE": 1.0}})

    completed, payload = _run(
        tmp_path,
        [f"candidate={run}"],
        "--t0-status",
        str(t0),
        "--t2-run",
        str(t2),
        "--t3-run",
        str(t3),
    )

    assert completed.returncode == 0
    assert [item["label"] for item in payload["historical_references"]] == [
        "t0_official_pilot",
        "t2_go2_nav2_pilot",
        "t3_obstacle_aware_continuous_pilot",
    ]
    assert len(payload["historical_differential"]) == 3
    assert all(
        item["strict_t5_gate_applied"] is False
        for item in payload["historical_references"]
    )
