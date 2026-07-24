from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.t5_step3_timeout_advisor_node as timeout_advisor
from internvla_t4_sensors.internvla_t4_sensors.step3_arrival_gate import (
    CONTINUE_NAVIGATION,
    REQUEST_CONFIRMATION,
    TERMINATE_ASSISTED,
    arrival_transition,
    completed_motion_action,
)
from scripts.t5_step3_timeout_advisor_node import (
    CAMERA_ORDER,
    PRIMITIVES,
    TimeoutAdvisorError,
    _arrival_outcome,
    _context,
)


ROOT = Path(__file__).resolve().parents[1]


def valid_context() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "motion_timeout_after_confirmed_safe_stop",
        "episode_id": "a::episode-1",
        "reset_generation": 0,
        "trigger_sequence_id": 7,
        "trigger_request_id": "a::episode-1:0:7",
        "expected_sequence_id": 8,
        "excluded_action": 1,
        "instruction": "walk to the doorway",
        "camera_order": list(CAMERA_ORDER),
        "advisor_round": 0,
    }


def valid_arrival_context(*, advisor_round: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "arrival_check_after_completed_motion",
        "episode_id": "a::episode-1",
        "reset_generation": 0,
        "trigger_sequence_id": 7,
        "trigger_request_id": "a::episode-1:0:7",
        "expected_sequence_id": 8,
        "completed_action": 1,
        "instruction": "walk to the doorway",
        "camera_order": list(CAMERA_ORDER),
        "camera_sensor_stamp_ns": 7_000_000_000,
        "minimum_snapshot_sim_stamp_ns": 0,
        "advisor_round": advisor_round,
        "required_confirmations": 2,
    }


def test_timeout_context_is_identity_bound_and_excludes_timed_out_action() -> None:
    value = _context(valid_context())
    assert value["expected_sequence_id"] == 8
    assert set(PRIMITIVES) - {value["excluded_action"]} == {2, 3}


def test_snapshot_capture_uses_trigger_not_next_action_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = valid_context()
    request_path = tmp_path / "request.json"
    observed: dict[str, object] = {}

    def matching_ack(_path: Path, _request_id: str, _remaining: float) -> dict:
        observed.update(json.loads(request_path.read_text(encoding="utf-8")))
        return {"status": "CAPTURED"}

    monkeypatch.setattr(timeout_advisor, "wait_for_matching_ack", matching_ack)
    monkeypatch.setattr(
        timeout_advisor,
        "validate_capture",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    timeout_advisor._request_snapshot(
        context=context,
        request_path=request_path,
        ack_path=tmp_path / "ack.json",
        result_root=tmp_path,
        contract_path=tmp_path / "contract.json",
        deadline=timeout_advisor.time.monotonic() + 1.0,
    )
    assert observed["sequence_id"] == context["trigger_sequence_id"] == 7
    assert observed["sequence_id"] != context["expected_sequence_id"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", "b::episode-1"),
        ("expected_sequence_id", 9),
        ("excluded_action", True),
        ("excluded_action", 4),
        ("camera_order", ["front", "rear"]),
    ],
)
def test_timeout_context_rejects_cross_lane_or_unbounded_values(
    field: str, value: object
) -> None:
    context = valid_context()
    context[field] = value
    with pytest.raises(TimeoutAdvisorError):
        _context(context)


def test_arrival_context_requires_two_identity_bound_rounds() -> None:
    assert _context(valid_arrival_context())["advisor_round"] == 1
    assert _context(valid_arrival_context(advisor_round=2))["advisor_round"] == 2
    invalid = valid_arrival_context(advisor_round=2)
    invalid["minimum_snapshot_sim_stamp_ns"] = -1
    with pytest.raises(TimeoutAdvisorError):
        _context(invalid)


@pytest.mark.parametrize(
    ("decision", "confidence", "fallback", "expected"),
    [
        ("target_found", 0.80, False, "ARRIVED"),
        ("target_found", 0.79, False, "NOT_ARRIVED"),
        ("target_found", 0.99, True, "NOT_ARRIVED"),
        ("abstain", 0.99, False, "NOT_ARRIVED"),
    ],
)
def test_arrival_outcome_is_confidence_bounded_and_fail_closed(
    decision: str, confidence: float, fallback: bool, expected: str
) -> None:
    status, _ = _arrival_outcome(
        SimpleNamespace(
            decision=decision,
            confidence=confidence,
            fallback_used=fallback,
        ),
        0.80,
    )
    assert status == expected


def test_arrival_gate_requires_two_rounds_before_assisted_termination() -> None:
    first = arrival_transition(
        valid_arrival_context(advisor_round=1),
        {"status": "ARRIVED", "snapshot_sim_stamp_ns": 200},
        required_confirmations=2,
    )
    second = arrival_transition(
        valid_arrival_context(advisor_round=2),
        {"status": "ARRIVED", "snapshot_sim_stamp_ns": 230},
        required_confirmations=2,
    )
    negative = arrival_transition(
        valid_arrival_context(advisor_round=1),
        {"status": "NOT_ARRIVED"},
        required_confirmations=2,
    )

    assert first.action == REQUEST_CONFIRMATION
    assert first.snapshot_sim_stamp_ns == 200
    assert second.action == TERMINATE_ASSISTED
    assert second.snapshot_sim_stamp_ns == 230
    assert negative.action == CONTINUE_NAVIGATION


def test_arrival_followup_retains_the_completed_bounded_action() -> None:
    assert completed_motion_action({"action": 1}) == 1
    assert completed_motion_action({"completed_action": 3}) == 3
    assert completed_motion_action({"completed_action": True}) is None
    assert completed_motion_action({"completed_action": 4}) is None


def test_revc_geometry_and_step3_runtime_are_frozen() -> None:
    contract = json.loads(
        (ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["camera_order"] == list(CAMERA_ORDER)
    assert [row["position_F_M_mm"] for row in contract["cameras"]] == [
        [30.0, 51.962, 20.0],
        [60.0, 0.0, 20.0],
        [30.0, -51.962, 20.0],
        [-60.0, 0.0, 20.0],
    ]
    config = (
        ROOT / "configs/slow_models/step3_vl_10b_timeout_advisor.yaml"
    ).read_text(encoding="utf-8")
    assert "revision: 5026053b0c2f5dfaa08fc2d149384162c3c8bca1" in config
    assert "precision_mode: bf16" in config
    assert "redact_raw_text: true" in config
    assert "bind: tcp://10.100.120.122:8200" in config


def test_advisor_has_no_direct_motion_or_terminal_stop_authority() -> None:
    advisor = (ROOT / "scripts/t5_step3_timeout_advisor_node.py").read_text(
        encoding="utf-8"
    )
    client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    assert 'create_publisher(\n            String, "/internvla/t5_step3_timeout_advice"' in advisor
    assert "cmd_vel" not in advisor
    assert '"status": "ADVISE"' not in advisor  # emitted through bounded fields
    assert 'command.action_source = 1' in client
    assert 'command.trajectory_valid = False' in client
    assert '"step3_assisted_stop": True' in client
    assert '"termination_source": "step3_assisted_arrival"' in client
    assert "STEP3_ARRIVAL_REQUIRED_CONFIRMATIONS = 2" in client
    assert "_queued_arrival_confirmation" in advisor
    assert "identity[:4] == self._active_identity[:4]" in advisor
    assert 'self.system2_replan_policy == "observation_bound"' in client
    assert 'INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in client


def test_timeout_advisor_has_bounded_multi_escape_budget() -> None:
    client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    assert 'INTERNVLA_T5_STEP3_TIMEOUT_MAX_INTERVENTIONS", "6"' in client
    assert "1 <= self._step3_timeout_max_interventions <= 6" in client
    post_reset = client.split(
        "def _install_post_reset_sensor_barrier", maxsplit=1
    )[1].split("def _reset_motion_gate", maxsplit=1)[0]
    assert "self._step3_timeout_interventions = 0" in post_reset
    assert "self._step3_arrival_completed_actions = 0" in post_reset
    assert "self._step3_arrival_checks = 0" in post_reset


def test_x86_advisor_runs_inside_the_existing_ros_container() -> None:
    runner = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text(
        encoding="utf-8"
    )
    assert "setsid docker exec --user admin --workdir \"$root\"" in runner
    assert "pyzmq-27.1.0.dist-info" in runner
    assert '-e ROS_LOCALHOST_ONLY=0 -e "PYTHONPATH=$root:$advisor_python_deps"' in runner
    assert "source /opt/ros/jazzy/setup.bash" in runner
    assert "source /workspaces/isaac/install/setup.bash" in runner
    assert runner.index('python3 -c "import rclpy, zmq, slow_planner') < runner.index(
        'printf "%s\\n" "$$" >"$STEP3_TIMEOUT_ADVISOR_PID_FILE"'
    )
    assert "step3_timeout_advisor.container.pid" in runner
    assert "stop_step3_timeout_advisor" in runner
