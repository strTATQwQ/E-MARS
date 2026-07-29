from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in (
    "internvla_ros2",
    "internvla_nav2_adapter",
    "internvla_t4_recovery",
    "internvla_t4_sensors",
):
    sys.path.insert(0, str(ROOT / package))

from internvla_ros2.client_node import ClientFailure, InternVLAClientNode  # noqa: E402
from internvla_ros2.recovery_contract import (  # noqa: E402
    OP_REQUEST_REPLAN,
    goal_identity,
    operation_identity,
    system2_primitive_signature,
    trajectory_signature,
)
from internvla_ros2_msgs.srv import RecoveryControl  # noqa: E402
from internvla_nav2_adapter.active_node import ActiveNav2Adapter  # noqa: E402
from internvla_t4_recovery.adapter_node import T4RecoveryAdapter  # noqa: E402
from internvla_t4_recovery.recovery_node import (  # noqa: E402
    RecoverySupervisor,
    _no_progress_recovery_eligible,
)
from internvla_t4_sensors import client_node as t4_client_module  # noqa: E402
from internvla_t4_sensors.client_node import T4OdometryClientNode  # noqa: E402


def _command(
    *,
    episode: str = "a::259",
    generation: int = 1,
    sequence: int = 17,
    source: int = 3,
    action: int = 1,
    observation_digest: str = "observation:17",
) -> SimpleNamespace:
    return SimpleNamespace(
        episode_id=episode,
        reset_generation=generation,
        sequence_id=sequence,
        request_id=f"{episode}:{generation}:{sequence}",
        observation_digest=observation_digest,
        action_source=source,
        discrete_action=action,
        stop=False,
        local_path=SimpleNamespace(poses=[]),
    )


def _resolution(message: str = "ok", *, goal: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=0,
        status_message=message,
        discrete_action=-1,
        stop=False,
        nav2_goal_sent=goal,
        nav2_plan_valid=goal,
    )


def _bare_client() -> T4OdometryClientNode:
    client = T4OdometryClientNode.__new__(T4OdometryClientNode)
    client._t5_sim_time_semantics = True
    client._t4_pose_lock = threading.Condition()
    client._system1_queue_hold_identity = None
    client._system1_queue_hold_count = 0
    client._pending_typed_replan = None
    client._last_typed_replan_sequence = -1
    client.allow_system2_recovery_replan = True
    client.episode_id = "a::259"
    client.reset_generation = 1
    client.replan_audit_path = None
    client._t4_odometry = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.25, y=-0.5),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
    )
    client._t4_odometry_serial = 2
    client._t4_odometry_barrier_serial = 1
    client._t4_odometry_stamp_ns = 200
    client._reset_sim_barrier_ns = 100
    client._sensor_future_tolerance_ns = 100
    client.navigation_odometry_timeout = 0.5
    client.last_committed_sequence = -1
    client._step3_timeout_condition = threading.Condition()
    client._step3_model_stop_escape_burst_count = 0
    client._step3_model_stop_escape_motion = None
    return client


def _task_state_advice_client(
    *,
    kind: str = "task_state_checkpoint_after_completed_motion",
) -> T4OdometryClientNode:
    client = _bare_client()
    client._step3_timeout_enabled = True
    client._step3_timeout_condition = threading.Condition()
    client._step3_timeout_pending = {
        "kind": kind,
        "episode_id": "a::259",
        "reset_generation": 1,
        "expected_sequence_id": 17,
        "trigger_sequence_id": 16,
        "trigger_request_id": "a::259:1:16",
        "stop_token": "step3-task-state:test",
    }
    client._step3_timeout_advice = {
        "status": "ADVISE",
        "advised_action": 2,
        "confidence": 0.8,
        "snapshot_id": "a::259::1::16",
        "service_wall_latency_sec": 0.1,
        "camera_count": 4,
    }
    client._step3_timeout_interventions = 0
    client._step3_task_state_control = False
    client._step3_model_stop_escapes = 0
    client._step3_model_stop_escape_burst_count = 0
    client._step3_timeout_override = None
    client._step3_model_refresh_pending = None
    return client


def test_oracle_rejected_model_stop_enters_existing_bounded_escape_gate() -> None:
    client = _bare_client()
    client._step3_timeout_enabled = True
    client._step3_last_completed_action = 2
    published: list[dict[str, object]] = []
    stopped: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    def publish_arrival(**kwargs: object) -> bool:
        published.append(dict(kwargs))
        return True

    client._publish_step3_arrival_context = publish_arrival  # type: ignore[method-assign]
    client._safe_stop = (  # type: ignore[method-assign]
        lambda _status, message: stopped.append(message)
    )
    client._record_motion_gate_event = (  # type: ignore[method-assign]
        lambda event, _token, payload: events.append((event, payload))
    )
    result: dict[str, object] = {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 44,
        "request_id": "a::259:1:44",
        "stop": False,
        "model_stop": True,
        "model_discrete_action": 0,
        "geometric_success": False,
    }

    assert client._gate_internvla_model_stop(result, oracle_rejected=True)

    assert result["stop"] is False
    assert result["model_stop"] is True
    assert result["internvla_stop_candidate"] is True
    assert result["step3_arrival_confirmation_pending"] is True
    assert len(published) == 1
    assert published[0]["advisor_round"] == 1
    assert published[0]["pending"] == {
        "sequence_id": 44,
        "request_id": "a::259:1:44",
        "action": 2,
        "model_stop_candidate": True,
        "model_discrete_action": 0,
    }
    assert len(stopped) == 1
    assert events[0][0] == "internvla_model_stop_candidate"


def test_failed_model_stop_escape_is_excluded_after_intervening_sequences() -> None:
    client = _bare_client()
    client._step3_timeout_enabled = True
    client._step3_last_completed_action = 3
    client._step3_model_stop_escape_burst_count = 1
    client._step3_model_stop_escape_motion = {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 41,
        "request_id": "a::259:1:41",
        "action": 1,
        "timed_out": False,
    }
    events: list[tuple[str, dict[str, object]]] = []
    published: list[dict[str, object]] = []
    client._record_motion_gate_event = (  # type: ignore[method-assign]
        lambda event, _token, payload: events.append((event, payload))
    )
    client._safe_stop = lambda *_args: None  # type: ignore[method-assign]
    client._publish_step3_arrival_context = (  # type: ignore[method-assign]
        lambda **kwargs: published.append(dict(kwargs)) or True
    )
    timeout = t4_client_module.GateDecision(
        kind=t4_client_module.DECISION_SAFE_STOP_TIMEOUT,
        reason="measured motion missed the sim-time deadline",
        permits_model_step=False,
        requires_safe_stop=True,
        keep_safe_stop=True,
    )

    client._settle_step3_model_stop_escape_motion(
        timeout,
        {
            "sequence_id": 41,
            "request_id": "a::259:1:41",
            "action": 1,
        },
    )

    assert client._step3_model_stop_escape_motion == {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 41,
        "request_id": "a::259:1:41",
        "action": 1,
        "timed_out": True,
        "minimum_stop_sequence_id": 42,
    }
    assert client._step3_model_stop_escape_burst_count == 1
    result: dict[str, object] = {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 45,
        "request_id": "a::259:1:45",
        "stop": False,
        "model_stop": True,
        "model_discrete_action": 0,
        "geometric_success": False,
    }

    assert client._gate_internvla_model_stop(result, oracle_rejected=True)
    assert published[0]["pending"] == {
        "sequence_id": 45,
        "request_id": "a::259:1:45",
        "action": 3,
        "model_stop_candidate": True,
        "model_discrete_action": 0,
        "excluded_action": 1,
        "recent_timeout_sequence_id": 41,
    }
    assert client._step3_model_stop_escape_motion is None
    assert [event for event, _payload in events] == [
        "step3_model_stop_escape_timed_out",
        "internvla_model_stop_candidate",
    ]


def test_model_stop_escape_exclusion_waits_for_next_qualifying_stop() -> None:
    client = _bare_client()
    timed_out = {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 41,
        "request_id": "a::259:1:41",
        "action": 1,
        "timed_out": True,
        "minimum_stop_sequence_id": 42,
    }
    client._step3_model_stop_escape_motion = dict(timed_out)
    assert client._consume_step3_model_stop_timeout(41) is None
    assert client._step3_model_stop_escape_motion == timed_out
    assert client._consume_step3_model_stop_timeout(45) == timed_out
    assert client._step3_model_stop_escape_motion is None

    client._step3_model_stop_escape_motion = {
        key: value
        for key, value in timed_out.items()
        if key not in {"timed_out", "minimum_stop_sequence_id"}
    }
    client._step3_model_stop_escape_motion["timed_out"] = False
    client._record_motion_gate_event = lambda *_args: None  # type: ignore[method-assign]
    complete = t4_client_module.GateDecision(
        kind=t4_client_module.DECISION_SAFE_STOP_COMPLETE,
        reason="measured motion completed",
        permits_model_step=True,
        requires_safe_stop=True,
        keep_safe_stop=True,
    )
    client._settle_step3_model_stop_escape_motion(
        complete,
        {
            "sequence_id": 41,
            "request_id": "a::259:1:41",
            "action": 1,
        },
    )
    assert client._step3_model_stop_escape_motion is None


def test_model_stop_escape_burst_rearms_only_after_effective_measured_motion() -> None:
    client = _bare_client()
    client._step3_model_stop_escape_burst_count = 1
    events: list[tuple[str, dict[str, object]]] = []
    client._record_motion_gate_event = (  # type: ignore[method-assign]
        lambda event, _token, payload: events.append((event, payload))
    )
    pending = {
        "sequence_id": 41,
        "request_id": "a::259:1:41",
        "action": 2,
    }
    timeout = t4_client_module.GateDecision(
        kind=t4_client_module.DECISION_SAFE_STOP_TIMEOUT,
        reason="measured motion missed the sim-time deadline",
        permits_model_step=False,
        requires_safe_stop=True,
        keep_safe_stop=True,
    )
    stale = t4_client_module.GateDecision(
        kind=t4_client_module.DECISION_SAFE_STOP_STALE,
        reason="odometry regressed",
        permits_model_step=False,
        requires_safe_stop=True,
        keep_safe_stop=True,
    )

    client._settle_step3_model_stop_escape_motion(timeout, pending)
    client._settle_step3_model_stop_escape_motion(stale, pending)

    assert client._step3_model_stop_escape_burst_count == 1
    assert events == []

    timed_out_state = {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 41,
        "request_id": "a::259:1:41",
        "action": 2,
        "timed_out": True,
        "minimum_stop_sequence_id": 42,
    }
    client._step3_model_stop_escape_motion = dict(timed_out_state)

    complete = t4_client_module.GateDecision(
        kind=t4_client_module.DECISION_SAFE_STOP_COMPLETE,
        reason="measured motion reached the preregistered completion threshold",
        permits_model_step=True,
        requires_safe_stop=True,
        keep_safe_stop=True,
        progress=0.24,
        required_progress=0.2,
        commanded_progress=0.25,
    )
    client._settle_step3_model_stop_escape_motion(complete, pending)

    assert client._step3_model_stop_escape_burst_count == 0
    assert client._step3_model_stop_escape_motion == timed_out_state
    assert events == [
        (
            "step3_model_stop_escape_rearmed_after_measured_motion",
            {
                "sequence_id": 41,
                "request_id": "a::259:1:41",
                "action": 2,
                "previous_burst_escape_count": 1,
                "progress": 0.24,
                "required_progress": 0.2,
            },
        )
    ]


def test_model_stop_escape_is_blocked_until_measured_motion_rearms_burst() -> None:
    client = _bare_client()
    client._step3_timeout_enabled = True
    client._step3_timeout_interventions = 0
    client._step3_model_stop_escapes = 0
    client._step3_timeout_override = None
    client._step3_model_refresh_pending = None
    client._record_motion_gate_event = lambda *_args: None  # type: ignore[method-assign]

    def install_escape(sequence_id: int) -> None:
        client._step3_timeout_pending = {
            "kind": "arrival_check_after_completed_motion",
            "episode_id": "a::259",
            "reset_generation": 1,
            "expected_sequence_id": sequence_id,
            "trigger_sequence_id": sequence_id - 1,
            "trigger_request_id": f"a::259:1:{sequence_id - 1}",
            "stop_token": f"step3-model-stop:{sequence_id}",
            "advisor_round": 2,
            "model_stop_candidate": True,
            "first_advised_action": 2,
        }
        client._step3_timeout_advice = {
            "status": "NOT_ARRIVED",
            "advised_action": 2,
            "confidence": 0.9,
            "snapshot_id": f"a::259::1::{sequence_id}",
            "snapshot_sim_stamp_ns": sequence_id * 1_000_000_000,
            "service_wall_latency_sec": 0.1,
            "camera_count": 4,
        }

    install_escape(17)
    first = _command(sequence=17, action=0)
    client._apply_step3_timeout_advice(first)

    assert first.discrete_action == 2
    assert client._step3_model_stop_escapes == 1
    assert client._step3_model_stop_escape_burst_count == 1

    install_escape(18)
    blocked = _command(sequence=18, action=0)
    client._apply_step3_timeout_advice(blocked)

    assert blocked.discrete_action == 0
    assert client._step3_timeout_pending is not None
    assert client._step3_model_stop_escapes == 1
    assert client._step3_model_stop_escape_burst_count == 1

    complete = t4_client_module.GateDecision(
        kind=t4_client_module.DECISION_SAFE_STOP_COMPLETE,
        reason="measured motion reached the preregistered completion threshold",
        permits_model_step=True,
        requires_safe_stop=True,
        keep_safe_stop=True,
        progress=0.24,
        required_progress=0.2,
        commanded_progress=0.25,
    )
    client._settle_step3_model_stop_escape_motion(
        complete,
        {
            "sequence_id": 17,
            "request_id": "a::259:1:17",
            "action": 2,
        },
    )
    client._apply_step3_timeout_advice(blocked)

    assert blocked.discrete_action == 2
    assert client._step3_model_stop_escapes == 2
    assert client._step3_model_stop_escape_burst_count == 1


def test_timed_out_escape_allows_one_different_bounded_retry() -> None:
    client = _bare_client()
    client._step3_timeout_enabled = True
    client._step3_timeout_interventions = 0
    client._step3_model_stop_escapes = 1
    client._step3_model_stop_escape_burst_count = 1
    client._step3_timeout_override = None
    client._step3_model_refresh_pending = None
    client._record_motion_gate_event = lambda *_args: None  # type: ignore[method-assign]
    client._step3_timeout_pending = {
        "kind": "arrival_check_after_completed_motion",
        "episode_id": "a::259",
        "reset_generation": 1,
        "expected_sequence_id": 18,
        "trigger_sequence_id": 17,
        "trigger_request_id": "a::259:1:17",
        "stop_token": "step3-model-stop:18",
        "advisor_round": 2,
        "model_stop_candidate": True,
        "first_advised_action": 3,
        "excluded_action": 2,
        "recent_timeout_sequence_id": 17,
    }
    client._step3_timeout_advice = {
        "status": "NOT_ARRIVED",
        "advised_action": 3,
        "confidence": 0.9,
        "snapshot_id": "a::259::1::18",
        "snapshot_sim_stamp_ns": 18_000_000_000,
        "service_wall_latency_sec": 0.1,
        "camera_count": 4,
    }
    retry = _command(sequence=18, action=0)

    client._apply_step3_timeout_advice(retry)

    assert retry.discrete_action == 3
    assert client._step3_model_stop_escapes == 2
    assert client._step3_model_stop_escape_burst_count == 2


def test_model_stop_confirmation_preserves_exclusion_across_two_rounds() -> None:
    client = _bare_client()
    client._step3_timeout_enabled = True
    client._step3_arrival_checks = 0
    client._camera_sensor_stamp_ns = 7_000_000_000
    client._last_navigation_instruction = "walk to the doorway"
    client.next_sequence_id = 42
    published: list[dict[str, object]] = []
    client.step3_timeout_context_publisher = SimpleNamespace(
        publish=lambda message: published.append(json.loads(message.data))
    )
    client._record_motion_gate_event = lambda *_args: None  # type: ignore[method-assign]
    pending: dict[str, object] = {
        "sequence_id": 41,
        "request_id": "a::259:1:41",
        "action": 3,
        "model_stop_candidate": True,
        "model_discrete_action": 0,
        "excluded_action": 1,
        "recent_timeout_sequence_id": 40,
    }

    assert client._publish_step3_arrival_context(
        token="a::259:1:model-stop:41",
        pending=pending,
        advisor_round=1,
    )
    round_one = dict(client._step3_timeout_pending or {})
    round_one["first_advised_action"] = 2
    assert client._publish_step3_arrival_context(
        token="a::259:1:model-stop:41",
        pending=round_one,
        advisor_round=2,
        minimum_snapshot_sim_stamp_ns=7_000_000_100,
    )

    assert [row["excluded_action"] for row in published] == [1, 1]
    assert [row["recent_timeout_sequence_id"] for row in published] == [40, 40]
    assert published[1]["first_advised_action"] == 2


def test_task_state_checkpoint_is_semantic_only_over_model_standstill() -> None:
    client = _task_state_advice_client()
    events: list[tuple[str, dict[str, object]]] = []
    client._record_motion_gate_event = (  # type: ignore[method-assign]
        lambda event, _token, payload: events.append((event, payload))
    )
    hold = _command(sequence=17, action=-1)

    client._apply_step3_timeout_advice(hold)

    assert hold.discrete_action == -1
    assert client._step3_timeout_pending is None
    assert client._step3_timeout_advice is None
    assert client._step3_timeout_interventions == 0
    assert client._step3_timeout_override is None
    assert client._step3_model_refresh_pending is None
    assert [event for event, _payload in events] == [
        "step3_task_state_checkpoint_semantic_only"
    ]
    assert events[0][1]["model_action_retained"] == -1
    assert events[0][1]["shadow_advised_action"] == 2
    assert events[0][1]["control_effect"] == "none"


def test_motion_timeout_advice_invalidates_only_its_stale_action_queue() -> None:
    client = _task_state_advice_client(
        kind="motion_timeout_after_confirmed_safe_stop"
    )
    client._record_motion_gate_event = lambda *_args: None  # type: ignore[method-assign]
    motion = _command(sequence=17, action=1)

    client._apply_step3_timeout_advice(motion)

    assert motion.discrete_action == 2
    assert client._step3_model_refresh_pending == {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 17,
        "stop_token": "step3-task-state:test",
    }


def test_task_state_checkpoint_retains_normal_internvla_motion() -> None:
    client = _task_state_advice_client()
    events: list[tuple[str, dict[str, object]]] = []
    client._record_motion_gate_event = (  # type: ignore[method-assign]
        lambda event, _token, payload: events.append((event, payload))
    )

    motion = _command(sequence=17, action=3)
    client._apply_step3_timeout_advice(motion)

    assert motion.discrete_action == 3
    assert client._step3_timeout_pending is None
    assert client._step3_timeout_advice is None
    assert client._step3_timeout_interventions == 0
    assert client._step3_timeout_override is None
    assert client._step3_model_refresh_pending is None
    assert events == [
        (
            "step3_task_state_checkpoint_semantic_only",
            {
                "episode_id": "a::259",
                "reset_generation": 1,
                "sequence_id": 17,
                "trigger_sequence_id": 16,
                "trigger_request_id": "a::259:1:16",
                "model_action_retained": 3,
                "shadow_advised_action": 2,
                "confidence": 0.8,
                "snapshot_id": "a::259::1::16",
                "service_wall_latency_sec": 0.1,
                "camera_count": 4,
                "control_effect": "none",
            },
        )
    ]


def test_task_state_checkpoint_can_apply_one_bounded_control_override() -> None:
    client = _task_state_advice_client()
    client._step3_task_state_control = True
    events: list[tuple[str, dict[str, object]]] = []
    client._record_motion_gate_event = (  # type: ignore[method-assign]
        lambda event, _token, payload: events.append((event, payload))
    )
    motion = _command(sequence=17, action=3)

    client._apply_step3_timeout_advice(motion)

    assert motion.discrete_action == 2
    assert motion.action_source == 1
    assert client._step3_timeout_interventions == 0
    assert client._step3_timeout_override is not None
    assert client._step3_timeout_override["intervention_kind"] == (
        "task_state_checkpoint"
    )
    assert client._step3_model_refresh_pending == {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 17,
        "stop_token": "step3-task-state:test",
    }
    assert [event for event, _payload in events] == [
        "step3_task_state_checkpoint_control_applied"
    ]


def _primitive(
    *,
    action: int = 2,
    sequence: int = 17,
    generation: int = 1,
    observation_digest: str = "observation:17",
) -> object:
    return system2_primitive_signature(
        action=action,
        episode_id="a::259",
        reset_generation=generation,
        sequence_id=sequence,
        observation_digest=observation_digest,
        x=1.25,
        y=-0.5,
        yaw_rad=0.0,
    )


def _pending_replan(old_signature: object, *, operation: str) -> dict[str, object]:
    return {
        "request_index": 4,
        "episode_id": "a::259",
        "reset_generation": 1,
        "goal_id": "goal:test",
        "recovery_id": "recovery:test",
        "operation_id": operation,
        "deadline_ns": 1_000,
        "active_sequence_id": 17,
        "excluded_absolute": {old_signature.absolute_sha256},
        "excluded_shape": {old_signature.shape_sha256},
        "rejection_count": 0,
    }


def test_shared_system2_primitive_signature_binds_the_source_observation() -> None:
    signature = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=17,
        observation_digest="observation:17",
        x=1.25,
        y=-0.5,
        yaw_rad=0.0,
    )
    same_action = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=18,
        observation_digest="observation:17",
        x=1.251,
        y=-0.499,
        yaw_rad=0.001,
    )
    different_action = _primitive(action=3, sequence=18)
    new_observation = _primitive(
        action=2,
        sequence=18,
        observation_digest="observation:18",
    )

    assert same_action.absolute_sha256 != signature.absolute_sha256
    assert same_action.shape_sha256 == signature.shape_sha256
    assert different_action.shape_sha256 != signature.shape_sha256
    assert new_observation.shape_sha256 != signature.shape_sha256


def test_client_rejects_same_observation_and_accepts_new_action_observation_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_signature = _primitive()
    client = _bare_client()
    client._pending_typed_replan = _pending_replan(
        old_signature, operation="operation:same"
    )
    events: list[str] = []
    client._record_replan = (  # type: ignore[method-assign]
        lambda event, *_args, **_extra: events.append(event)
    )
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 100)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=False),
    )

    response = client._resolve_nav2(_command(sequence=18, source=1, action=2))

    assert response.nav2_goal_sent is False
    assert client._pending_typed_replan is not None
    assert "old_trajectory_rejected" in events

    different = _bare_client()
    different._pending_typed_replan = _pending_replan(
        old_signature, operation="operation:different"
    )
    different._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=True),
    )
    accepted = different._resolve_nav2(
        _command(sequence=18, source=1, action=3)
    )
    assert accepted.nav2_goal_sent is True
    assert different._pending_typed_replan is None

    refreshed = _bare_client()
    refreshed._pending_typed_replan = _pending_replan(
        old_signature, operation="operation:new-observation"
    )
    refreshed._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    accepted = refreshed._resolve_nav2(
        _command(
            sequence=18,
            source=1,
            action=2,
            observation_digest="observation:18",
        )
    )
    assert accepted.nav2_goal_sent is True
    assert refreshed._pending_typed_replan is None


def test_raw_wire_client_warns_and_releases_same_observation_system2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_signature = _primitive()
    client = _bare_client()
    client.system2_replan_policy = "raw_wire_warn"
    client._pending_typed_replan = _pending_replan(
        old_signature, operation="operation:raw-wire"
    )
    events: list[str] = []
    client._record_replan = (  # type: ignore[method-assign]
        lambda event, *_args, **_extra: events.append(event)
    )
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 100)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=True),
    )

    response = client._resolve_nav2(_command(sequence=18, source=1, action=2))

    assert response.nav2_goal_sent is True
    assert client._pending_typed_replan is None
    assert "raw_wire_semantic_bypass_warn" in events
    assert "consumed_by_fresh_model_step" in events


def test_strict_client_requires_generated_path_and_primitive_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    client.system2_replan_policy = "strict"
    client._pending_typed_replan = _pending_replan(
        _primitive(), operation="operation:strict"
    )
    client._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    client._safe_stop = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 100)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=True),
    )

    with pytest.raises(ClientFailure, match="excluded recovery trajectory"):
        client._resolve_nav2(
            _command(
                sequence=18,
                source=1,
                action=2,
                observation_digest="observation:18",
            )
        )


def test_raw_wire_client_still_rejects_stale_odometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    client.system2_replan_policy = "raw_wire_warn"
    client._pending_typed_replan = _pending_replan(
        _primitive(), operation="operation:raw-wire-stale"
    )
    client._t4_odometry_stamp_ns = 1
    client._reset_sim_barrier_ns = 0
    client.navigation_odometry_timeout = 1e-7
    client._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    client._safe_stop = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 200)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=True),
    )

    with pytest.raises(ClientFailure, match="excluded recovery trajectory"):
        client._resolve_nav2(_command(sequence=18, source=1, action=2))


def test_raw_wire_client_still_rejects_nonfinite_odometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    client.system2_replan_policy = "raw_wire_warn"
    client._pending_typed_replan = _pending_replan(
        _primitive(), operation="operation:raw-wire-nonfinite"
    )
    client._t4_odometry.pose.pose.position.x = float("nan")
    client._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    client._safe_stop = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 200)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=True),
    )

    with pytest.raises(ClientFailure, match="excluded recovery trajectory"):
        client._resolve_nav2(_command(sequence=18, source=1, action=2))


@pytest.mark.parametrize(
    ("stamp_ns", "future_tolerance_ns"),
    [(1, 100), (500, 100)],
)
def test_client_will_not_sign_stale_or_too_future_system2_odometry(
    monkeypatch: pytest.MonkeyPatch,
    stamp_ns: int,
    future_tolerance_ns: int,
) -> None:
    client = _bare_client()
    old_signature = _primitive()
    client._pending_typed_replan = _pending_replan(
        old_signature, operation=f"operation:stamp:{stamp_ns}"
    )
    client._t4_odometry_stamp_ns = stamp_ns
    client._reset_sim_barrier_ns = 0
    client._sensor_future_tolerance_ns = future_tolerance_ns
    client.navigation_odometry_timeout = 1e-7
    client._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    client._safe_stop = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 200)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(goal=True),
    )

    with pytest.raises(ClientFailure, match="excluded recovery trajectory"):
        client._resolve_nav2(_command(sequence=18, source=1, action=3))


def _bare_recovery_supervisor() -> RecoverySupervisor:
    supervisor = RecoverySupervisor.__new__(RecoverySupervisor)
    supervisor._recovery_uses_sim_time = True
    supervisor._lock = threading.RLock()
    supervisor.motion_enabled = False
    supervisor.latest_system2_turn_action = None
    supervisor._system2_turn_progress_armed = False
    supervisor._system2_turn_motion_started_ns = 0
    supervisor._motion_true_edge_serial = 0
    supervisor._consumed_motion_true_edge_serial = 0
    supervisor._motion_true_edge_semantic_ns = 0
    supervisor.episode_id = "a::259"
    supervisor.generation = 1
    supervisor.last_sequence_id = 17
    supervisor.latest_command_semantic_ns = 0
    supervisor.latest_command_signature = None
    supervisor.latest_signature_source = ""
    supervisor.samples = [(90, 1.25, -0.5, 0.0)]
    supervisor._semantic_now_ns = lambda: 100  # type: ignore[method-assign]
    return supervisor


def test_t5_no_progress_recovery_waits_for_the_bounded_action_window() -> None:
    assert not _no_progress_recovery_eligible(
        t5_completion_sim=True, command_age_sec=0.25
    )
    assert not _no_progress_recovery_eligible(
        t5_completion_sim=True, command_age_sec=3.0
    )
    assert _no_progress_recovery_eligible(
        t5_completion_sim=True, command_age_sec=3.01
    )
    assert _no_progress_recovery_eligible(
        t5_completion_sim=False, command_age_sec=0.0
    )


def test_turn_progress_arms_for_both_command_and_motion_edge_orders() -> None:
    command_first = _bare_recovery_supervisor()
    command_first._on_command(_command(sequence=18, source=1, action=2))
    assert command_first._system2_turn_progress_armed is False
    command_first._on_motion(SimpleNamespace(data=True))
    assert command_first._system2_turn_progress_armed is True
    assert command_first._system2_turn_motion_started_ns == 100
    command_first._semantic_now_ns = lambda: 200  # type: ignore[method-assign]
    command_first._on_motion(SimpleNamespace(data=True))
    assert command_first._system2_turn_motion_started_ns == 100
    assert command_first._motion_true_edge_serial == 1

    motion_first = _bare_recovery_supervisor()
    motion_first._on_motion(SimpleNamespace(data=True))
    assert motion_first._system2_turn_progress_armed is False
    assert motion_first._motion_true_edge_serial == 1
    motion_first._on_command(_command(sequence=18, source=1, action=2))
    assert motion_first._system2_turn_progress_armed is True
    assert motion_first._system2_turn_motion_started_ns == 100


def test_false_and_nonturn_consume_stale_motion_true_edges() -> None:
    after_false = _bare_recovery_supervisor()
    after_false._on_motion(SimpleNamespace(data=True))
    after_false._on_motion(SimpleNamespace(data=False))
    after_false._on_command(_command(sequence=18, source=1, action=2))
    assert after_false._system2_turn_progress_armed is False

    after_nonturn = _bare_recovery_supervisor()
    after_nonturn._on_motion(SimpleNamespace(data=True))
    after_nonturn._on_command(_command(sequence=18, source=1, action=1))
    after_nonturn._on_command(_command(sequence=19, source=1, action=2))
    assert after_nonturn._system2_turn_progress_armed is False

    delayed_false = _bare_recovery_supervisor()
    delayed_false.motion_enabled = True
    delayed_false._motion_true_edge_serial = 1
    delayed_false._consumed_motion_true_edge_serial = 1
    delayed_false._on_command(_command(sequence=18, source=1, action=2))
    delayed_false._on_motion(SimpleNamespace(data=False))
    assert delayed_false.latest_system2_turn_action == 2
    delayed_false._semantic_now_ns = lambda: 200  # type: ignore[method-assign]
    delayed_false._on_motion(SimpleNamespace(data=True))
    assert delayed_false._system2_turn_progress_armed is True
    assert delayed_false._system2_turn_motion_started_ns == 200


def test_adapter_excludes_same_system2_primitive_but_accepts_different_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    old_signature = _primitive(action=2, sequence=17)
    adapter.allow_system2_recovery_replan = True
    adapter.recovery_latched = True
    adapter.recovery_latched_id = "recovery:test"
    adapter.recovery_latched_goal_id = "goal:test"
    adapter.recovery_replan_armed = True
    adapter.recovery_replan_deadline_ns = 1_000
    adapter.recovery_excluded_absolute = {old_signature.absolute_sha256}
    adapter.recovery_excluded_shape = {old_signature.shape_sha256}
    adapter.recovery_excluded_hold_count = 0
    adapter.odom_lock = threading.Lock()
    adapter.latest_odom = (1.25, -0.5, 0.0)
    adapter._recovery_contract_now_ns = lambda: 100  # type: ignore[method-assign]
    adapter._system2_path = (  # type: ignore[method-assign]
        lambda _command, action: SimpleNamespace(
            poses=[
                _pose(0.0, 0.0),
                _pose(0.0, 0.25 if action == 2 else -0.25),
            ]
        )
    )
    committed: list[int] = []
    adapter._check_identity = (  # type: ignore[method-assign]
        lambda command: committed.append(int(command.sequence_id))
    )
    adapter._apply_ablation = lambda _request: None  # type: ignore[method-assign]
    adapter._append = lambda _record: None  # type: ignore[method-assign]
    adapter._publish_motion = lambda _enabled: None  # type: ignore[method-assign]

    def resolve_fresh(
        self: object, request: object, response: object
    ) -> object:
        adapter._check_identity(request.command)
        resolved = _resolution(goal=True)
        for name, value in vars(resolved).items():
            setattr(response, name, value)
        return response

    monkeypatch.setattr(ActiveNav2Adapter, "_resolve", resolve_fresh)

    same = adapter._resolve_recovery_latched(
        SimpleNamespace(command=_command(sequence=18, source=1, action=2)),
        SimpleNamespace(),
    )
    assert same.nav2_goal_sent is False
    assert adapter.recovery_latched is True

    different = adapter._resolve_recovery_latched(
        SimpleNamespace(command=_command(sequence=19, source=1, action=3)),
        SimpleNamespace(),
    )
    assert different.nav2_goal_sent is True
    assert adapter.recovery_latched is False
    assert committed == [18, 19]


def test_adapter_accepts_fresh_in_place_turn_without_path_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-translation T5 turn is identified by its primitive signature."""

    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    old_signature = _primitive(action=3, sequence=17)
    adapter.allow_system2_recovery_replan = True
    adapter.recovery_latched = True
    adapter.recovery_latched_id = "recovery:in-place"
    adapter.recovery_latched_goal_id = "goal:in-place"
    adapter.recovery_replan_armed = True
    adapter.recovery_replan_deadline_ns = 1_000
    adapter.recovery_excluded_absolute = {old_signature.absolute_sha256}
    adapter.recovery_excluded_shape = {old_signature.shape_sha256}
    adapter.recovery_excluded_hold_count = 0
    adapter.odom_lock = threading.Lock()
    adapter.latest_odom = (1.25, -0.5, 0.0)
    adapter._recovery_contract_now_ns = lambda: 100  # type: ignore[method-assign]
    adapter._system2_path = (  # type: ignore[method-assign]
        lambda _command, _action: SimpleNamespace(
            poses=[_pose(1.25, -0.5) for _ in range(9)]
        )
    )
    committed: list[int] = []
    adapter._check_identity = (  # type: ignore[method-assign]
        lambda command: committed.append(int(command.sequence_id))
    )
    adapter._apply_ablation = lambda _request: None  # type: ignore[method-assign]
    events: list[dict[str, object]] = []
    adapter._append = lambda record: events.append(record)  # type: ignore[method-assign]
    adapter._publish_motion = lambda _enabled: None  # type: ignore[method-assign]

    def resolve_fresh(
        self: object, request: object, response: object
    ) -> object:
        adapter._check_identity(request.command)
        resolved = _resolution(goal=True)
        for name, value in vars(resolved).items():
            setattr(response, name, value)
        return response

    monkeypatch.setattr(ActiveNav2Adapter, "_resolve", resolve_fresh)

    response = adapter._resolve_recovery_latched(
        SimpleNamespace(command=_command(sequence=18, source=1, action=2)),
        SimpleNamespace(),
    )

    assert response.nav2_goal_sent is True
    assert adapter.recovery_latched is False
    assert committed == [18]
    accepted = [
        event
        for event in events
        if event["event"] == "recovery_latch_fresh_trajectory_accepted"
    ]
    assert accepted[0]["trajectory_source"] == "system2_primitive"


def test_raw_wire_adapter_warns_and_releases_excluded_system2_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    old_signature = _primitive(action=2, sequence=17)
    adapter.system2_replan_policy = "raw_wire_warn"
    adapter.allow_system2_recovery_replan = True
    adapter.recovery_latched = True
    adapter.recovery_latched_id = "recovery:raw-wire"
    adapter.recovery_latched_goal_id = "goal:raw-wire"
    adapter.recovery_replan_armed = True
    adapter.recovery_replan_deadline_ns = 1_000
    adapter.recovery_excluded_absolute = {old_signature.absolute_sha256}
    adapter.recovery_excluded_shape = {old_signature.shape_sha256}
    adapter.recovery_excluded_hold_count = 0
    adapter.odom_lock = threading.Lock()
    adapter.latest_odom = (1.25, -0.5, 0.0)
    adapter._recovery_contract_now_ns = lambda: 100  # type: ignore[method-assign]
    adapter._system2_path = (  # type: ignore[method-assign]
        lambda _command, _action: SimpleNamespace(
            poses=[_pose(1.25, -0.5) for _ in range(9)]
        )
    )
    committed: list[int] = []
    adapter._check_identity = (  # type: ignore[method-assign]
        lambda command: committed.append(int(command.sequence_id))
    )
    adapter._apply_ablation = lambda _request: None  # type: ignore[method-assign]
    events: list[dict[str, object]] = []
    adapter._append = lambda record: events.append(record)  # type: ignore[method-assign]
    adapter._publish_motion = lambda _enabled: None  # type: ignore[method-assign]

    def resolve_fresh(
        self: object, request: object, response: object
    ) -> object:
        adapter._check_identity(request.command)
        resolved = _resolution(goal=True)
        for name, value in vars(resolved).items():
            setattr(response, name, value)
        return response

    monkeypatch.setattr(ActiveNav2Adapter, "_resolve", resolve_fresh)

    response = adapter._resolve_recovery_latched(
        SimpleNamespace(command=_command(sequence=18, source=1, action=2)),
        SimpleNamespace(),
    )

    assert response.nav2_goal_sent is True
    assert adapter.recovery_latched is False
    assert committed == [18]
    assert any(
        event["event"] == "raw_wire_system2_signature_bypass_warn"
        for event in events
    )


def test_missing_system1_queue_target_drains_as_nonterminal_same_reset_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    cleared: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    client._clear_safe_stop = lambda message: cleared.append(message)  # type: ignore[method-assign]
    client._record_replan = (  # type: ignore[method-assign]
        lambda event, _index, _sequence, **extra: events.append((event, extra))
    )
    client._stand_resolution = (  # type: ignore[method-assign]
        lambda _command, message: _resolution(message)
    )

    def missing(_self: object, _command: object) -> object:
        raise ClientFailure(2, T4OdometryClientNode.SYSTEM1_QUEUE_TARGET_MISSING)

    monkeypatch.setattr(InternVLAClientNode, "_resolve_nav2", missing)
    first = client._resolve_nav2_with_terminal_fallback(_command())
    second = client._resolve_nav2_with_terminal_fallback(_command(sequence=18))

    assert first.status_code == second.status_code == 0
    assert "awaiting fresh same-reset trajectory" in first.status_message
    assert client._system1_queue_hold_identity == ("a::259", 1)
    assert client._system1_queue_hold_count == 2
    assert len(cleared) == 2
    assert [event for event, _ in events] == [
        "system1_queue_target_missing_safe_hold",
        "system1_queue_target_missing_safe_hold",
    ]


def test_missing_queue_hold_does_not_cross_reset_and_fresh_result_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    client._clear_safe_stop = lambda _message: None  # type: ignore[method-assign]
    client._record_replan = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    client._stand_resolution = (  # type: ignore[method-assign]
        lambda _command, message: _resolution(message)
    )

    def missing(_self: object, _command: object) -> object:
        raise ClientFailure(2, T4OdometryClientNode.SYSTEM1_QUEUE_TARGET_MISSING)

    monkeypatch.setattr(InternVLAClientNode, "_resolve_nav2", missing)
    client._resolve_nav2_with_terminal_fallback(_command())
    client._resolve_nav2_with_terminal_fallback(
        _command(episode="a::676", generation=2, sequence=1)
    )
    assert client._system1_queue_hold_identity == ("a::676", 2)
    assert client._system1_queue_hold_count == 1

    monkeypatch.setattr(
        InternVLAClientNode, "_resolve_nav2", lambda _self, _command: _resolution()
    )
    client._resolve_nav2_with_terminal_fallback(
        _command(episode="a::676", generation=2, sequence=2, source=2)
    )
    assert client._system1_queue_hold_identity is None
    assert client._system1_queue_hold_count == 0


def test_missing_queue_safe_hold_is_bounded_at_registered_trajectory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    client._clear_safe_stop = lambda _message: None  # type: ignore[method-assign]
    events: list[str] = []
    client._record_replan = (  # type: ignore[method-assign]
        lambda event, *_args, **_kwargs: events.append(event)
    )
    client._stand_resolution = (  # type: ignore[method-assign]
        lambda _command, message: _resolution(message)
    )

    def missing(_self: object, _command: object) -> object:
        raise ClientFailure(2, T4OdometryClientNode.SYSTEM1_QUEUE_TARGET_MISSING)

    monkeypatch.setattr(InternVLAClientNode, "_resolve_nav2", missing)
    for sequence in range(T4OdometryClientNode.SYSTEM1_QUEUE_SAFE_HOLD_LIMIT):
        response = client._resolve_nav2_with_terminal_fallback(
            _command(sequence=sequence)
        )
        assert response.status_code == 0

    with pytest.raises(ClientFailure, match="identity-bound absolute target"):
        client._resolve_nav2_with_terminal_fallback(
            _command(sequence=T4OdometryClientNode.SYSTEM1_QUEUE_SAFE_HOLD_LIMIT)
        )
    assert events[-1] == "system1_queue_target_missing_hold_limit_exceeded"


def _pose(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))


def test_t5_oracle_termination_compares_global_odom_to_world_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    adapter.factors = {"termination_mode": "oracle_termination"}
    adapter.ablation_episodes = [
        {
            "episode_id": "259",
            "start_position": [6.4, 3.6, -9.2],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "reference_path": [
                [6.4, 3.6, -9.2],
                [5.14005, 3.6, -4.26937],
            ],
        }
    ]
    adapter.odom_lock = threading.Lock()
    adapter.latest_odom = (5.8637, 6.2630, 0.0)
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")

    reference = adapter._episode_reference("a::259", 0)
    _, goal_distance = adapter._oracle_local_path("a::259", 0)

    assert reference == [(6.4, 9.2), (5.14005, 4.26937)]
    assert goal_distance == pytest.approx(2.1209, abs=1e-3)
    assert goal_distance <= 2.5


def test_t5_oracle_episode_lookup_follows_runtime_order_and_binds_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    adapter.factors = {"termination_mode": "oracle_termination"}
    adapter.ablation_episodes = [
        {
            "episode_id": "121",
            "reference_path": [[4.0, 0.0, 1.0]],
        },
        {
            "episode_id": "628",
            "reference_path": [[9.0, 0.0, 2.0]],
        },
    ]
    adapter._t5_ablation_generation_episode_ids = {}
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")

    assert adapter._episode_reference("a::628", 0) == [(9.0, -2.0)]
    assert adapter._t5_ablation_generation_episode_ids == {0: "a::628"}
    with pytest.raises(TimeoutError, match="already bound"):
        adapter._episode_reference("a::121", 0)


def test_oracle_termination_does_not_cache_stand_as_last_motion() -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    adapter.factors = {
        "system_mode": "full_system1_system2",
        "trajectory_mode": "full_trajectory",
        "termination_mode": "oracle_termination",
    }
    adapter.latest_system2_action = None
    adapter.latest_non_stop_system2_action = None
    adapter.activation_counts = {"system": 0, "trajectory": 0, "termination": 0}
    adapter.ablation_variant_id = "none"
    adapter._oracle_local_path = (  # type: ignore[method-assign]
        lambda _episode, _generation: ([(0.0, 0.0), (1.0, 0.0)], 4.0)
    )

    turn = _command(source=1, action=2)
    adapter._apply_ablation(SimpleNamespace(command=turn))
    assert adapter.latest_non_stop_system2_action == 2

    hold = _command(sequence=18, source=1, action=-1)
    adapter._apply_ablation(SimpleNamespace(command=hold))
    assert adapter.latest_non_stop_system2_action == 2

    false_stop = _command(sequence=19, source=1, action=0)
    adapter._apply_ablation(SimpleNamespace(command=false_stop))

    assert false_stop.stop is False
    assert false_stop.discrete_action == 2


def test_frozen_t4_episode_lookup_still_uses_generation_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    adapter.factors = {"termination_mode": "oracle_termination"}
    adapter.ablation_episodes = [
        {
            "episode_id": "121",
            "start_position": [4.0, 0.0, 1.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "reference_path": [[4.0, 0.0, 1.0]],
        },
        {
            "episode_id": "628",
            "start_position": [9.0, 0.0, 2.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "reference_path": [[9.0, 0.0, 2.0]],
        },
    ]
    monkeypatch.delenv("INTERNNAV_T5_LANE", raising=False)

    assert adapter._episode_reference("628", 1) == [(0.0, 0.0)]
    with pytest.raises(TimeoutError, match="does not match"):
        adapter._episode_reference("121", 1)


def test_excluded_system2_recovery_path_has_bounded_safe_fallback() -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    generated = SimpleNamespace(poses=[_pose(0.0, 0.0), _pose(0.25, 0.0)])
    signature = trajectory_signature([(0.0, 0.0), (0.25, 0.0)])
    adapter.allow_system2_recovery_replan = True
    adapter.recovery_latched = True
    adapter.recovery_latched_id = "recovery:test"
    adapter.recovery_latched_goal_id = "goal:test"
    adapter.recovery_replan_armed = True
    adapter.recovery_replan_deadline_ns = 200
    adapter.recovery_excluded_absolute = {signature.absolute_sha256}
    adapter.recovery_excluded_shape = {signature.shape_sha256}
    adapter.recovery_excluded_hold_count = 0
    adapter.odom_lock = threading.Lock()
    adapter.latest_odom = (1.25, -0.5, 0.0)
    adapter._system2_path = lambda _command, _action: generated  # type: ignore[method-assign]
    adapter._recovery_contract_now_ns = lambda: 100  # type: ignore[method-assign]
    committed: list[int] = []
    adapter._check_identity = (  # type: ignore[method-assign]
        lambda command: committed.append(int(command.sequence_id))
    )
    events: list[dict[str, object]] = []
    adapter._append = lambda record: events.append(record)  # type: ignore[method-assign]
    motion: list[bool] = []
    adapter._publish_motion = lambda enabled: motion.append(enabled)  # type: ignore[method-assign]

    responses = []
    for sequence in (9, 10, 11):
        response = SimpleNamespace()
        responses.append(
            adapter._resolve_recovery_latched(
                SimpleNamespace(command=_command(sequence=sequence, source=1)), response
            )
        )

    assert committed == [9, 10, 11]
    assert all(response.discrete_action == -1 and not response.stop for response in responses)
    assert responses[-1].status_message == T4OdometryClientNode.RECOVERY_BOUNDED_FALLBACK
    assert adapter.recovery_latched is False
    assert adapter.recovery_excluded_absolute == set()
    assert adapter.recovery_excluded_shape == set()
    assert motion == [False]
    assert sum(event["event"] == "recovery_latch_bounded_fallback" for event in events) == 1


def test_adapter_consumes_expired_replan_identity_without_sending_goal() -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    adapter.recovery_latched = True
    adapter.recovery_latched_id = "recovery:expired"
    adapter.recovery_latched_goal_id = "goal:expired"
    adapter.recovery_replan_armed = True
    adapter.recovery_replan_deadline_ns = 50
    adapter.recovery_excluded_absolute = {"ab" * 32}
    adapter.recovery_excluded_shape = {"cd" * 32}
    adapter.recovery_excluded_hold_count = 0
    adapter._recovery_contract_now_ns = lambda: 51  # type: ignore[method-assign]
    committed: list[int] = []
    adapter._check_identity = (  # type: ignore[method-assign]
        lambda command: committed.append(int(command.sequence_id))
    )
    events: list[dict[str, object]] = []
    adapter._append = lambda record: events.append(record)  # type: ignore[method-assign]
    motion: list[bool] = []
    adapter._publish_motion = lambda enabled: motion.append(enabled)  # type: ignore[method-assign]

    response = adapter._resolve_recovery_latched(
        SimpleNamespace(command=_command(episode="a::676", generation=2, sequence=133)),
        SimpleNamespace(),
    )

    assert committed == [133]
    assert response.status_code == 0
    assert response.stop is False
    assert response.nav2_goal_sent is False
    assert response.nav2_plan_valid is False
    assert adapter.recovery_latched is False
    assert motion == [False]
    assert [event["event"] for event in events] == [
        "recovery_latch_deadline_bounded_fallback"
    ]


def test_client_and_adapter_replan_consumers_share_one_absolute_deadline() -> None:
    source = (
        ROOT
        / "internvla_t4_recovery"
        / "internvla_t4_recovery"
        / "recovery_node.py"
    ).read_text(encoding="utf-8")
    recovery = source.split("    def _recover(", 1)[1].split("\n\ndef main(", 1)[0]
    client_request = recovery.index("client_replan_request = self._recovery_request(")
    deadline_copy = recovery.index("shared_replan_deadline_ns = (", client_request)
    adapter_request = recovery.index("adapter_replan_request = self._recovery_request(")
    override = recovery.index(
        "deadline_ns_override=shared_replan_deadline_ns", adapter_request
    )

    assert client_request < deadline_copy < adapter_request < override
    adapter_source = (
        ROOT
        / "internvla_t4_recovery"
        / "internvla_t4_recovery"
        / "adapter_node.py"
    ).read_text(encoding="utf-8")
    assert "self.recovery_replan_deadline_ns = identity.deadline_ns" in adapter_source


def test_expired_arm_request_releases_unarmed_latch_fail_closed() -> None:
    adapter = T4RecoveryAdapter.__new__(T4RecoveryAdapter)
    adapter.operation_lock = threading.RLock()
    adapter.active_episode = "a::676"
    adapter.active_generation = 2
    adapter.last_sequence = 132
    adapter.recovery_latched = True
    adapter.recovery_latched_id = "recovery:expired-arm"
    adapter.recovery_latched_goal_id = goal_identity("a::676", 2, 132)
    adapter.recovery_replan_armed = False
    adapter.recovery_replan_deadline_ns = 0
    adapter.recovery_excluded_hold_count = 0
    adapter.recovery_excluded_absolute = set()
    adapter.recovery_excluded_shape = set()
    adapter._typed_recovery_results = {}
    adapter._recovery_contract_now_ns = lambda: 51  # type: ignore[method-assign]
    events: list[dict[str, object]] = []
    adapter._append = lambda record: events.append(record)  # type: ignore[method-assign]
    motion: list[bool] = []
    adapter._publish_motion = lambda enabled: motion.append(enabled)  # type: ignore[method-assign]

    request = RecoveryControl.Request()
    request.episode_id = "a::676"
    request.reset_generation = 2
    request.goal_id = adapter.recovery_latched_goal_id
    request.recovery_id = adapter.recovery_latched_id
    request.operation = OP_REQUEST_REPLAN
    request.operation_id = operation_identity(request.recovery_id, request.operation)
    request.deadline.sec = 0
    request.deadline.nanosec = 50
    request.excluded_absolute_sha256 = ["ab" * 32]
    request.excluded_shape_sha256 = ["cd" * 32]

    response = adapter._on_typed_replan_arm(request, RecoveryControl.Response())

    assert response.success is False
    assert response.status_code != 0
    assert response.status_message == "recovery operation deadline expired"
    assert adapter.recovery_latched is False
    assert motion == [False]
    assert [event["event"] for event in events] == [
        "recovery_latch_expired_before_arm_released"
    ]


def test_expired_sim_time_replan_returns_bounded_hold_not_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client()
    client._pending_typed_replan = {
        "request_index": 4,
        "episode_id": "a::676",
        "reset_generation": 2,
        "goal_id": "goal:test",
        "recovery_id": "recovery:test",
        "operation_id": "operation:test",
        "deadline_ns": 50,
        "active_sequence_id": 132,
        "excluded_absolute": set(),
        "excluded_shape": set(),
        "rejection_count": 0,
    }
    client.episode_id = "a::676"
    client.reset_generation = 2
    events: list[str] = []
    client._record_replan = (  # type: ignore[method-assign]
        lambda event, *_args, **_kwargs: events.append(event)
    )
    monkeypatch.setattr(t4_client_module, "_contract_now_ns", lambda _node: 51)
    monkeypatch.setattr(
        InternVLAClientNode,
        "_resolve_nav2",
        lambda _self, _command: _resolution(
            T4OdometryClientNode.RECOVERY_BOUNDED_FALLBACK
        ),
    )

    response = client._resolve_nav2(
        _command(episode="a::676", generation=2, sequence=133, source=1)
    )

    assert response.status_code == 0
    assert response.stop is False
    assert client._pending_typed_replan is None
    assert events == ["deadline_expired", "deadline_expired_bounded_fallback"]
