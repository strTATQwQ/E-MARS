from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("internvla_ros2", "internvla_t4_recovery"):
    sys.path.insert(0, str(ROOT / package))

from internvla_ros2 import client_node as client_module  # noqa: E402
from internvla_ros2.client_node import (  # noqa: E402
    ClientFailure,
    InternVLAClientNode,
    T5_STEP_ACTION_CLIENT_MAX_COMPLETED_GOALS,
    T5_STEP_GOAL_ACCEPTANCE_LIVENESS_SEC,
    T5_STEP_RESULT_PRIMARY_LIVENESS_SEC,
    T5_STEP_RESULT_REFETCH_LIVENESS_SEC,
)
from internvla_ros2.model_node import InternVLAModelNode  # noqa: E402
from internvla_ros2.protocol import (  # noqa: E402
    RequestIdentity,
    STATUS_OBSERVATION_MISSING,
    STATUS_OK,
    STATUS_RESET_MISMATCH,
    STATUS_STALE,
    STATUS_TIMEOUT,
)
from internvla_t4_recovery.model_node import T4RecoveryModelNode  # noqa: E402


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


class _GoalHandle:
    def __init__(self, *, cancel_error: BaseException | None = None) -> None:
        self.accepted = True
        self.result_requests: list[object] = []
        self.cancel_count = 0
        self.cancel_error = cancel_error

    def get_result_async(self) -> object:
        token = object()
        self.result_requests.append(token)
        return token

    def cancel_goal_async(self) -> object:
        self.cancel_count += 1
        if self.cancel_error is not None:
            raise self.cancel_error
        return object()


class _StepClient:
    def __init__(self) -> None:
        self.sent_goals: list[object] = []
        self.destroyed = False

    def send_goal_async(self, goal: object) -> object:
        self.sent_goals.append(goal)
        return object()

    def destroy(self) -> None:
        self.destroyed = True

    def wait_for_server(self, timeout_sec: float) -> bool:
        del timeout_sec
        return True


def _bare_client(*, t5_sim_time_only: bool) -> InternVLAClientNode:
    client = InternVLAClientNode.__new__(InternVLAClientNode)
    client.t5_sim_time_only = t5_sim_time_only
    client.current_goal_lock = threading.Lock()
    client.current_goal = None
    client.step_result_refetch_count = 0
    client.observation_tuple_republish_count = 0
    client.step_action_client_completed_goals = 0
    client.step_action_client_rotation_count = 0
    client.step_action_client_transport_poisoned = False
    client._step_action_callback_group = object()
    client._test_logger = _Logger()
    client.get_logger = lambda: client._test_logger  # type: ignore[method-assign]
    client._test_safe_stops = []
    client._safe_stop = (  # type: ignore[method-assign]
        lambda status, message: client._test_safe_stops.append((status, message))
    )
    client.rgb_publisher = _Publisher()
    client.depth_publisher = _Publisher()
    client.metadata_publisher = _Publisher()
    return client


def test_completed_goal_bound_rotates_quiescent_action_client_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    previous = _StepClient()
    replacement = _StepClient()
    client.step_client = previous
    client.step_action_client_completed_goals = (
        T5_STEP_ACTION_CLIENT_MAX_COMPLETED_GOALS
    )
    handle = _GoalHandle()
    wrapped = _wrapped(STATUS_OK)

    action_client_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def action_client(*args: object, **kwargs: object) -> _StepClient:
        action_client_calls.append((args, kwargs))
        return replacement

    monkeypatch.setattr(client_module, "ActionClient", action_client)

    def future_result(_future: object, _timeout: float, description: str) -> object:
        return handle if description == "step goal acceptance" else wrapped

    monkeypatch.setattr(client_module, "_future_result", future_result)
    goal = _goal()

    result, refetches = client._send_step_goal_once(goal, 300.0, 300.0)

    assert result is wrapped
    assert refetches == 0
    assert previous.destroyed is True
    assert previous.sent_goals == []
    assert replacement.sent_goals == [goal]
    assert client.step_action_client_rotation_count == 1
    assert client.step_action_client_completed_goals == 1
    assert client.step_action_client_transport_poisoned is False
    assert action_client_calls[0][1]["callback_group"] is (
        client._step_action_callback_group
    )


def test_initial_and_rotated_step_clients_use_the_dedicated_callback_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    created = _StepClient()
    calls: list[dict[str, object]] = []

    def action_client(*_args: object, **kwargs: object) -> _StepClient:
        calls.append(kwargs)
        return created

    monkeypatch.setattr(client_module, "ActionClient", action_client)

    assert client._new_step_action_client() is created
    assert calls == [
        {"callback_group": client._step_action_callback_group}
    ]
    source = (
        ROOT / "internvla_ros2/internvla_ros2/client_node.py"
    ).read_text(encoding="utf-8")
    assert source.count("self.step_client = self._new_step_action_client()") == 2


def test_goal_acceptance_timeout_marks_transport_poisoned_without_goal_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    step_client = _StepClient()
    client.step_client = step_client

    def future_result(_future: object, _timeout: float, description: str) -> object:
        raise ClientFailure(STATUS_TIMEOUT, description)

    monkeypatch.setattr(client_module, "_future_result", future_result)
    goal = _goal()

    with pytest.raises(ClientFailure, match="step goal acceptance"):
        client._send_step_goal_once(goal, 300.0, 300.0)

    assert step_client.sent_goals == [goal]
    assert client.step_action_client_transport_poisoned is True
    assert client.step_action_client_completed_goals == 0
    assert client._test_safe_stops == [
        (
            STATUS_TIMEOUT,
            "Step goal acceptance transport timed out; client marked poisoned",
        )
    ]


def _goal() -> SimpleNamespace:
    return SimpleNamespace(
        episode_id="a::259",
        reset_generation=1,
        sequence_id=17,
        request_id="a::259:1:17",
        observation_digest="ab" * 32,
    )


def _wrapped(status_code: int, **identity_overrides: object) -> SimpleNamespace:
    identity: dict[str, object] = {
        "episode_id": "a::259",
        "reset_generation": 1,
        "sequence_id": 17,
        "request_id": "a::259:1:17",
    }
    identity.update(identity_overrides)
    return SimpleNamespace(
        result=SimpleNamespace(status_code=status_code, **identity)
    )


def test_result_continuation_reuses_same_future_without_second_get_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    client.step_client = _StepClient()
    handle = _GoalHandle()
    recovered = _wrapped(STATUS_OK)
    waits: list[tuple[object, float, str]] = []

    def future_result(future: object, timeout: float, description: str) -> object:
        waits.append((future, timeout, description))
        if description == "step goal acceptance":
            return handle
        if description == "step result":
            raise ClientFailure(STATUS_TIMEOUT, "primary result delivery stalled")
        assert description == "same-goal step result continuation"
        return recovered

    monkeypatch.setattr(client_module, "_future_result", future_result)
    goal = SimpleNamespace(sequence_id=17, request_id="a::259:1:17")

    result, refetch_count = client._send_step_goal_once(goal, 300.0, 300.0)

    assert result is recovered
    assert refetch_count == 1
    assert client.step_client.sent_goals == [goal]
    assert len(handle.result_requests) == 1
    assert handle.cancel_count == 0
    assert client._test_safe_stops == []
    assert waits[0][1:] == (
        T5_STEP_GOAL_ACCEPTANCE_LIVENESS_SEC,
        "step goal acceptance",
    )
    assert waits[1][1:] == (T5_STEP_RESULT_PRIMARY_LIVENESS_SEC, "step result")
    assert waits[2][0] is waits[1][0]
    assert waits[2][1:] == (
        T5_STEP_RESULT_REFETCH_LIVENESS_SEC,
        "same-goal step result continuation",
    )


def test_second_same_goal_result_failure_cancels_and_safe_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    client.step_client = _StepClient()
    handle = _GoalHandle()

    def future_result(_future: object, _timeout: float, description: str) -> object:
        if description == "step goal acceptance":
            return handle
        raise ClientFailure(STATUS_TIMEOUT, description)

    monkeypatch.setattr(client_module, "_future_result", future_result)

    with pytest.raises(ClientFailure, match="same-goal step result continuation"):
        client._send_step_goal_once(object(), 300.0, 300.0)

    assert len(client.step_client.sent_goals) == 1
    assert len(handle.result_requests) == 1
    assert handle.cancel_count == 1
    assert client.step_result_refetch_count == 1
    assert client._test_safe_stops == [
        (
            STATUS_TIMEOUT,
            "step result timeout after bounded same-goal re-fetch; cancel requested",
        )
    ]


def test_non_timeout_result_failure_is_not_refetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    client.step_client = _StepClient()
    handle = _GoalHandle()

    def future_result(_future: object, _timeout: float, description: str) -> object:
        if description == "step goal acceptance":
            return handle
        raise ClientFailure(STATUS_OBSERVATION_MISSING, "result transport failed")

    monkeypatch.setattr(client_module, "_future_result", future_result)

    with pytest.raises(ClientFailure) as raised:
        client._send_step_goal_once(_goal(), 300.0, 300.0)

    assert raised.value.status_code == STATUS_OBSERVATION_MISSING
    assert len(handle.result_requests) == 1
    assert handle.cancel_count == 1
    assert client.step_result_refetch_count == 0


def test_cancel_failure_does_not_mask_result_failure_or_prevent_safe_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    client.step_client = _StepClient()
    handle = _GoalHandle(cancel_error=RuntimeError("cancel transport failed"))

    def future_result(_future: object, _timeout: float, description: str) -> object:
        if description == "step goal acceptance":
            return handle
        raise ClientFailure(STATUS_TIMEOUT, description)

    monkeypatch.setattr(client_module, "_future_result", future_result)

    with pytest.raises(ClientFailure, match="same-goal step result continuation"):
        client._send_step_goal_once(_goal(), 300.0, 300.0)

    assert handle.cancel_count == 1
    assert client.step_result_refetch_count == 1
    assert client._test_safe_stops == [
        (
            STATUS_TIMEOUT,
            "step result timeout after bounded same-goal re-fetch; cancel requested",
        )
    ]
    assert any(
        "best-effort Step cancel failed after safe-stop" in warning
        for warning in client._test_logger.warnings
    )


def test_status7_retry_republishes_exact_tuple_and_reuses_protocol_identity_once() -> None:
    client = _bare_client(t5_sim_time_only=True)
    goal = _goal()
    rgb = SimpleNamespace(stamp_ns=123, payload=b"rgb")
    depth = SimpleNamespace(stamp_ns=123, payload=b"depth")
    metadata = SimpleNamespace(
        stamp_ns=123,
        sequence_id=17,
        request_id="a::259:1:17",
        observation_digest="ab" * 32,
    )
    sent: list[object] = []
    responses = iter((_wrapped(STATUS_OBSERVATION_MISSING), _wrapped(STATUS_OK)))

    def send_once(candidate: object, *_args: object) -> tuple[object, int]:
        sent.append(candidate)
        return next(responses), 0

    client._send_step_goal_once = send_once  # type: ignore[method-assign]

    wrapped, result_refetches, republishes = (
        client._request_step_with_bounded_observation_retry(
            goal=goal,
            rgb_message=rgb,
            depth_message=depth,
            metadata=metadata,
            action_liveness_timeout_sec=300.0,
            result_liveness_timeout_sec=300.0,
        )
    )

    assert wrapped.result.status_code == STATUS_OK
    assert result_refetches == 0
    assert republishes == 1
    assert sent == [goal, goal]
    assert goal.sequence_id == metadata.sequence_id == 17
    assert goal.request_id == metadata.request_id
    assert goal.observation_digest == metadata.observation_digest
    assert client.rgb_publisher.messages == [rgb]
    assert client.depth_publisher.messages == [depth]
    assert client.metadata_publisher.messages == [metadata]


@pytest.mark.parametrize(
    ("identity_field", "mismatched_value"),
    [
        ("episode_id", "a::different"),
        ("reset_generation", 2),
        ("sequence_id", 18),
        ("request_id", "a::259:1:other"),
    ],
)
def test_mismatched_status7_never_republishes_or_resends(
    identity_field: str,
    mismatched_value: object,
) -> None:
    client = _bare_client(t5_sim_time_only=True)
    goal = _goal()
    sent: list[object] = []
    response = _wrapped(
        STATUS_OBSERVATION_MISSING,
        **{identity_field: mismatched_value},
    )

    def send_once(candidate: object, *_args: object) -> tuple[object, int]:
        sent.append(candidate)
        return response, 0

    client._send_step_goal_once = send_once  # type: ignore[method-assign]

    wrapped, _, republishes = client._request_step_with_bounded_observation_retry(
        goal=goal,
        rgb_message=object(),
        depth_message=object(),
        metadata=object(),
        action_liveness_timeout_sec=300.0,
        result_liveness_timeout_sec=300.0,
    )

    assert wrapped is response
    assert sent == [goal]
    assert republishes == 0
    assert client.rgb_publisher.messages == []


@pytest.mark.parametrize(
    ("exact_t5", "status_code", "expected_sends", "expected_republishes"),
    [
        (True, STATUS_OBSERVATION_MISSING, 2, 1),
        (False, STATUS_OBSERVATION_MISSING, 1, 0),
        (True, STATUS_STALE, 1, 0),
        (True, STATUS_RESET_MISMATCH, 1, 0),
    ],
)
def test_observation_retry_is_exact_status7_only_and_bounded(
    exact_t5: bool,
    status_code: int,
    expected_sends: int,
    expected_republishes: int,
) -> None:
    client = _bare_client(t5_sim_time_only=exact_t5)
    sent: list[object] = []
    response = _wrapped(status_code)

    def send_once(goal: object, *_args: object) -> tuple[object, int]:
        sent.append(goal)
        return response, 0

    client._send_step_goal_once = send_once  # type: ignore[method-assign]
    wrapped, _, republishes = client._request_step_with_bounded_observation_retry(
        goal=_goal(),
        rgb_message=object(),
        depth_message=object(),
        metadata=object(),
        action_liveness_timeout_sec=300.0,
        result_liveness_timeout_sec=300.0,
    )

    assert wrapped.result.status_code == status_code
    assert len(sent) == expected_sends
    assert republishes == expected_republishes
    assert len(client.rgb_publisher.messages) == expected_republishes


def test_non_t5_result_wait_has_no_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _bare_client(t5_sim_time_only=False)
    handle = _GoalHandle()
    recovered = _wrapped(STATUS_OK)
    waits: list[tuple[float, str]] = []

    def future_result(_future: object, timeout: float, description: str) -> object:
        waits.append((timeout, description))
        return recovered

    monkeypatch.setattr(client_module, "_future_result", future_result)

    wrapped, refetch_count = client._wait_for_accepted_step_result(handle, 7.5)

    assert wrapped is recovered
    assert refetch_count == 0
    assert len(handle.result_requests) == 1
    assert waits == [(7.5, "step result")]


def test_action_server_retains_results_beyond_both_liveness_windows() -> None:
    source = (
        ROOT / "internvla_ros2/internvla_ros2/model_node.py"
    ).read_text(encoding="utf-8")
    assert "result_timeout=900.0" in source
    assert 900.0 > (
        T5_STEP_RESULT_PRIMARY_LIVENESS_SEC
        + T5_STEP_RESULT_REFETCH_LIVENESS_SEC
    )


class _Agent:
    def __init__(self) -> None:
        self.reset_calls: list[list[int]] = []
        self.s2_output = SimpleNamespace(
            output_action=[3, 3],
            output_latent=object(),
            output_pixel=object(),
        )
        self.s2_output_lock = threading.Lock()
        self.policy_history = ["kitchen", "left-exit"]

    def reset(self, value: list[int]) -> None:
        self.reset_calls.append(value)


def _bare_recovery_model() -> T4RecoveryModelNode:
    node = T4RecoveryModelNode.__new__(T4RecoveryModelNode)
    node._agent = _Agent()
    node._system1_queue_remaining = 4
    node._results = {"prior": object()}
    node.recovery_clear_count = 0
    node.cache_epoch = 0
    node._barrier = SimpleNamespace(
        episode_id="a::259", reset_generation=1, last_sequence_id=12
    )
    node._rgb_cache = OrderedDict([(123, "rgb-sequence-13")])
    node._depth_cache = OrderedDict([(123, "depth-sequence-13")])
    node._metadata_cache = OrderedDict([(123, "metadata-sequence-13")])
    node._accepted_observation_tuples = OrderedDict()
    node._test_audit = []
    node._write_audit = node._test_audit.append  # type: ignore[method-assign]
    node._test_clear_observations = []
    node._clear_observations = (  # type: ignore[method-assign]
        lambda: node._test_clear_observations.append(True)
    )
    return node


def test_exact_t5_typed_history_clear_preserves_inflight_observation_tuple() -> None:
    node = _bare_recovery_model()

    epoch = node._clear_history_locked(
        "recovery:a::259:1:12",
        "operation:clear",
        preserve_observation_tuple=True,
    )

    assert epoch == 1
    assert node._agent.reset_calls == [[0]]
    assert node._system1_queue_remaining == 0
    assert node._results == {}
    assert node._test_clear_observations == []
    assert node._rgb_cache[123] == "rgb-sequence-13"
    assert node._depth_cache[123] == "depth-sequence-13"
    assert node._metadata_cache[123] == "metadata-sequence-13"
    assert node._test_audit[-1]["observation_tuple_preserved"] is True


def test_legacy_and_non_t5_history_clear_still_clear_observation_cache() -> None:
    node = _bare_recovery_model()

    node._clear_history_locked("legacy_trigger", "legacy")

    assert node._test_clear_observations == [True]
    assert node._test_audit[-1]["observation_tuple_preserved"] is False


def test_step3_refresh_invalidates_only_action_queue_and_preserves_history() -> None:
    node = _bare_recovery_model()
    latent = node._agent.s2_output.output_latent
    pixel = node._agent.s2_output.output_pixel
    history = list(node._agent.policy_history)

    epoch = node._invalidate_step3_action_queue_locked(
        "step3-refresh:goal-digest",
        "operation:step3-refresh",
    )

    assert epoch == 1
    assert node._agent.reset_calls == []
    assert node._agent.s2_output.output_action is None
    assert node._agent.s2_output.output_latent is latent
    assert node._agent.s2_output.output_pixel is pixel
    assert node._agent.policy_history == history
    assert node._system1_queue_remaining == 0
    assert node._results == {}
    assert node._test_clear_observations == []
    assert node._test_audit[-1]["event"] == "step3_action_queue_invalidated"
    assert node._test_audit[-1]["policy_history_preserved"] is True
    assert node._test_audit[-1]["system2_latent_preserved"] is True


def test_system2_receding_horizon_discards_only_remaining_actions() -> None:
    node = InternVLAModelNode.__new__(InternVLAModelNode)
    node._agent = _Agent()
    node._system1_queue_remaining = 3
    messages: list[str] = []
    node.get_logger = lambda: SimpleNamespace(info=messages.append)  # type: ignore[method-assign]
    latent = node._agent.s2_output.output_latent
    pixel = node._agent.s2_output.output_pixel
    history = list(node._agent.policy_history)

    node._discard_stale_system2_action_queue(
        RequestIdentity("a::1720", 0, 27, "a::1720:0:27")
    )

    assert node._agent.s2_output.output_action is None
    assert node._agent.s2_output.output_latent is latent
    assert node._agent.s2_output.output_pixel is pixel
    assert node._agent.policy_history == history
    assert node._system1_queue_remaining == 0
    assert "sequence=27 discarded=2" in messages[-1]


def test_system1_receding_horizon_discards_only_remaining_actions() -> None:
    node = InternVLAModelNode.__new__(InternVLAModelNode)
    node._agent = _Agent()
    node._system1_queue_remaining = 2
    messages: list[str] = []
    node.get_logger = lambda: SimpleNamespace(info=messages.append)  # type: ignore[method-assign]
    latent = node._agent.s2_output.output_latent
    pixel = node._agent.s2_output.output_pixel
    history = list(node._agent.policy_history)

    node._discard_stale_system1_action_queue(
        RequestIdentity("a::1720", 0, 66, "a::1720:0:66")
    )

    assert node._agent.s2_output.output_action is None
    assert node._agent.s2_output.output_latent is latent
    assert node._agent.s2_output.output_pixel is pixel
    assert node._agent.policy_history == history
    assert node._system1_queue_remaining == 0
    assert "sequence=66 discarded=2" in messages[-1]


def test_system1_receding_horizon_is_wired_through_exact_t5_lane_contract() -> None:
    model_source = (
        ROOT / "internvla_ros2/internvla_ros2/model_node.py"
    ).read_text(encoding="utf-8")
    fast_source = (
        ROOT / "coordination/run_t5_fast_lane_online.sh"
    ).read_text(encoding="utf-8")
    lane_source = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text(
        encoding="utf-8"
    )

    env_name = "INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON"
    assert env_name in model_source
    assert "ACTION_SOURCE_SYSTEM1_NEW" in model_source
    assert env_name in fast_source
    assert 'test "$lane" = a || usage' in fast_source
    assert env_name in lane_source
    assert '"system1_queue_horizon": int(sys.argv[28])' in lane_source


def test_typed_preservation_is_exact_t5_and_reset_abort_still_clear() -> None:
    recovery_source = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/model_node.py"
    ).read_text(encoding="utf-8")
    base_source = (
        ROOT / "internvla_ros2/internvla_ros2/model_node.py"
    ).read_text(encoding="utf-8")

    assert "preserve_observation_tuple=self._recovery_uses_sim_time" in recovery_source
    assert "self._recovery_uses_sim_time = t5_completion_sim_enabled()" in recovery_source
    assert 'identity.recovery_id.startswith("step3-refresh:")' in recovery_source
    assert "_invalidate_step3_action_queue_locked" in recovery_source
    reset_body = base_source.split("    def _on_reset(", 1)[1].split(
        "    def _on_shutdown(", 1
    )[0]
    abort_body = base_source.split("    def _abort_generation_locked(", 1)[1].split(
        "    def _clear_observations(", 1
    )[0]
    assert "self._clear_observations()" in reset_body
    assert "self._clear_observations()" in abort_body
