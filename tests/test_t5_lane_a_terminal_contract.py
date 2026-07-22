from __future__ import annotations

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
from internvla_t4_recovery.recovery_node import RecoverySupervisor  # noqa: E402
from internvla_t4_sensors import client_node as t4_client_module  # noqa: E402
from internvla_t4_sensors.client_node import T4OdometryClientNode  # noqa: E402


def _command(
    *,
    episode: str = "a::259",
    generation: int = 1,
    sequence: int = 17,
    source: int = 3,
    action: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        episode_id=episode,
        reset_generation=generation,
        sequence_id=sequence,
        request_id=f"{episode}:{generation}:{sequence}",
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
    return client


def _primitive(
    *, action: int = 2, sequence: int = 17, generation: int = 1
) -> object:
    return system2_primitive_signature(
        action=action,
        episode_id="a::259",
        reset_generation=generation,
        sequence_id=sequence,
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


def test_shared_system2_primitive_signature_has_stable_semantic_shape() -> None:
    signature = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=17,
        x=1.25,
        y=-0.5,
        yaw_rad=0.0,
    )
    same_action = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=18,
        x=1.251,
        y=-0.499,
        yaw_rad=0.001,
    )
    different_action = _primitive(action=3, sequence=18)

    assert same_action.absolute_sha256 != signature.absolute_sha256
    assert same_action.shape_sha256 == signature.shape_sha256
    assert different_action.shape_sha256 != signature.shape_sha256


def test_client_rejects_same_system2_primitive_and_accepts_different_action(
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
