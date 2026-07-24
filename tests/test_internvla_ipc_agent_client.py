from __future__ import annotations

import sys
import types
import gzip
import hashlib
import json
from pathlib import Path

import pytest

_NUMPY_STUBBED = False
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    _NUMPY_STUBBED = True


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
CONTROLLER_PACKAGE = SCRIPTS.parent / "internvla_go2_controller"
if str(CONTROLLER_PACKAGE) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_PACKAGE))

from internvla_ipc_agent_client import (  # noqa: E402
    ROS2IPCAgentClient,
    SAFE_STOP_ACTION,
    _dataset_episode_ids,
    _ordered_dataset_episode_ids,
)
import internvla_ipc_agent_client as ipc_client_module  # noqa: E402
import internnav_go2_runtime as go2_runtime_module  # noqa: E402
import internvla_go2_continuous_agent_client as continuous_client_module  # noqa: E402

if _NUMPY_STUBBED:
    sys.modules.pop("numpy", None)


class _Connection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Config:
    model_name = "InternVLA-N1"


class _FakeImage:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return b"bounded-test-image"


class _FakeListArray:
    def __init__(self, value) -> None:
        self.value = value

    def tolist(self):
        return list(self.value)


class _FakeNumpy:
    uint8 = "uint8"
    float32 = "float32"
    int64 = "int64"
    float64 = "float64"

    @staticmethod
    def ascontiguousarray(value, dtype):
        del dtype
        return value

    @staticmethod
    def asarray(value, dtype):
        del dtype
        return _FakeListArray(value)


class _FaultSnapshot:
    def __init__(
        self, active: dict[str, str] | None = None, observed_sim_ns: int = 1
    ) -> None:
        self.active = active or {}
        self.observed_sim_ns = observed_sim_ns

    def event_for(self, kind: str) -> str | None:
        return self.active.get(kind)


class _FaultControl:
    def __init__(self, snapshots: list[_FaultSnapshot]) -> None:
        self.snapshots = list(snapshots)
        self.records: list[tuple[object, ...]] = []

    def read(self) -> _FaultSnapshot:
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def record(self, *values: object) -> None:
        self.records.append(values)


def _valid_observation() -> list[dict[str, object]]:
    return [
        {
            "rgb": _FakeImage((480, 640, 3), "uint8"),
            "depth": _FakeImage((480, 640, 1), "float32"),
            "instruction": "go",
            "instruction_tokens": [1, 2],
        }
    ]


def _step_client() -> ROS2IPCAgentClient:
    client = _unconnected_client(allow_reconnect=True)
    client.connection = _Connection()
    client.tcp_endpoint = "tcp://unused:25140"
    return client


def _unconnected_client(*, allow_reconnect: bool) -> ROS2IPCAgentClient:
    client = ROS2IPCAgentClient.__new__(ROS2IPCAgentClient)
    client.connection = None
    client.allow_episode_reset_reconnect = allow_reconnect
    client.episode_ordinal = 0
    client.last_result = None
    client.last_error = "previous step failed"
    return client


def test_fault_timeout_classification_is_exact() -> None:
    snapshot = _FaultSnapshot(
        {"model_request_timeout": "fi-01-model-request-timeout"}
    )
    exact = RuntimeError(
        "local InternVLA step failed: 4 injected T5 completion_sim model request timeout"
    )
    assert (
        ROS2IPCAgentClient._expected_model_timeout_event(exact, snapshot)
        == "fi-01-model-request-timeout"
    )
    assert (
        ROS2IPCAgentClient._expected_model_timeout_event(
            RuntimeError(str(exact) + " drift"), snapshot
        )
        is None
    )
    assert (
        ROS2IPCAgentClient._expected_model_timeout_event(
            exact, _FaultSnapshot()
        )
        is None
    )


def test_restart_maintenance_waits_for_clear_then_checks_same_episode(
    monkeypatch,
) -> None:
    client = _unconnected_client(allow_reconnect=True)
    client.identity_prefix = "a::"
    client.dataset_episode_ids = ("episode-0",)
    active = _FaultSnapshot(
        {"model_service_restart": "fi-02-model-service-restart"}, 100
    )
    control = _FaultControl([active, _FaultSnapshot({}, 100)])
    client._fault_control = control
    requests: list[dict[str, object]] = []
    reconnects: list[bool] = []

    def exchange(request: dict[str, object]) -> dict[str, object]:
        requests.append(request)
        return {
            "schema_version": 1,
            "status": "ok",
            "status_code": 0,
            "initialized": True,
            "episode_id": "a::episode-0",
        }

    monkeypatch.setattr(client, "_exchange", exchange)
    monkeypatch.setattr(client, "_connect", lambda: reconnects.append(True))
    monkeypatch.setattr(ipc_client_module.time, "sleep", lambda _seconds: None)
    client._wait_for_fault_restart_maintenance(active)

    assert requests == [{"schema_version": 1, "operation": "health"}]
    assert reconnects == [True]
    assert [record[2] for record in control.records] == [
        "maintenance_wait_started",
        "maintenance_wait_finished",
    ]


def test_completion_sim_reset_reconnects_once_after_safe_stop(monkeypatch, capsys) -> None:
    client = _unconnected_client(allow_reconnect=True)
    connection = _Connection()
    requests: list[dict[str, object]] = []

    def connect() -> None:
        client.connection = connection

    def exchange(request: dict[str, object]) -> dict[str, object]:
        requests.append(request)
        return {
            "schema_version": 1,
            "status": "ok",
            "status_code": 0,
            "episode_id": "isaac-evaluator-episode-1",
            "reset_generation": 1,
        }

    monkeypatch.setattr(client, "_connect", connect)
    monkeypatch.setattr(client, "_exchange", exchange)

    response = client.reset()

    assert response["reset_generation"] == 1
    assert requests == [
        {
            "schema_version": 1,
            "operation": "reset",
            "next_episode_id": "isaac-evaluator-episode-1",
        }
    ]
    assert client.connection is connection
    assert client.last_error == ""
    assert "INTERNVLA_LOCAL_IPC_RESET_RECONNECTED" in capsys.readouterr().err


def test_strict_reset_does_not_reconnect() -> None:
    client = _unconnected_client(allow_reconnect=False)
    with pytest.raises(ConnectionError, match="not connected"):
        client.reset()
    assert client.connection is None


def test_initial_evaluator_cycle_only_initializes(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    def connect(client: ROS2IPCAgentClient) -> None:
        client.connection = _Connection()

    def exchange(
        _client: ROS2IPCAgentClient, request: dict[str, object]
    ) -> dict[str, object]:
        requests.append(request)
        return {
            "schema_version": 1,
            "status": "ok",
            "status_code": 0,
            "episode_id": "a::isaac-evaluator-episode-0",
            "reset_generation": 0,
        }

    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "a::")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "0")
    monkeypatch.setattr(ROS2IPCAgentClient, "_connect", connect)
    monkeypatch.setattr(ROS2IPCAgentClient, "_exchange", exchange)

    client = ROS2IPCAgentClient(_Config())

    assert [request["operation"] for request in requests] == ["initialize"]
    assert client.handshake_response["reset_generation"] == 0


def test_completion_sim_continuation_resets_once_after_exact_initialize_conflict(
    monkeypatch, capsys
) -> None:
    requests: list[dict[str, object]] = []
    connections: list[_Connection] = []

    def connect(client: ROS2IPCAgentClient) -> None:
        connection = _Connection()
        connections.append(connection)
        client.connection = connection

    def exchange(
        _client: ROS2IPCAgentClient, request: dict[str, object]
    ) -> dict[str, object]:
        requests.append(request)
        if request["operation"] == "initialize":
            return {
                "schema_version": 1,
                "status": "error",
                "status_code": 2,
                "message": "already initialized for a different episode; use reset",
            }
        return {
            "schema_version": 1,
            "status": "ok",
            "status_code": 0,
            "episode_id": "a::isaac-evaluator-episode-0",
            "reset_generation": 5,
        }

    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "a::")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "1")
    monkeypatch.setattr(ROS2IPCAgentClient, "_connect", connect)
    monkeypatch.setattr(ROS2IPCAgentClient, "_exchange", exchange)

    client = ROS2IPCAgentClient(_Config())

    assert [request["operation"] for request in requests] == ["initialize", "reset"]
    assert requests[1]["next_episode_id"] == "a::isaac-evaluator-episode-0"
    assert len(connections) == 2
    assert connections[0].closed is True
    assert client.connection is connections[1]
    assert client.handshake_response["reset_generation"] == 5
    assert "INTERNVLA_LOCAL_IPC_CONTINUATION_RESET" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("allow_reconnect", "continuation", "status_code", "message"),
    [
        (False, "1", 2, "already initialized for a different episode; use reset"),
        (True, "0", 2, "already initialized for a different episode; use reset"),
        (True, "1", 3, "already initialized for a different episode; use reset"),
        (True, "1", 2, "different failure"),
    ],
)
def test_continuation_reset_is_exactly_gated(
    monkeypatch,
    allow_reconnect: bool,
    continuation: str,
    status_code: int,
    message: str,
) -> None:
    client = _unconnected_client(allow_reconnect=allow_reconnect)
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", continuation)
    assert (
        client._requires_evaluator_continuation_reset(
            {
                "status": "error",
                "status_code": status_code,
                "message": message,
            }
        )
        is False
    )


def test_continuous_client_uses_continuation_handshake_generation(monkeypatch) -> None:
    execution_identities: list[tuple[str, int]] = []
    obstacle_generations: list[int] = []

    def parent_init(client: ROS2IPCAgentClient, _config: object) -> None:
        client.connection = None
        client.handshake_response = {
            "episode_id": "a::episode-zero",
            "reset_generation": 5,
        }

    monkeypatch.setattr(ROS2IPCAgentClient, "__init__", parent_init)
    monkeypatch.setattr(continuous_client_module, "_scenario_manifest", lambda: ({}, {}))
    monkeypatch.setattr(
        continuous_client_module,
        "reset_execution_identity",
        lambda episode_id, generation: execution_identities.append(
            (episode_id, generation)
        ),
    )
    monkeypatch.setattr(
        continuous_client_module,
        "reset_obstacle_scenario",
        obstacle_generations.append,
    )

    continuous_client_module.ContinuousROS2IPCAgentClient(_Config())

    assert execution_identities == [("a::episode-zero", 5)]
    assert obstacle_generations == [5]


def test_continuous_evaluator_only_terminates_for_task_level_stop() -> None:
    gate_hold = {
        "stop": True,
        "model_stop": False,
        "motion_observation_gate_only": True,
    }
    assert continuous_client_module._continuous_evaluator_action(gate_hold) == (
        continuous_client_module.CONTINUOUS_TRIGGER
    )
    assert continuous_client_module._continuous_execution_stop(gate_hold) is True
    assert continuous_client_module._continuous_evaluator_action(
        {"stop": True, "model_stop": True}
    ) == continuous_client_module.SAFE_STOP
    assert continuous_client_module._continuous_execution_stop(
        {"stop": True, "model_stop": True}
    ) is True
    assert continuous_client_module._continuous_evaluator_action(
        {"stop": True}
    ) == continuous_client_module.SAFE_STOP


@pytest.mark.parametrize(
    ("gate_only", "expected_action"),
    [
        (True, continuous_client_module.CONTINUOUS_TRIGGER),
        (False, continuous_client_module.SAFE_STOP),
    ],
)
def test_continuous_wrapper_keeps_gate_stop_latched_without_false_termination(
    monkeypatch, gate_only: bool, expected_action: list[dict[str, object]]
) -> None:
    response = {
        "episode_id": "a::145",
        "reset_generation": 0,
        "sequence_id": 3,
        "stop": True,
        "model_stop": not gate_only,
        "motion_observation_gate_only": gate_only,
    }
    client = continuous_client_module.ContinuousROS2IPCAgentClient.__new__(
        continuous_client_module.ContinuousROS2IPCAgentClient
    )
    client.connection = None
    client.scenario_by_instruction_digest = {}
    execution_stops: list[bool] = []

    def parent_step(instance, _obs):
        instance.last_result = response
        return [{"action": [-1], "ideal_flag": True}]

    monkeypatch.setattr(ROS2IPCAgentClient, "step", parent_step)
    monkeypatch.setattr(
        continuous_client_module,
        "set_execution_identity",
        lambda _episode, _generation, _sequence, *, stop: execution_stops.append(
            stop
        ),
    )

    action = client.step([{"instruction": "go"}])

    assert action == expected_action
    assert execution_stops == [True]


def test_step_failure_records_diagnostic_and_closes_before_safe_stop(capsys) -> None:
    client = _unconnected_client(allow_reconnect=True)
    connection = _Connection()
    client.connection = connection

    result = client.step([])

    assert result == SAFE_STOP_ACTION
    assert connection.closed is True
    assert client.connection is None
    assert "one observation dictionary" in client.last_error
    stderr = capsys.readouterr().err
    assert "INTERNVLA_LOCAL_IPC_STEP_ERROR" in stderr
    assert "INTERNVLA_MODEL_ACTION_OK" not in stderr


def test_continuous_wrapper_aborts_batch_after_unexpected_ipc_safe_stop(
    monkeypatch,
) -> None:
    client = continuous_client_module.ContinuousROS2IPCAgentClient.__new__(
        continuous_client_module.ContinuousROS2IPCAgentClient
    )
    client.connection = None
    client.scenario_by_instruction_digest = {}

    def parent_step(instance, _obs):
        instance.last_result = None
        instance.last_error = "RuntimeError('model failed')"
        instance.last_step_safe_stop_kind = "unexpected_ipc_error"
        return SAFE_STOP_ACTION

    monkeypatch.setattr(ROS2IPCAgentClient, "step", parent_step)

    with pytest.raises(RuntimeError, match="unexpected InternVLA IPC step failure"):
        client.step([{"instruction": "go"}])


def test_continuous_wrapper_preserves_expected_fault_safe_stop(
    monkeypatch,
) -> None:
    client = continuous_client_module.ContinuousROS2IPCAgentClient.__new__(
        continuous_client_module.ContinuousROS2IPCAgentClient
    )
    client.connection = None
    client.scenario_by_instruction_digest = {}

    def parent_step(instance, _obs):
        instance.last_result = None
        instance.last_error = "injected timeout"
        instance.last_step_safe_stop_kind = "expected_model_timeout"
        return SAFE_STOP_ACTION

    monkeypatch.setattr(ROS2IPCAgentClient, "step", parent_step)

    assert client.step([{"instruction": "go"}]) == SAFE_STOP_ACTION


def test_successful_step_emits_exactly_one_model_action_marker(
    monkeypatch, capsys
) -> None:
    client = _step_client()
    client.episode_ordinal = 4
    response = {
        "schema_version": 1,
        "status": "ok",
        "status_code": 0,
        "discrete_action": "7",
    }
    monkeypatch.setattr(ipc_client_module, "np", _FakeNumpy)
    monkeypatch.setattr(client, "_exchange", lambda _request: response)

    result = client.step(_valid_observation())

    assert result == [{"action": [7], "ideal_flag": True}]
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("INTERNVLA_MODEL_ACTION_OK ")
    ]
    assert len(lines) == 1
    marker = json.loads(lines[0].split(" ", 1)[1])
    assert marker == {
        "schema_version": 1,
        "episode_ordinal": 4,
        "discrete_action": 7,
    }


def test_failed_step_response_does_not_emit_model_action_marker(
    monkeypatch, capsys
) -> None:
    client = _step_client()
    monkeypatch.setattr(ipc_client_module, "np", _FakeNumpy)
    monkeypatch.setattr(
        client,
        "_exchange",
        lambda _request: {
            "schema_version": 1,
            "status": "error",
            "status_code": 13,
            "message": "bounded failure",
        },
    )

    result = client.step(_valid_observation())

    assert result == SAFE_STOP_ACTION
    stderr = capsys.readouterr().err
    assert "INTERNVLA_LOCAL_IPC_STEP_ERROR" in stderr
    assert "INTERNVLA_MODEL_ACTION_OK" not in stderr


def test_t5_step_forwards_real_source_frame_metadata_and_rejects_replay(
    monkeypatch,
) -> None:
    client = _step_client()
    client._t5_camera_sensor_identity = True
    client._t5_last_camera_sensor_sequence = 0
    client._t5_last_camera_sensor_stamp_ns = 0
    requests: list[dict[str, object]] = []

    def exchange(request):
        requests.append(request)
        return {
            "schema_version": 1,
            "status": "ok",
            "status_code": 0,
            "discrete_action": 1,
        }

    observation = _valid_observation()
    observation[0]["camera_sensor_metadata"] = {
        "schema_version": 1,
        "source": "x86_isaac_pano_camera_0",
        "sequence": 7,
        "sim_stamp_ns": 123_000_000_000,
    }
    monkeypatch.setattr(ipc_client_module, "np", _FakeNumpy)
    monkeypatch.setattr(client, "_exchange", exchange)

    assert client.step(observation) == [{"action": [1], "ideal_flag": True}]
    assert requests[0]["camera_sensor_metadata"] == observation[0][
        "camera_sensor_metadata"
    ]
    assert client.step(observation) == SAFE_STOP_ACTION
    assert "did not strictly advance" in client.last_error


def test_x86_sensor_capture_uses_source_sim_clock_and_is_t5_only(
    monkeypatch,
) -> None:
    observation = {"rgb": object(), "depth": object()}
    monkeypatch.delenv("INTERNNAV_RUNTIME_POLICY", raising=False)
    assert (
        go2_runtime_module._attach_t5_camera_source_metadata(dict(observation))
        == observation
    )

    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setattr(go2_runtime_module, "_T5_SIM_CLOCK_NS", 321_000_000_000)
    monkeypatch.setattr(go2_runtime_module, "_T5_CAMERA_SOURCE_SEQUENCE", 10)
    stamped = go2_runtime_module._attach_t5_camera_source_metadata(
        dict(observation)
    )
    assert stamped["camera_sensor_metadata"] == {
        "schema_version": 1,
        "source": "x86_isaac_pano_camera_0",
        "sequence": 11,
        "sim_stamp_ns": 321_000_000_000,
    }


def test_t5_camera_sequence_continues_from_bounded_completion_seed(
    monkeypatch,
) -> None:
    observation = {"rgb": object(), "depth": object()}
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_CAMERA_SOURCE_SEQUENCE_START", "900")
    monkeypatch.setattr(go2_runtime_module, "_T5_SIM_CLOCK_NS", 321_000_000_000)
    monkeypatch.setattr(go2_runtime_module, "_T5_CAMERA_SOURCE_SEQUENCE", 0)

    stamped = go2_runtime_module._attach_t5_camera_source_metadata(
        dict(observation)
    )

    assert stamped["camera_sensor_metadata"]["sequence"] == 901


def test_t5_camera_sequence_seed_cannot_escape_completion_sim(monkeypatch) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "strict_evidence")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_CAMERA_SOURCE_SEQUENCE_START", "900")
    monkeypatch.setattr(go2_runtime_module, "_T5_SIM_CLOCK_NS", 321_000_000_000)
    monkeypatch.setattr(go2_runtime_module, "_T5_CAMERA_SOURCE_SEQUENCE", 0)

    with pytest.raises(RuntimeError, match="escaped completion_sim"):
        go2_runtime_module._next_t5_camera_source_sequence()


def test_t5_sim_clock_continues_from_bounded_completion_seed(monkeypatch) -> None:
    class _ClockSocket:
        def __init__(self) -> None:
            self.sent: list[tuple[bytes, tuple[str, int]]] = []

        def sendto(self, payload, endpoint) -> None:
            self.sent.append((payload, endpoint))

    clock_socket = _ClockSocket()
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_CLOCK_UDP_ENDPOINT", "127.0.0.1:47911")
    monkeypatch.setenv("INTERNVLA_T5_SIM_CLOCK_START_NS", "321000000000")
    monkeypatch.setattr(go2_runtime_module, "_T5_SIM_CLOCK_NS", None)
    monkeypatch.setattr(go2_runtime_module, "_T5_CLOCK_SOCKET", clock_socket)

    go2_runtime_module._advance_t5_sim_clock(0.02)

    assert go2_runtime_module._T5_SIM_CLOCK_NS == 321_020_000_000
    assert clock_socket.sent == [
        (b"321020000000", ("127.0.0.1", 47911))
    ]


def test_t5_sim_clock_seed_cannot_escape_completion_sim(monkeypatch) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "strict_evidence")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_CLOCK_UDP_ENDPOINT", "127.0.0.1:47911")
    monkeypatch.setenv("INTERNVLA_T5_SIM_CLOCK_START_NS", "321000000000")
    monkeypatch.setattr(go2_runtime_module, "_T5_SIM_CLOCK_NS", None)
    monkeypatch.setattr(go2_runtime_module, "_T5_CLOCK_SOCKET", None)

    with pytest.raises(RuntimeError, match="escaped completion_sim"):
        go2_runtime_module._advance_t5_sim_clock(0.02)


def test_completion_model_identity_uses_frozen_dataset_order(tmp_path: Path) -> None:
    dataset = tmp_path / "val_unseen.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump(
            {"episodes": [{"episode_id": "145"}, {"episode_id": "1720"}]},
            stream,
        )
    client = _unconnected_client(allow_reconnect=True)
    client.dataset_episode_ids = _dataset_episode_ids(str(dataset))
    assert client._episode_id() == "145"
    client.episode_ordinal = 1
    assert client._episode_id() == "1720"
    client.episode_ordinal = 2
    with pytest.raises(RuntimeError, match="beyond frozen model dataset"):
        client._episode_id()


def test_dataset_episode_identity_rejects_duplicates(tmp_path: Path) -> None:
    dataset = tmp_path / "duplicates.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump(
            {"episodes": [{"episode_id": "same"}, {"episode_id": "same"}]},
            stream,
        )
    with pytest.raises(RuntimeError, match="duplicate episode_id"):
        _dataset_episode_ids(str(dataset))


def test_t5_model_identity_follows_bound_materialized_order_manifest(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "val_unseen.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "episodes": [
                    {"trajectory_id": "t1", "episode_id": "121"},
                    {"trajectory_id": "t2", "episode_id": "259"},
                    {"trajectory_id": "t3", "episode_id": "628"},
                ]
            },
            stream,
        )
    manifest = tmp_path / "ordered_episode_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "dataset_episode_count": 3,
                "raw_episode_keys": ["t1_121", "t2_259", "t3_628"],
                "ordered_episode_keys": ["t3_628", "t2_259", "t1_121"],
                "ordered_episode_ids": ["628", "259", "121"],
            }
        ),
        encoding="utf-8",
    )
    assert _ordered_dataset_episode_ids(str(dataset), str(manifest)) == (
        "628",
        "259",
        "121",
    )
    assert _ordered_dataset_episode_ids(str(dataset), "") == (
        "121",
        "259",
        "628",
    )


def test_dataset_episode_identity_rejects_unbound_order_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "val_unseen.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump(
            {"episodes": [{"trajectory_id": "t", "episode_id": "121"}]},
            stream,
        )
    manifest = tmp_path / "bad-order.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "dataset_sha256": "0" * 64,
                "dataset_episode_count": 1,
                "raw_episode_keys": ["t_121"],
                "ordered_episode_keys": ["t_121"],
                "ordered_episode_ids": ["121"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        _ordered_dataset_episode_ids(str(dataset), str(manifest))
