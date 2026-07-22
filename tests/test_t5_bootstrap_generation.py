from __future__ import annotations

import ast
import json
import socket
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_go2_controller"))

from internvla_go2_controller.runtime import ControllerIPCClient  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from internnav_go2_runtime import (  # noqa: E402
    _continuous_state_only_identity,
    _t5_continuation_bootstrap_identity_enabled,
)


BRIDGE = ROOT / "internvla_go2_controller/internvla_go2_controller/bridge_node.py"
ISAAC_RUNTIME = ROOT / "scripts/internnav_go2_runtime.py"


def _load_pure_function(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace: dict[str, Any] = {"Any": Any}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def _receive_json(connection: socket.socket) -> dict[str, Any]:
    header = connection.recv(4)
    assert len(header) == 4
    remaining = struct.unpack("!I", header)[0]
    payload = bytearray()
    while remaining:
        block = connection.recv(remaining)
        assert block
        payload.extend(block)
        remaining -= len(block)
    value = json.loads(payload.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def _send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _query_once(response: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    client_socket, server_socket = socket.socketpair()
    observed: dict[str, Any] = {}

    def serve() -> None:
        try:
            observed.update(_receive_json(server_socket))
            _send_json(server_socket, response)
        finally:
            server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = ControllerIPCClient("/unused/t5-controller.sock")
    client.connection = client_socket
    try:
        identity = client.query_active_identity(expected_episode_prefix="a::")
    finally:
        client.close()
        thread.join(timeout=2.0)
    assert not thread.is_alive()
    return identity.reset_generation, observed


def test_replacement_evaluators_query_persistent_generation_instead_of_zero() -> None:
    query_response = {
        "schema_version": 1,
        "status": "ok",
        "operation": "query_active_identity",
        "active_episode_id": "a::episode-5",
        "active_reset_generation": 5,
        "active_sequence_id": 103,
        "state_replay_epoch": 5,
        "linear_x": 0.0,
        "angular_z": 0.0,
        "emergency_stop": True,
    }

    first_generation, first_request = _query_once(query_response)
    continuation_generation, continuation_request = _query_once(query_response)

    assert first_generation == continuation_generation == 5
    assert first_request == continuation_request == {
        "schema_version": 1,
        "operation": "query_active_identity",
    }


def test_initial_t5_evaluator_keeps_local_zero_bootstrap(monkeypatch) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "0")

    assert _t5_continuation_bootstrap_identity_enabled() is False


def test_only_explicit_t5_continuation_queries_persistent_identity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "1")

    assert _t5_continuation_bootstrap_identity_enabled() is True

    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "unexpected")
    with pytest.raises(RuntimeError, match="invalid T5 evaluator continuation"):
        _t5_continuation_bootstrap_identity_enabled()


def test_initial_state_only_keeps_bounded_bootstrap_identity(monkeypatch) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "0")

    assert _continuous_state_only_identity(
        SimpleNamespace(
            episode_id="a::real-episode",
            reset_generation=8,
            sequence_id=0,
        ),
        0,
    ) == ("bootstrap-episode-0", 0, 0, True)


def test_continuation_state_only_uses_post_reset_real_identity(monkeypatch) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "1")

    assert _continuous_state_only_identity(
        SimpleNamespace(
            episode_id="a::real-episode",
            reset_generation=8,
            sequence_id=0,
        ),
        7,
    ) == ("a::real-episode", 8, 0, True)


def test_continuation_state_only_preserves_safe_hold_sequence(monkeypatch) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "1")

    assert _continuous_state_only_identity(
        SimpleNamespace(
            episode_id="a::real-episode",
            reset_generation=8,
            sequence_id=4,
        ),
        8,
    ) == ("a::real-episode", 8, 4, True)


@pytest.mark.parametrize(
    "identity,bootstrap_generation",
    [
        (
            SimpleNamespace(
                episode_id="b::cross-lane", reset_generation=8, sequence_id=0
            ),
            7,
        ),
        (
            SimpleNamespace(
                episode_id="a::stale", reset_generation=6, sequence_id=0
            ),
            7,
        ),
        (
            SimpleNamespace(
                episode_id="a::invalid", reset_generation=8, sequence_id=-1
            ),
            7,
        ),
    ],
)
def test_continuation_state_only_rejects_nonexact_reset_identity(
    monkeypatch, identity: SimpleNamespace, bootstrap_generation: int
) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "1")

    with pytest.raises(RuntimeError, match="no current identity"):
        _continuous_state_only_identity(identity, bootstrap_generation)


def test_active_identity_query_is_read_only_safe_stop_and_fails_closed() -> None:
    response = _load_pure_function(BRIDGE, "_active_identity_query_response")
    available = response(("a::episode-5", 5, 103), 7)
    assert available["status"] == "ok"
    assert available["active_reset_generation"] == 5
    assert available["state_replay_epoch"] == 7
    assert available["linear_x"] == available["angular_z"] == 0.0
    assert available["emergency_stop"] is True

    unavailable = response(("", -1, -1), 0)
    assert unavailable["status"] == "error"
    assert unavailable["emergency_stop"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_episode_id", "b::wrong-lane"),
        ("active_reset_generation", -1),
        ("active_reset_generation", True),
        ("active_sequence_id", -2),
    ],
)
def test_identity_query_rejects_invalid_or_cross_lane_response(
    field: str, value: object
) -> None:
    response = {
        "schema_version": 1,
        "status": "ok",
        "operation": "query_active_identity",
        "active_episode_id": "a::episode-5",
        "active_reset_generation": 5,
        "active_sequence_id": 103,
    }
    response[field] = value
    with pytest.raises(RuntimeError):
        _query_once(response)


def test_t5_post_reset_binds_after_runtime_reset_while_t4_keeps_local_counter() -> None:
    source = ISAAC_RUNTIME.read_text(encoding="utf-8")
    bridge_source = BRIDGE.read_text(encoding="utf-8")
    bind_call = source.index("].bind_persistent_bootstrap_generation()")
    reset_call = source.rfind("reset()", 0, bind_call)
    assert reset_call >= 0
    assert bind_call > reset_call
    assert "continuous_reset_generation += 1" in source
    assert "continuous.set_bootstrap_generation(continuous_reset_generation)" in source
    assert "continuous.set_bootstrap_generation(-1)" in source
    assert 'expected_episode_prefix=f"{lane}::"' in source
    assert "_t5_continuation_bootstrap_identity_enabled()" in source
    assert 'continuation == "1"' in source
    assert "response = self._handle_ipc_request(request)" in bridge_source
    assert 'request.get("operation") != "query_active_identity"' in bridge_source
