from __future__ import annotations

import ast
import os
import socket
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_t5_agent_episode_identity_is_lane_prefixed() -> None:
    path = ROOT / "scripts/internvla_ipc_agent_client.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "_episode_id")
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    episode_id = namespace["_episode_id"]
    client = type("Client", (), {})()
    client.dataset_episode_ids = ("episode-7",)
    client.episode_ordinal = 0
    client.identity_prefix = "a::"
    assert episode_id(client) == "a::episode-7"
    client.identity_prefix = "b::"
    assert episode_id(client) == "b::episode-7"


def test_t5_identity_prefix_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _lane_identity_prefix = _function(
        ROOT / "scripts/internvla_ipc_agent_client.py", "_lane_identity_prefix"
    )

    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "lane-a")
    with pytest.raises(RuntimeError):
        _lane_identity_prefix()
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "a::")
    assert _lane_identity_prefix() == "a::"


@pytest.mark.parametrize(
    ("prefix", "runtime_episode_id", "expected_dataset_episode_id"),
    [
        ("a::", "a::121", "121"),
        ("b::", "b::121", "121"),
    ],
)
def test_t5_static_map_lookup_strips_only_the_configured_lane_prefix(
    prefix: str, runtime_episode_id: str, expected_dataset_episode_id: str
) -> None:
    lookup = _function(
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py",
        "_static_map_dataset_episode_id",
    )
    assert lookup(runtime_episode_id, prefix) == expected_dataset_episode_id


def test_t5_static_map_lookup_rejects_foreign_and_unscoped_identity() -> None:
    lookup = _function(
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py",
        "_static_map_dataset_episode_id",
    )
    with pytest.raises(ValueError, match="does not belong"):
        lookup("b::121", "a::")
    with pytest.raises(ValueError, match="does not belong"):
        lookup("121", "a::")


def test_t5_static_map_defers_only_explicit_state_only_bootstrap_identity() -> None:
    lookup = _function(
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py",
        "_static_map_dataset_episode_id",
    )
    assert (
        lookup("bootstrap-episode-0", "a::", state_only=True)
        == "bootstrap-episode-0"
    )
    with pytest.raises(ValueError, match="does not belong"):
        lookup("bootstrap-episode-0", "a::", state_only=False)
    with pytest.raises(ValueError, match="does not belong"):
        lookup("b::121", "a::", state_only=True)
    with pytest.raises(ValueError, match="does not belong"):
        lookup("bootstrap-episode-forged", "a::", state_only=True)


def test_sensor_bridge_lane_prefix_environment_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane_prefix = _function(
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py",
        "_lane_identity_prefix",
    )
    monkeypatch.delenv("INTERNNAV_T5_ID_PREFIX", raising=False)
    assert lane_prefix() == ""
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "b::")
    assert lane_prefix() == "b::"
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "lane-b")
    with pytest.raises(RuntimeError, match="must be empty"):
        lane_prefix()


def test_t5_bootstrap_tf_uses_steady_timer_without_changing_t4_default() -> None:
    source = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
    ).read_text(encoding="utf-8")
    assert "if self.lane_identity_prefix:" in source
    assert "clock_type=ClockType.STEADY_TIME" in source
    assert "clock=self._bootstrap_timer_clock" in source
    assert "else:\n            # Empty Lane identity is the frozen T4 path." in source
    assert "self.create_timer(0.05, self._publish_bootstrap_tf)" in source
    assert "transform.header.stamp = self.get_clock().now().to_msg()" in source
    assert "if self._real_pose_published:\n                return" in source


def test_t4_static_map_lookup_keeps_empty_prefix_identity_unchanged() -> None:
    path = ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
    lookup = _function(path, "_static_map_dataset_episode_id")
    assert lookup("121", "") == "121"

    source = path.read_text(encoding="utf-8")
    assert "self._pending_dataset_episode_id = dataset_episode_id" in source
    assert '"episode_id": self._pending_episode_id' in source
    assert "identity: tuple[str, int, int] | None = None" in source
    assert "replay_epoch: int | None = None" in source
    assert source.count("identity: tuple[str, int, int] | None = None") == 2
    assert source.count("replay_epoch: int | None = None") == 2


def test_t4_ipc_default_and_two_t5_aliases_are_the_only_options() -> None:
    runner = (ROOT / "scripts/run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    assert 'INTERNVLA_T4_IPC_ALIAS_OVERRIDE:-/tmp/internnav_t4_ipc' in runner
    assert "/tmp/internnav_t5_a_ipc" in runner
    assert "/tmp/internnav_t5_b_ipc" in runner


def test_oracle_request_and_reset_audits_include_lane_prefix() -> None:
    oracle = (ROOT / "internvla_ros2/internvla_ros2/oracle_node.py").read_text()
    client = (ROOT / "internvla_ros2/internvla_ros2/client_node.py").read_text()
    assert 'f"{self.identity_prefix}oracle:' in oracle
    assert '"reset_id": f"{self.identity_prefix}{self.generation}"' in oracle
    assert '"reset_id": f"{self.identity_prefix}{self.reset_generation}"' in client
    assert "class OraclePeerClosed(EOFError)" in oracle
    assert "except OraclePeerClosed:" in oracle
    assert "clean_eof_ok=True" in oracle


def test_oracle_transport_accepts_only_clean_between_frame_eof() -> None:
    path = ROOT / "internvla_ros2/internvla_ros2/oracle_node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    peer_closed = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "OraclePeerClosed"
    )
    recv_exact_node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_recv_exact"
    )
    namespace: dict[str, object] = {"socket": socket}
    exec(
        compile(
            ast.Module(body=[peer_closed, recv_exact_node], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    recv_exact = namespace["_recv_exact"]
    clean_error = namespace["OraclePeerClosed"]

    reader, writer = socket.socketpair()
    writer.close()
    with reader:
        with pytest.raises(clean_error):
            recv_exact(reader, 4, clean_eof_ok=True)

    reader, writer = socket.socketpair()
    writer.sendall(b"x")
    writer.close()
    with reader:
        with pytest.raises(ConnectionError, match="disconnected"):
            recv_exact(reader, 4, clean_eof_ok=True)
