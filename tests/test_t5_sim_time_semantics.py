from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "internvla_nav2_adapter/internvla_nav2_adapter/active_node.py"
CONTROLLER = ROOT / "internvla_go2_controller/internvla_go2_controller/bridge_node.py"
T4_CLIENT = ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
T4_RECOVERY_ADAPTER = (
    ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
)
T4_SENSOR_BRIDGE = (
    ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
)


def _load_clock_helpers(path: Path) -> tuple[dict[str, Any], SimpleNamespace]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_semantic_now", "_semantic_age"}
    ]
    assert {node.name for node in definitions} == {"_semantic_now", "_semantic_age"}
    module = ast.Module(body=definitions, type_ignores=[])
    fake_time = SimpleNamespace(monotonic=lambda: 42.0)
    namespace: dict[str, Any] = {"Any": Any, "time": fake_time}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, fake_time


def _load_pure_function(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("path", [ACTIVE, CONTROLLER])
def test_exact_t5_freshness_uses_sim_time_and_fails_closed(path: Path) -> None:
    namespace, fake_time = _load_clock_helpers(path)
    clock = SimpleNamespace(value=10_000_000_000)
    node = SimpleNamespace(
        _t5_sim_time_semantics=True,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=clock.value)
        ),
    )
    fake_time.monotonic = lambda: pytest.fail("T5 semantic age read wall time")

    assert namespace["_semantic_now"](node) == pytest.approx(10.0)
    assert namespace["_semantic_age"](node, 9.8) == pytest.approx(0.2)

    clock.value = 0
    assert namespace["_semantic_age"](node, 9.8) is None
    clock.value = 9_700_000_000
    assert namespace["_semantic_age"](node, 9.8) is None
    assert namespace["_semantic_now"](node) == 0.0
    assert namespace["_semantic_age"](node, 9.7) is None
    clock.value = 10_000_000_000
    assert namespace["_semantic_age"](node, 9.7) == pytest.approx(0.3)


@pytest.mark.parametrize("path", [ACTIVE, CONTROLLER])
def test_non_t5_freshness_retains_legacy_monotonic_clock(path: Path) -> None:
    namespace, _fake_time = _load_clock_helpers(path)
    node = SimpleNamespace(_t5_sim_time_semantics=False)
    assert namespace["_semantic_now"](node) == 42.0
    assert namespace["_semantic_age"](node, 41.5) == 0.5


def test_same_sim_stamp_requires_a_post_barrier_command_receipt() -> None:
    fresh = _load_pure_function(CONTROLLER, "_command_is_fresh")
    common = {
        "safe_cmd_stamp": 10.0,
        "barrier_stamp": 10.0,
        "command_age": 0.0,
        "timeout_sec": 0.3,
    }
    assert not fresh(safe_cmd_serial=7, barrier_cmd_serial=7, **common)
    assert fresh(safe_cmd_serial=8, barrier_cmd_serial=7, **common)


def test_completion_sim_collision_is_metric_only_but_other_hazards_remain_fatal() -> None:
    hazard = _load_pure_function(CONTROLLER, "_runtime_state_hazard")

    assert not hazard(
        t5_completion_sim=True,
        nan_detected=False,
        fallen=False,
        physical_collision=True,
    )
    assert hazard(
        t5_completion_sim=False,
        nan_detected=False,
        fallen=False,
        physical_collision=True,
    )
    for name in ("nan_detected", "fallen"):
        values = {
            "t5_completion_sim": True,
            "nan_detected": False,
            "fallen": False,
            "physical_collision": False,
        }
        values[name] = True
        assert hazard(**values)


def test_controller_rejects_state_update_after_reset_epoch_changes() -> None:
    current = _load_pure_function(CONTROLLER, "_state_update_is_current")
    common = {
        "t5_sim_time_semantics": True,
        "update_identity": ("a::episode", 4, 10),
        "active_identity": ("a::episode", 4, 11),
    }
    assert current(update_epoch=7, current_epoch=7, **common)
    assert not current(update_epoch=7, current_epoch=8, **common)
    assert not current(
        t5_sim_time_semantics=True,
        update_identity=("a::old", 3, 10),
        active_identity=("a::new", 4, 0),
        update_epoch=8,
        current_epoch=8,
    )
    assert current(
        t5_sim_time_semantics=False,
        update_identity=("legacy-old", 3, 10),
        active_identity=("legacy-new", 4, 0),
        update_epoch=7,
        current_epoch=8,
    )


def test_controller_accepts_only_safe_same_generation_bootstrap_state() -> None:
    current = _load_pure_function(CONTROLLER, "_state_update_is_current")
    common = {
        "t5_sim_time_semantics": True,
        "active_identity": ("a::628", 4, -1),
        "update_epoch": 7,
        "current_epoch": 7,
    }
    assert current(
        update_identity=("bootstrap-episode-4", 4, 0),
        state_only=True,
        **common,
    )
    assert not current(
        update_identity=("bootstrap-episode-4", 4, 0),
        state_only=False,
        **common,
    )
    assert not current(
        update_identity=("bootstrap-episode-3", 4, 0),
        state_only=True,
        **common,
    )
    assert not current(
        update_identity=("bootstrap-episode-4", 4, 1),
        state_only=True,
        **common,
    )
    assert not current(
        update_identity=("bootstrap-episode-4", 4, 0),
        state_only=True,
        t5_sim_time_semantics=True,
        active_identity=("a::628", 4, -1),
        update_epoch=6,
        current_epoch=7,
    )


def test_group_c_is_exactly_guarded_and_keeps_wall_liveness_waits() -> None:
    active = ACTIVE.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    client = T4_CLIENT.read_text(encoding="utf-8")
    recovery_adapter = T4_RECOVERY_ADAPTER.read_text(encoding="utf-8")
    sensor_bridge = T4_SENSOR_BRIDGE.read_text(encoding="utf-8")

    for source in (active, controller):
        assert 'os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in source
        assert 'os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"' in source
        assert 'os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}' in source
        assert 'not self.has_parameter("use_sim_time")' in source
        assert 'not bool(self.get_parameter("use_sim_time").value)' in source

    assert "deadline = time.monotonic() + self.cmd_wait_timeout" in active
    assert "deadline = time.monotonic() + self.server_timeout" in active
    assert "cmd_age = _semantic_age(self, self.latest_cmd_monotonic)" in active
    assert "plan_age = _semantic_age(self, self.latest_plan_monotonic)" in active

    assert "wall_now = time.monotonic()" in controller
    assert "command_age = _semantic_age(self, self.last_safe_cmd_monotonic)" in controller
    assert '"command_age_sec": command_age' in controller
    assert "safe_cmd_serial > barrier_cmd_serial" in controller
    assert "fresh = _command_is_fresh(" in controller
    assert "self.navigation_barrier_cmd_serial = self.safe_cmd_serial" in controller
    assert "self.latest_state_monotonic = semantic_now" in controller
    assert "self._state_replay_epoch += 1" in controller
    assert "update_epoch=replay_epoch" in controller

    assert "t5_completion_sim_enabled" in client
    assert "semantic_age_sec" in client
    assert "now_ns < last_ns" in client
    assert "node._last_semantic_clock_ns = now_ns" in client
    assert "now_ns=_contract_now_ns(self)" in client
    assert '_contract_now_ns(self) >= int(pending["deadline_ns"])' in client
    assert "deadline = time.monotonic() + liveness_timeout" in client
    assert "age = _semantic_age(self, self._t4_odometry_monotonic)" in client
    assert "self._t4_odometry_barrier_serial = self._t4_odometry_serial" in client
    assert "> self._t4_odometry_barrier_serial" in client

    assert "cmd_age = _semantic_age(self, self.latest_cmd_monotonic)" in recovery_adapter
    assert "cmd_age is not None" in recovery_adapter

    assert "semantic_now = t3_bridge._semantic_now(self)" in sensor_bridge
    assert "self._state_replay_epoch += 1" in sensor_bridge
    assert "self.latest_state = None" in sensor_bridge
    assert "self.latest_pointcloud_monotonic = semantic_now" in sensor_bridge
    assert "self.latest_detected_target_monotonic = semantic_now" in sensor_bridge
