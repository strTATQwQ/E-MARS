from __future__ import annotations

from pathlib import Path

from sensor_runtime.contract import PROFILES
from sensor_runtime.runtime_policy import POLICIES, policy_for_session_profile


ROOT = Path(__file__).resolve().parents[1]


def test_completion_sim_is_explicit_and_legacy_profiles_remain_strict() -> None:
    assert policy_for_session_profile("bootstrap").name == "strict_evidence"
    assert policy_for_session_profile("soak").name == "strict_evidence"
    assert policy_for_session_profile("completion_sim").name == "completion_sim"
    assert policy_for_session_profile("completion_sim_map").name == "completion_sim"
    assert PROFILES["completion_sim"].duration_sec == 60.0
    assert PROFILES["completion_sim_map"].duration_sec == 60.0
    assert POLICIES["strict_evidence"].sensor_wire_send_timeout_sec == 0.2
    assert POLICIES["completion_sim"].sensor_wire_send_timeout_sec == 5.0
    assert POLICIES["strict_evidence"].render_resync_limit == 1
    assert POLICIES["completion_sim"].render_resync_limit == 4


def test_completion_sim_keeps_only_functional_safety_invariants() -> None:
    policy = POLICIES["completion_sim"]
    assert policy.recorder_mode == "nonfatal_consumer_shadow"
    assert policy.bridge_timeout_sec == 5.0
    assert policy.transform_tolerance_sec == 2.5
    assert policy.collision_monitor_mode == "warn_only"
    assert policy.nvblox_mode == "shadow"
    assert policy.global_map_mode == "static_with_lidar_local"
    assert policy.render_stamp_deviation_tolerance_ns == 1_000_000
    assert policy.sigterm_143_is_normal_with_zero_residuals is True


def test_real_go2_is_forbidden_from_completion_sim_overlay() -> None:
    text = (ROOT / "configs/runtime/completion_sim.yaml").read_text(encoding="utf-8")
    assert "forbidden_targets: [real_go2, hardware_motion]" in text
    assert "bounded_velocity_required: true" in text
    assert "simulation_estop_required: true" in text


def test_recorder_uses_consumer_groups_and_nonfatal_shadow() -> None:
    text = (ROOT / "sensor_runtime/downstream_recorder.py").read_text(encoding="utf-8")
    for name in ("nav2", "nvblox_depth", "nvblox_lidar", "internvla"):
        assert f'"{name}"' in text
    assert "partial consumer frame retained as shadow" in text
    assert "if self.completion_shadow:" in text
    assert "rclpy.shutdown()" in text  # strict_evidence still keeps fail-closed behavior
    assert '"action": "continue_without_global_shutdown"' in (
        ROOT / "sensor_runtime/ros_sidecar.py"
    ).read_text(encoding="utf-8")
    bridge = (
        ROOT / "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py"
    ).read_text(encoding="utf-8")
    assert "completion_partial_timeout" in bridge
    assert "record_and_drop_partial_then_continue" in bridge
    assert 'self.runtime_policy.name != "completion_sim" and expired' in bridge
    assert "bridge batch incomplete after" in bridge  # strict path still fails
    assert '"target_matched": selected == target' in bridge
    sidecar = (ROOT / "sensor_runtime/ros_sidecar.py").read_text(encoding="utf-8")
    recorder = (ROOT / "sensor_runtime/downstream_recorder.py").read_text(
        encoding="utf-8"
    )
    assert "coordinated_completion_shutdown" in sidecar
    assert "coordinated_completion_shutdown" in recorder
