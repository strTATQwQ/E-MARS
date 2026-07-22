from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_t4_sensors"))

from internvla_t4_sensors.motion_observation_gate import (  # noqa: E402
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    DECISION_HOLD_MOTION,
    DECISION_HOLD_CANCEL_ACK,
    DECISION_HOLD_POST_STOP,
    DECISION_READY,
    DECISION_SAFE_STOP_COMPLETE,
    DECISION_SAFE_STOP_TIMEOUT,
    MotionGateConfig,
    MotionObservationGate,
    ODOMETRY_STAMP_ADVANCED,
    ODOMETRY_STAMP_DUPLICATE,
    ODOMETRY_STAMP_REGRESSED,
    Pose2D,
    classify_odometry_stamp,
    is_system1_queue_plan_motion,
    resolved_motion_bounds,
)


NANOSECOND = 1_000_000_000


def _gate(action: int) -> MotionObservationGate:
    gate = MotionObservationGate()
    gate.reset("a::episode", 3)
    decision = gate.arm(
        episode_id="a::episode",
        reset_generation=3,
        sequence_id=7,
        request_id="a::request-7",
        action=action,
        start_pose=Pose2D(1.0, 2.0, 0.0),
        start_sim_ns=10 * NANOSECOND,
        start_odom_stamp_ns=10 * NANOSECOND,
        start_odom_serial=10,
    )
    assert decision.kind == DECISION_HOLD_MOTION
    return gate


def test_odometry_stamp_classifier_drops_equal_but_rejects_regression() -> None:
    assert classify_odometry_stamp(100, 100) == ODOMETRY_STAMP_DUPLICATE
    assert classify_odometry_stamp(100, 99) == ODOMETRY_STAMP_REGRESSED
    assert classify_odometry_stamp(100, 101) == ODOMETRY_STAMP_ADVANCED
    assert classify_odometry_stamp(0, 100) == ODOMETRY_STAMP_ADVANCED


def _observe(
    gate: MotionObservationGate,
    *,
    sim_sec: float,
    serial: int,
    x: float = 1.0,
    y: float = 2.0,
    yaw_deg: float = 0.0,
    camera_sec: float | None = None,
    camera_serial: int | None = None,
):
    stamp_ns = int(sim_sec * NANOSECOND)
    return gate.observe(
        episode_id="a::episode",
        reset_generation=3,
        sim_stamp_ns=stamp_ns,
        odom_stamp_ns=stamp_ns,
        odom_serial=serial,
        camera_sensor_stamp_ns=int(
            (sim_sec if camera_sec is None else camera_sec) * NANOSECOND
        ),
        camera_sensor_serial=(serial if camera_serial is None else camera_serial),
        pose=Pose2D(x, y, math.radians(yaw_deg)),
    )


def _ack(
    gate: MotionObservationGate,
    *,
    sim_sec: float,
    serial: int,
    camera_serial: int,
):
    stamp_ns = int(sim_sec * NANOSECOND)
    return gate.acknowledge_stop(
        episode_id="a::episode",
        reset_generation=3,
        sim_stamp_ns=stamp_ns,
        odom_stamp_ns=stamp_ns,
        odom_serial=serial,
        camera_sensor_stamp_ns=stamp_ns,
        camera_sensor_serial=camera_serial,
    )


def test_hard_contract_values_are_frozen() -> None:
    assert MotionGateConfig() == MotionGateConfig(
        turn_command_deg=15.0,
        turn_required_deg=12.0,
        turn_timeout_sec=2.0,
        forward_command_m=0.25,
        forward_required_m=0.20,
        forward_timeout_sec=3.0,
    )


def test_left_turn_safe_stops_at_measured_twelve_degrees() -> None:
    gate = _gate(ACTION_LEFT)
    assert _observe(gate, sim_sec=11.0, serial=11, yaw_deg=11.9).kind == DECISION_HOLD_MOTION
    completed = _observe(gate, sim_sec=11.5, serial=12, yaw_deg=12.0)
    assert completed.kind == DECISION_SAFE_STOP_COMPLETE
    assert completed.requires_safe_stop is True
    assert gate.state == "awaiting_post_stop_observation"


def test_wrong_way_turn_does_not_satisfy_progress_and_times_out_in_sim_time() -> None:
    gate = _gate(ACTION_RIGHT)
    assert _observe(gate, sim_sec=11.9, serial=11, yaw_deg=20.0).kind == DECISION_HOLD_MOTION
    timed_out = _observe(gate, sim_sec=12.0, serial=12, yaw_deg=20.0)
    assert timed_out.kind == DECISION_SAFE_STOP_TIMEOUT
    assert timed_out.elapsed_sim_sec == 2.0


def test_forward_requires_projected_distance_not_lateral_drift() -> None:
    gate = _gate(ACTION_FORWARD)
    lateral = _observe(gate, sim_sec=11.0, serial=11, x=1.0, y=2.5)
    assert lateral.kind == DECISION_HOLD_MOTION
    completed = _observe(gate, sim_sec=11.5, serial=12, x=1.20, y=2.5)
    assert completed.kind == DECISION_SAFE_STOP_COMPLETE
    assert math.isclose(completed.progress, 0.20, abs_tol=1e-9)


def test_post_stop_requires_new_sim_stamp_and_new_odometry() -> None:
    gate = _gate(ACTION_LEFT)
    assert _observe(gate, sim_sec=11.0, serial=11, yaw_deg=12.0).requires_safe_stop
    before_ack = _observe(gate, sim_sec=11.1, serial=12, yaw_deg=12.0)
    assert before_ack.kind == DECISION_HOLD_CANCEL_ACK
    _ack(gate, sim_sec=11.1, serial=12, camera_serial=12)
    same_odom = _observe(gate, sim_sec=11.1, serial=11, yaw_deg=12.0)
    assert same_odom.kind == DECISION_HOLD_POST_STOP
    same_stamp = _observe(
        gate, sim_sec=11.1, serial=13, yaw_deg=12.0, camera_serial=13
    )
    assert same_stamp.kind == DECISION_HOLD_POST_STOP
    same_camera = _observe(
        gate,
        sim_sec=11.2,
        serial=13,
        yaw_deg=12.0,
        camera_sec=11.1,
        camera_serial=12,
    )
    assert same_camera.kind == DECISION_HOLD_POST_STOP
    ready = _observe(
        gate,
        sim_sec=11.2,
        serial=13,
        yaw_deg=12.0,
        camera_sec=11.2,
        camera_serial=13,
    )
    assert ready.kind == DECISION_READY
    assert ready.permits_model_step is True


def test_progress_reached_at_deadline_is_timeout_not_late_pass() -> None:
    gate = _gate(ACTION_LEFT)
    late = _observe(gate, sim_sec=12.0, serial=11, yaw_deg=15.0)
    assert late.kind == DECISION_SAFE_STOP_TIMEOUT


def test_resolved_short_forward_path_scales_required_progress() -> None:
    gate = MotionObservationGate()
    gate.reset("a::episode", 3)
    gate.arm(
        episode_id="a::episode",
        reset_generation=3,
        sequence_id=7,
        request_id="a::request-7",
        action=ACTION_FORWARD,
        start_pose=Pose2D(1.0, 2.0, 0.0),
        start_sim_ns=10 * NANOSECOND,
        start_odom_stamp_ns=10 * NANOSECOND,
        start_odom_serial=10,
        resolved_commanded_progress=0.10,
        resolved_required_progress=0.08,
    )
    completed = _observe(gate, sim_sec=11.0, serial=11, x=1.08)
    assert completed.kind == DECISION_SAFE_STOP_COMPLETE
    assert math.isclose(completed.commanded_progress, 0.10)
    assert math.isclose(completed.required_progress, 0.08)


def test_inference_interval_odometry_is_absorbed_into_command_start_baseline() -> None:
    gate = MotionObservationGate()
    gate.reset("a::episode", 3)
    # The robot moved while synchronous inference/Nav2 resolution was running.
    # Arming from the latest post-resolution sample prevents that earlier move
    # from being credited to the newly issued command.
    gate.arm(
        episode_id="a::episode",
        reset_generation=3,
        sequence_id=7,
        request_id="a::request-7",
        action=ACTION_FORWARD,
        start_pose=Pose2D(1.19, 2.0, 0.0),
        start_sim_ns=15 * NANOSECOND,
        start_odom_stamp_ns=15 * NANOSECOND,
        start_odom_serial=15,
    )
    decision = _observe(gate, sim_sec=16.0, serial=16, x=1.20)
    assert decision.kind == DECISION_HOLD_MOTION
    assert math.isclose(decision.progress, 0.01, abs_tol=1e-9)


def test_reset_discards_pending_motion_and_rejects_old_identity() -> None:
    gate = _gate(ACTION_FORWARD)
    gate.reset("a::next", 4)
    assert gate.state == "ready"
    try:
        gate.observe(
            episode_id="a::episode",
            reset_generation=3,
            sim_stamp_ns=11 * NANOSECOND,
            odom_stamp_ns=11 * NANOSECOND,
            odom_serial=11,
            pose=Pose2D(1.3, 2.0, 0.0),
        )
    except ValueError as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("old reset identity was admitted")


def test_confirmed_old_identity_stop_becomes_new_identity_sensor_barrier() -> None:
    gate = _gate(ACTION_FORWARD)
    failed = gate.fail_active_motion(
        episode_id="a::episode",
        reset_generation=3,
        sim_stamp_ns=11 * NANOSECOND,
        odom_stamp_ns=11 * NANOSECOND,
        odom_serial=11,
        reason="reset",
    )
    assert failed.requires_safe_stop is True
    _ack(gate, sim_sec=11.0, serial=11, camera_serial=11)
    old_barrier = gate.snapshot()["stop_barrier"]
    assert old_barrier["cancel_acknowledged"] is True

    gate.reset("a::next", 4)
    inherited = gate.inherit_confirmed_stop_barrier(
        episode_id="a::next",
        reset_generation=4,
        sequence_id=-1,
        request_id="post-reset:a::next:4",
        sim_stamp_ns=12 * NANOSECOND,
        odom_stamp_ns=12 * NANOSECOND,
        odom_serial=11,
        camera_sensor_stamp_ns=12 * NANOSECOND,
        camera_sensor_serial=11,
        reason="confirmed old cancellation; wait for post-reset sensors",
    )
    assert inherited.permits_model_step is False
    held = gate.observe(
        episode_id="a::next",
        reset_generation=4,
        sim_stamp_ns=12 * NANOSECOND,
        odom_stamp_ns=12 * NANOSECOND,
        odom_serial=12,
        camera_sensor_stamp_ns=12 * NANOSECOND,
        camera_sensor_serial=12,
        pose=Pose2D(1.0, 2.0, 0.0),
    )
    assert held.kind == DECISION_HOLD_POST_STOP
    ready = gate.observe(
        episode_id="a::next",
        reset_generation=4,
        sim_stamp_ns=13 * NANOSECOND,
        odom_stamp_ns=13 * NANOSECOND,
        odom_serial=13,
        camera_sensor_stamp_ns=13 * NANOSECOND,
        camera_sensor_serial=13,
        pose=Pose2D(1.0, 2.0, 0.0),
    )
    assert ready.kind == DECISION_READY


def test_active_adapter_scopes_stop_ack_and_quarter_meter_step_to_exact_t5() -> None:
    source = (
        ROOT
        / "internvla_nav2_adapter"
        / "internvla_nav2_adapter"
        / "active_node.py"
    ).read_text(encoding="utf-8")
    assert '"/internvla/stop"' in source
    assert "self._on_safe_stop" in source
    assert '"/internvla/t5_nav2_stop_request"' in source
    assert '"/internvla/t5_nav2_stop_ack"' in source
    assert "self._cancel_active(wait=True)" in source
    assert "if self._t5_sim_time_semantics" in source
    assert "else 0.35" in source
    assert "system2_local_poses(" in source
    primitives = (
        ROOT
        / "internvla_nav2_adapter"
        / "internvla_nav2_adapter"
        / "motion_primitives.py"
    ).read_text(encoding="utf-8")
    assert "forward_step_m * value" in primitives
    assert "if t5_completion_sim:" in primitives


def test_client_serializes_inference_and_binds_reset_epoch_camera_and_odom() -> None:
    source = (
        ROOT
        / "internvla_t4_sensors"
        / "internvla_t4_sensors"
        / "client_node.py"
    ).read_text(encoding="utf-8")
    assert "with self.operation_lock:" in source
    assert "callback_epoch != self._motion_gate_epoch" in source
    assert "odometry header stamp did not strictly advance" in source
    odometry_callback = source[
        source.index("    def _on_t4_odometry(") :
        source.index("    def _fail_active_motion_from_callback(")
    ]
    assert "stamp_order == ODOMETRY_STAMP_DUPLICATE" in odometry_callback
    assert odometry_callback.index("if duplicate_sample:\n            return") < (
        odometry_callback.index("if invalid_reason is not None:")
    )
    assert "stamp_order == ODOMETRY_STAMP_REGRESSED" in odometry_callback
    assert "camera sensor source sim stamp did not strictly advance" in source
    assert 'kwargs.pop("camera_sensor_stamp_ns", None)' in source
    assert 'camera_source != "x86_isaac_pano_camera_0"' in source
    assert "stop_requested" in source
    assert "cancel_ack" in source
    resolution_index = source.index("result = super().step_arrays(")
    baseline_index = source.index(") = self._motion_start_baseline()")
    assert baseline_index > resolution_index
    assert "start_pose=start_pose" in source
    assert "start_sim_ns=start_sim_ns" in source
    assert "start_odom_stamp_ns=start_odom_stamp_ns" in source
    assert "start_odom_serial=start_odom_serial" in source
    reset_source = source[source.index("    def reset(") : source.index("    def _reset_motion_gate(")]
    assert reset_source.index("_confirm_stop_before_identity_change") < reset_source.index(
        "super().reset"
    )
    assert reset_source.index("super().reset") < reset_source.index(
        "_install_post_reset_sensor_barrier"
    )
    assert "except BaseException as exc:" in source
    assert "_stop_resolved_goal_after_failure(result, message)" in source


@pytest.mark.parametrize("action", [ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT])
def test_valid_system1_queue_plan_uses_preregistered_motion_gate(action: int) -> None:
    assert is_system1_queue_plan_motion(
        action_source=3,
        trajectory_valid=False,
        nav2_goal_sent=True,
        nav2_plan_valid=True,
        stop=False,
        model_action=action,
    )


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"action_source": 2},
        {"trajectory_valid": True},
        {"nav2_goal_sent": False},
        {"nav2_plan_valid": False},
        {"stop": True},
        {"model_action": 0},
    ],
)
def test_invalid_or_new_system1_response_cannot_use_queue_gate(overrides: dict) -> None:
    values = {
        "action_source": 3,
        "trajectory_valid": False,
        "nav2_goal_sent": True,
        "nav2_plan_valid": True,
        "stop": False,
        "model_action": ACTION_FORWARD,
    }
    values.update(overrides)
    assert not is_system1_queue_plan_motion(**values)


def test_queue_forward_uses_frozen_bounds_without_local_path() -> None:
    assert resolved_motion_bounds(
        ACTION_FORWARD,
        MotionGateConfig(),
        [],
        use_preregistered_bounds=True,
    ) == (0.25, 0.20)


def test_system1_new_forward_still_requires_resolved_local_path() -> None:
    with pytest.raises(ValueError, match="too short"):
        resolved_motion_bounds(
            ACTION_FORWARD,
            MotionGateConfig(),
            [],
            use_preregistered_bounds=False,
        )


def test_client_stop_ack_can_run_while_odometry_waits_and_future_slack_is_bounded() -> None:
    source = (
        ROOT
        / "internvla_t4_sensors"
        / "internvla_t4_sensors"
        / "client_node.py"
    ).read_text(encoding="utf-8")
    assert "MutuallyExclusiveCallbackGroup" in source
    assert "self._motion_stop_ack_callback_group = MutuallyExclusiveCallbackGroup()" in source
    assert "self._t4_odometry_callback_group = MutuallyExclusiveCallbackGroup()" in source
    ack_subscription = source[
        source.index('"/internvla/t5_nav2_stop_ack"') :
        source.index('self.create_subscription(\n            Odometry')
    ]
    assert "callback_group=self._motion_stop_ack_callback_group" in ack_subscription
    odometry_subscription = source[
        source.index('self.create_subscription(\n            Odometry') :
        source.index('self.create_subscription(\n            Int32')
    ]
    assert "callback_group=self._t4_odometry_callback_group" in odometry_subscription
    assert "self._t4_odometry_callback_group = self._motion_stop_ack_callback_group" not in source
    assert 'self.declare_parameter("sensor_future_tolerance_sec", 0.0)' in source
    assert "0.0 <= self.sensor_future_tolerance <= 0.55" in source
    assert "sensor future tolerance is restricted to T5 completion_sim" in source
    assert source.count("> self._sensor_future_tolerance_ns") == 3
    assert "gate_sim_stamp_ns = max(sim_stamp_ns, odom_stamp_ns)" in source
    assert "sim_stamp_ns = max(sim_stamp_ns, odom_stamp_ns, int(camera_stamp_ns))" in source


def test_continuous_nav2_goal_arms_gate_from_model_action() -> None:
    source = (
        ROOT
        / "internvla_t4_sensors"
        / "internvla_t4_sensors"
        / "client_node.py"
    ).read_text(encoding="utf-8")
    fallback = source[
        source.index("resolved_action = int(") : source.index("should_arm = bool(")
    ]
    assert "resolved_action == ACTION_STAND_STILL" in fallback
    assert "and resolved_goal_sent" in fallback
    assert 'bool(result.get("nav2_plan_valid", False))' in fallback
    assert "resolved_action = model_action" in fallback

    bounds_call = source[
        source.index("self._resolved_motion_bounds(") : source.index(
            ")\n                    )", source.index("self._resolved_motion_bounds(")
        )
    ]
    assert "use_preregistered_bounds=system1_queue_plan_motion" in bounds_call


def test_x86_camera_metadata_is_captured_at_sensor_source_not_ipc_receive() -> None:
    runtime_source = (ROOT / "scripts" / "internnav_go2_runtime.py").read_text(
        encoding="utf-8"
    )
    facade_source = (
        ROOT / "scripts" / "internvla_ipc_agent_client.py"
    ).read_text(encoding="utf-8")
    assert "get_rgb_depth_with_source_metadata" in runtime_source
    assert 'observation["camera_sensor_metadata"]' in runtime_source
    assert '"sim_stamp_ns": int(source_stamp_ns)' in runtime_source
    assert "_T5_SIM_CLOCK_NS" in runtime_source
    assert "_t5_camera_sensor_sequence" not in facade_source
    assert 'item.get("camera_sensor_metadata")' in facade_source
