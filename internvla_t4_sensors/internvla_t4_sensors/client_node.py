"""Typed InternVLA client that discards evaluator truth and uses navigation odometry."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from internvla_ros2.client_node import (
    ClientFailure,
    InternVLAClientNode,
    LocalIPCServer,
    TCPIPCServer,
)
from internvla_ros2.model_health import validate_uninitialized_model_health
from internvla_ros2.protocol import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAND_STILL,
    STATUS_OK,
    STATUS_INTERNAL_ERROR,
    STATUS_INVALID_REQUEST,
    STATUS_STALE,
    STATUS_TIMEOUT,
)
from internvla_ros2.recovery_contract import (
    OP_REQUEST_REPLAN,
    STATUS_CONFLICT,
    STATUS_INTERNAL,
    RecoveryContractError,
    goal_identity,
    initialize_response,
    reject_response,
    response_snapshot,
    restore_response,
    semantic_age_sec,
    system2_replan_policy,
    system2_primitive_signature,
    t5_completion_sim_enabled,
    trajectory_signature,
    validate_request,
)
from internvla_ros2_msgs.srv import RecoveryControl, ResolveCommand
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Int32, String

from .motion_observation_gate import (
    DECISION_SAFE_STOP_COMPLETE,
    DECISION_SAFE_STOP_STALE,
    DECISION_SAFE_STOP_TIMEOUT,
    GateDecision,
    MotionObservationGate,
    ODOMETRY_STAMP_DUPLICATE,
    ODOMETRY_STAMP_REGRESSED,
    Pose2D,
    classify_odometry_stamp,
    is_system1_queue_plan_motion,
    resolved_motion_bounds,
)
from .step3_arrival_gate import (
    CONTINUE_NAVIGATION,
    REQUEST_CONFIRMATION,
    TERMINATE_ASSISTED,
    arrival_transition,
    completed_motion_action,
)


def _t5_semantic_now_ns(node: Any) -> int | None:
    now_ns = int(node.get_clock().now().nanoseconds)
    if now_ns <= 0:
        return None
    clock_lock = getattr(node, "_semantic_clock_lock", None)
    if clock_lock is None:
        last_ns = int(getattr(node, "_last_semantic_clock_ns", 0))
        if now_ns < last_ns:
            return None
        node._last_semantic_clock_ns = now_ns
    else:
        with clock_lock:
            last_ns = int(getattr(node, "_last_semantic_clock_ns", 0))
            if now_ns < last_ns:
                return None
            node._last_semantic_clock_ns = now_ns
    return now_ns


def _contract_now_ns(node: Any) -> int:
    if not getattr(node, "_t5_sim_time_semantics", False):
        return time.time_ns()
    now_ns = _t5_semantic_now_ns(node)
    if now_ns is None:
        raise RuntimeError("T5 simulation clock is unavailable or regressed")
    return now_ns


def _semantic_now(node: Any) -> float:
    if not getattr(node, "_t5_sim_time_semantics", False):
        return time.monotonic()
    now_ns = _t5_semantic_now_ns(node)
    return now_ns / 1_000_000_000 if now_ns is not None else 0.0


def _semantic_age(node: Any, stamp: float) -> float | None:
    now = _semantic_now(node)
    if getattr(node, "_t5_sim_time_semantics", False):
        return semantic_age_sec(
            int(now * 1_000_000_000), int(stamp * 1_000_000_000)
        )
    return now - stamp


def _stamp_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _pose2d(message: Odometry) -> Pose2D:
    pose = message.pose.pose
    rotation = pose.orientation
    yaw = math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )
    return Pose2D(float(pose.position.x), float(pose.position.y), float(yaw))


class T4OdometryClientNode(InternVLAClientNode):
    SYSTEM1_QUEUE_SOURCE = 3
    SYSTEM1_QUEUE_SAFE_HOLD_LIMIT = 32
    SYSTEM1_QUEUE_TARGET_MISSING = (
        "System 1 queue has no identity-bound absolute target; "
        "fresh trajectory required"
    )
    RECOVERY_BOUNDED_FALLBACK = (
        "recovery latch bounded fallback requires fresh trajectory"
    )
    STEP3_ARRIVAL_ACTION_INTERVAL = 8
    STEP3_ARRIVAL_MAX_CHECKS = 24
    STEP3_ARRIVAL_REQUIRED_CONFIRMATIONS = 2
    STEP3_ARRIVAL_MINIMUM_CONFIDENCE = 0.80

    def __init__(self) -> None:
        super().__init__()
        self._t5_sim_time_semantics = t5_completion_sim_enabled()
        if self._t5_sim_time_semantics and (
            not self.has_parameter("use_sim_time")
            or not bool(self.get_parameter("use_sim_time").value)
        ):
            raise RuntimeError("T5 completion_sim requires use_sim_time=true")
        self._semantic_clock_lock = threading.Lock()
        self._last_semantic_clock_ns = 0
        self.declare_parameter("navigation_odometry_topic", "/odom")
        self.declare_parameter("navigation_odometry_timeout_sec", 0.50)
        self.declare_parameter("allow_nearest_navigation_odometry", False)
        self.declare_parameter("sensor_future_tolerance_sec", 0.0)
        self.navigation_odometry_topic = str(
            self.get_parameter("navigation_odometry_topic").value
        )
        self.navigation_odometry_timeout = float(
            self.get_parameter("navigation_odometry_timeout_sec").value
        )
        if not 0.1 <= self.navigation_odometry_timeout <= 2.0:
            raise RuntimeError("invalid navigation odometry timeout")
        self.allow_nearest_navigation_odometry = bool(
            self.get_parameter("allow_nearest_navigation_odometry").value
        )
        self.sensor_future_tolerance = float(
            self.get_parameter("sensor_future_tolerance_sec").value
        )
        if not 0.0 <= self.sensor_future_tolerance <= 0.55:
            raise RuntimeError("invalid sensor future tolerance")
        if self.sensor_future_tolerance > 0.0 and not self._t5_sim_time_semantics:
            raise RuntimeError(
                "sensor future tolerance is restricted to T5 completion_sim"
            )
        self._sensor_future_tolerance_ns = int(
            self.sensor_future_tolerance * 1_000_000_000
        )
        if self.allow_nearest_navigation_odometry and not (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        ):
            raise RuntimeError(
                "nearest navigation odometry is restricted to Isaac completion_sim"
            )
        self.allow_system2_recovery_replan = (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        )
        self.system2_replan_policy = system2_replan_policy()
        self._t4_pose_lock = threading.Condition()
        self._t4_odometry: Odometry | None = None
        self._t4_odometry_monotonic = 0.0
        self._t4_odometry_stamp_ns = 0
        self._t4_odometry_serial = 0
        self._t4_odometry_barrier_serial = 0
        self._t4_map_reset_generation = -1
        self._replan_requested = False
        self._replan_request_index = 0
        self._pending_typed_replan: dict[str, Any] | None = None
        self._last_typed_replan_sequence = -1
        self._system1_queue_hold_identity: tuple[str, int] | None = None
        self._system1_queue_hold_count = 0
        self._typed_replan_results: dict[
            str, tuple[tuple[object, ...], dict[str, object]]
        ] = {}
        self._motion_gate_enabled = self._t5_sim_time_semantics
        self._motion_gate_lock = threading.Lock()
        self._motion_gate_record_lock = threading.Lock()
        self._motion_gate = MotionObservationGate()
        self._motion_gate_epoch = 0
        self._motion_gate_fault: str | None = None
        self._reset_sim_barrier_ns = 0
        self._camera_sensor_serial = 0
        self._camera_sensor_stamp_ns = 0
        self._motion_stop_counter = 0
        self._stop_ack_condition = threading.Condition()
        self._stop_ack_by_token: dict[str, dict[str, Any]] = {}
        self._step3_timeout_enabled = bool(
            os.environ.get("INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR", "0") == "1"
            and os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
            and os.environ.get("INTERNNAV_T5_LANE", "") == "a"
            and os.environ.get("INTERNNAV_T5_LANE_NAMESPACE", "") == "/t5/lane_a"
            and self.system2_replan_policy == "observation_bound"
        )
        if os.environ.get("INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR", "0") not in {
            "0",
            "1",
        }:
            raise RuntimeError("invalid Step3 timeout-advisor flag")
        if (
            os.environ.get("INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR", "0") == "1"
            and not self._step3_timeout_enabled
        ):
            raise RuntimeError(
                "Step3 timeout advisor requires Lane-A Isaac completion_sim "
                "with observation_bound System2 replans"
            )
        self._step3_timeout_condition = threading.Condition()
        self._step3_timeout_pending: dict[str, Any] | None = None
        self._step3_timeout_advice: dict[str, Any] | None = None
        self._step3_timeout_override: dict[str, Any] | None = None
        self._step3_timeout_interventions = 0
        self._step3_timeout_max_interventions = int(
            os.environ.get("INTERNVLA_T5_STEP3_TIMEOUT_MAX_INTERVENTIONS", "6")
        )
        if not 1 <= self._step3_timeout_max_interventions <= 6:
            raise RuntimeError("Step3 timeout interventions must be in [1, 6]")
        self._step3_arrival_completed_actions = 0
        self._step3_arrival_checks = 0
        self._last_navigation_instruction = ""
        self.motion_gate_audit_path = (
            self.result_dir / "motion_observation_gate_records.jsonl"
            if self.result_dir is not None and self._motion_gate_enabled
            else None
        )
        if (
            self.motion_gate_audit_path is not None
            and self.motion_gate_audit_path.exists()
        ):
            raise FileExistsError(self.motion_gate_audit_path)
        self.motion_stop_request_publisher = None
        self.step3_timeout_context_publisher = None
        self._motion_stop_ack_callback_group = None
        self._t4_odometry_callback_group = None
        if self._motion_gate_enabled:
            # An odometry callback can synchronously request a fail-closed stop.
            # Keep the acknowledgement in a distinct callback group so the
            # bounded wait cannot block its own acknowledgement callback.
            self._motion_stop_ack_callback_group = MutuallyExclusiveCallbackGroup()
            # The odometry callback may wait for ``operation_lock`` while the
            # IPC step holding that lock waits for Step/reset/resolve futures.
            # A separate group prevents that wait from occupying the default
            # group needed to settle those client futures.
            self._t4_odometry_callback_group = MutuallyExclusiveCallbackGroup()
            self.motion_stop_request_publisher = self.create_publisher(
                String, "/internvla/t5_nav2_stop_request", 20
            )
            self.create_subscription(
                String,
                "/internvla/t5_nav2_stop_ack",
                self._on_motion_stop_ack,
                20,
                callback_group=self._motion_stop_ack_callback_group,
            )
            if self._step3_timeout_enabled:
                self.step3_timeout_context_publisher = self.create_publisher(
                    String, "/internvla/t5_step3_timeout_context", 10
                )
                self.create_subscription(
                    String,
                    "/internvla/t5_step3_timeout_advice",
                    self._on_step3_timeout_advice,
                    10,
                    callback_group=self._motion_stop_ack_callback_group,
                )
        self.create_subscription(
            Odometry,
            self.navigation_odometry_topic,
            self._on_t4_odometry,
            20,
            callback_group=self._t4_odometry_callback_group,
        )
        self.create_subscription(
            Int32,
            "/internvla_t4/map_reset_generation",
            self._on_t4_reset_generation,
            10,
        )
        self.episode_prime_publisher = self.create_publisher(
            String, "/internvla_t4/episode_prime", 10
        )
        self.create_service(
            RecoveryControl,
            "/internvla/t4_recovery_client",
            self._on_typed_replan_request,
        )
        self.create_subscription(
            Bool,
            "/internvla_t4/request_new_trajectory",
            self._on_replan_request,
            10,
        )
        self.pose_audit_path = (
            self.result_dir / "client_pose_source_records.jsonl"
            if self.result_dir is not None
            else None
        )
        if self.pose_audit_path is not None and self.pose_audit_path.exists():
            raise FileExistsError(self.pose_audit_path)
        self.replan_audit_path = (
            self.result_dir / "replan_request_records.jsonl"
            if self.result_dir is not None
            else None
        )
        if self.replan_audit_path is not None and self.replan_audit_path.exists():
            raise FileExistsError(self.replan_audit_path)

    def _on_replan_request(self, message: Bool) -> None:
        if not bool(message.data):
            return
        with self._t4_pose_lock:
            self._replan_requested = True
            self._replan_request_index += 1
            index = self._replan_request_index
        self._record_replan("requested", index, None)

    def _on_motion_stop_ack(self, message: String) -> None:
        """Accept only a bounded, identity-bound acknowledgement."""

        try:
            value = json.loads(str(message.data))
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise ValueError("invalid stop-ack schema")
            token = str(value.get("token", ""))
            if not token or len(token) > 256:
                raise ValueError("invalid stop-ack token")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"discarding malformed Nav2 stop ack: {exc}")
            return
        with self._stop_ack_condition:
            self._stop_ack_by_token[token] = value
            if len(self._stop_ack_by_token) > 32:
                self._stop_ack_by_token.pop(next(iter(self._stop_ack_by_token)))
            self._stop_ack_condition.notify_all()

    def _on_step3_timeout_advice(self, message: String) -> None:
        """Accept one identity-bound, non-control Step3 result."""

        if not getattr(self, "_step3_timeout_enabled", False):
            return
        try:
            value = json.loads(str(message.data))
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise ValueError("invalid advice schema")
            confidence = float(value.get("confidence", 0.0))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("invalid advice confidence")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"discarding malformed Step3 advice: {exc}")
            return
        with self._step3_timeout_condition:
            pending = self._step3_timeout_pending
            if pending is None:
                return
            pending_kind = str(pending.get("kind", ""))
            status = str(value.get("status", ""))
            if pending_kind == "motion_timeout_after_confirmed_safe_stop":
                allowed_statuses = {"ADVISE", "FALLBACK"}
            elif pending_kind == "arrival_check_after_completed_motion":
                allowed_statuses = {"ARRIVED", "NOT_ARRIVED", "FALLBACK"}
            else:
                self.get_logger().warning("discarding advice for unknown Step3 context")
                return
            if status not in allowed_statuses:
                self.get_logger().warning("discarding Step3 advice with invalid status")
                return
            observed_identity = (
                str(value.get("episode_id", "")),
                int(value.get("reset_generation", -1)),
                int(value.get("trigger_sequence_id", -1)),
                str(value.get("trigger_request_id", "")),
                int(value.get("advisor_round", -1)),
            )
            expected_identity = (
                pending["episode_id"],
                pending["reset_generation"],
                pending["trigger_sequence_id"],
                pending["trigger_request_id"],
                pending["advisor_round"],
            )
            if observed_identity != expected_identity:
                self.get_logger().warning("discarding stale/cross-reset Step3 advice")
                return
            if pending_kind == "motion_timeout_after_confirmed_safe_stop" and status == "ADVISE":
                action = value.get("advised_action")
                if (
                    isinstance(action, bool)
                    or not isinstance(action, int)
                    or action not in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}
                    or action == pending["excluded_action"]
                    or confidence < 0.35
                ):
                    self.get_logger().warning(
                        "discarding illegal or low-confidence Step3 advice"
                    )
                    return
            if pending_kind == "arrival_check_after_completed_motion":
                snapshot_stamp_ns = value.get("snapshot_sim_stamp_ns")
                if status != "FALLBACK" and (
                    isinstance(snapshot_stamp_ns, bool)
                    or not isinstance(snapshot_stamp_ns, int)
                    or snapshot_stamp_ns
                    <= int(pending.get("minimum_snapshot_sim_stamp_ns", 0))
                    or snapshot_stamp_ns
                    < int(pending.get("camera_sensor_stamp_ns", 0))
                    or int(value.get("camera_count", 0)) != 4
                    or (
                        status == "ARRIVED"
                        and confidence < self.STEP3_ARRIVAL_MINIMUM_CONFIDENCE
                    )
                ):
                    self.get_logger().warning(
                        "discarding stale, incomplete, or low-confidence "
                        "Step3 arrival advice"
                    )
                    return
            self._step3_timeout_advice = value
            self._step3_timeout_condition.notify_all()

    def _publish_step3_timeout_context(
        self,
        *,
        token: str,
        pending: dict[str, Any],
    ) -> None:
        if (
            not self._step3_timeout_enabled
            or self.step3_timeout_context_publisher is None
            or self._step3_timeout_interventions
            >= self._step3_timeout_max_interventions
        ):
            return
        action = int(pending.get("action", ACTION_STAND_STILL))
        if action not in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}:
            return
        value = {
            "schema_version": 1,
            "kind": "motion_timeout_after_confirmed_safe_stop",
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "trigger_sequence_id": int(
                pending.get("sequence_id", self.last_committed_sequence)
            ),
            "trigger_request_id": str(pending.get("request_id", "")),
            "expected_sequence_id": int(self.next_sequence_id),
            "excluded_action": action,
            "instruction": self._last_navigation_instruction,
            "stop_token": token,
            "wall_deadline_monotonic_ns": time.monotonic_ns()
            + 12_000_000_000,
            "camera_order": ["front_left", "front", "front_right", "rear"],
            "advisor_round": 0,
        }
        with self._step3_timeout_condition:
            self._step3_timeout_pending = value
            self._step3_timeout_advice = None
        message = String()
        message.data = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.step3_timeout_context_publisher.publish(message)
        self._record_motion_gate_event("step3_timeout_requested", token, value)

    def _publish_step3_arrival_context(
        self,
        *,
        token: str,
        pending: dict[str, Any],
        advisor_round: int,
        minimum_snapshot_sim_stamp_ns: int = 0,
    ) -> bool:
        """Request one non-control arrival classification while motion is stopped."""

        if (
            not self._step3_timeout_enabled
            or self.step3_timeout_context_publisher is None
            or self._step3_arrival_checks >= self.STEP3_ARRIVAL_MAX_CHECKS
            or advisor_round not in {1, 2}
        ):
            return False
        action = completed_motion_action(pending)
        if action is None:
            return False
        with self._t4_pose_lock:
            camera_stamp_ns = int(self._camera_sensor_stamp_ns)
        value = {
            "schema_version": 1,
            "kind": "arrival_check_after_completed_motion",
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "trigger_sequence_id": int(
                pending.get("sequence_id", self.last_committed_sequence)
            ),
            "trigger_request_id": str(pending.get("request_id", "")),
            "expected_sequence_id": int(self.next_sequence_id),
            "completed_action": action,
            "instruction": self._last_navigation_instruction,
            "stop_token": token,
            "wall_deadline_monotonic_ns": time.monotonic_ns()
            + 12_000_000_000,
            "camera_order": ["front_left", "front", "front_right", "rear"],
            "camera_sensor_stamp_ns": camera_stamp_ns,
            "minimum_snapshot_sim_stamp_ns": int(minimum_snapshot_sim_stamp_ns),
            "advisor_round": advisor_round,
            "required_confirmations": self.STEP3_ARRIVAL_REQUIRED_CONFIRMATIONS,
        }
        with self._step3_timeout_condition:
            self._step3_timeout_pending = value
            self._step3_timeout_advice = None
        self._step3_arrival_checks += 1
        message = String()
        message.data = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.step3_timeout_context_publisher.publish(message)
        self._record_motion_gate_event("step3_arrival_requested", token, value)
        return True

    def _maybe_publish_step3_arrival_context(
        self, *, token: str, pending: dict[str, Any]
    ) -> None:
        if not self._step3_timeout_enabled:
            return
        self._step3_arrival_completed_actions += 1
        if (
            self._step3_arrival_completed_actions
            % self.STEP3_ARRIVAL_ACTION_INTERVAL
            != 0
        ):
            return
        self._publish_step3_arrival_context(
            token=token,
            pending=pending,
            advisor_round=1,
        )

    def _request_motion_stop_and_wait(self, decision: GateDecision) -> GateDecision:
        """Publish safe-stop, prove Nav2 cancellation, then arm sensor baselines.

        This method runs under ``operation_lock``.  The acknowledgement callback
        uses a separate condition so no new inference can interleave while the
        executor waits for the adapter's bounded cancellation response.
        """

        if not self._motion_gate_enabled or self.motion_stop_request_publisher is None:
            raise ClientFailure(STATUS_INTERNAL_ERROR, "T5 stop handshake is unavailable")
        self._motion_stop_counter += 1
        token = (
            f"{self.episode_id}:{self.reset_generation}:"
            f"{self._motion_stop_counter}"
        )
        with self._motion_gate_lock:
            snapshot = self._motion_gate.snapshot()
        pending = snapshot.get("pending") or snapshot.get("stop_barrier") or {}
        request_value = {
            "schema_version": 1,
            "token": token,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sequence_id": int(pending.get("sequence_id", self.last_committed_sequence)),
            "request_id": str(pending.get("request_id", "")),
            "reason": decision.reason,
        }
        stop_status = (
            STATUS_TIMEOUT
            if decision.kind == DECISION_SAFE_STOP_TIMEOUT
            else STATUS_STALE
            if decision.kind == DECISION_SAFE_STOP_STALE
            else STATUS_OK
        )
        self._safe_stop(stop_status, decision.reason)
        request_message = String()
        request_message.data = json.dumps(
            request_value, sort_keys=True, separators=(",", ":")
        )
        self.motion_stop_request_publisher.publish(request_message)
        audit_errors: list[str] = []
        try:
            self._record_motion_gate_event("stop_requested", token, request_value)
        except BaseException as exc:
            audit_errors.append(f"stop_requested audit failed: {exc}")

        deadline = time.monotonic() + 2.0
        with self._stop_ack_condition:
            while token not in self._stop_ack_by_token:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._motion_gate_fault = "Nav2 cancellation acknowledgement timed out"
                    self._record_motion_gate_event(
                        "cancel_ack_failed", token, {"reason": self._motion_gate_fault}
                    )
                    raise ClientFailure(STATUS_TIMEOUT, self._motion_gate_fault)
                self._stop_ack_condition.wait(remaining)
            ack = self._stop_ack_by_token.pop(token)
        expected_identity = (
            self.episode_id,
            self.reset_generation,
            int(request_value["sequence_id"]),
            str(request_value["request_id"]),
        )
        observed_identity = (
            str(ack.get("episode_id", "")),
            int(ack.get("reset_generation", -1)),
            int(ack.get("sequence_id", -2)),
            str(ack.get("request_id", "")),
        )
        if ack.get("status") != "ok" or observed_identity != expected_identity:
            self._motion_gate_fault = (
                "Nav2 cancellation was not confirmed for the active identity"
            )
            self._record_motion_gate_event(
                "cancel_ack_failed", token, {"ack": ack, "reason": self._motion_gate_fault}
            )
            raise ClientFailure(STATUS_STALE, self._motion_gate_fault)
        try:
            self._record_motion_gate_event("cancel_ack", token, ack)
        except BaseException as exc:
            audit_errors.append(f"cancel_ack audit failed: {exc}")
        sim_stamp_ns = _t5_semantic_now_ns(self)
        if sim_stamp_ns is None:
            self._motion_gate_fault = "simulation clock unavailable at cancel ack"
            raise ClientFailure(STATUS_STALE, self._motion_gate_fault)
        with self._t4_pose_lock:
            odom_stamp_ns = self._t4_odometry_stamp_ns
            odom_serial = self._t4_odometry_serial
            camera_stamp_ns = self._camera_sensor_stamp_ns
            camera_serial = self._camera_sensor_serial
        with self._motion_gate_lock:
            ack_decision = self._motion_gate.acknowledge_stop(
                episode_id=self.episode_id,
                reset_generation=self.reset_generation,
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                camera_sensor_stamp_ns=camera_stamp_ns,
                camera_sensor_serial=camera_serial,
            )
            ack_snapshot = self._motion_gate.snapshot()
        self._motion_gate_fault = None
        try:
            self._record_motion_gate(
                ack_decision,
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                snapshot=ack_snapshot,
            )
        except BaseException as exc:
            audit_errors.append(f"cancel barrier audit failed: {exc}")
        if decision.kind == DECISION_SAFE_STOP_TIMEOUT:
            self._publish_step3_timeout_context(token=token, pending=pending)
        elif decision.kind == DECISION_SAFE_STOP_COMPLETE:
            self._maybe_publish_step3_arrival_context(token=token, pending=pending)
        if audit_errors:
            self._motion_gate_fault = "; ".join(audit_errors)
            self.get_logger().error(self._motion_gate_fault)
            raise ClientFailure(STATUS_INTERNAL_ERROR, self._motion_gate_fault)
        return ack_decision

    def _record_motion_gate_event(
        self, event: str, token: str, payload: dict[str, Any]
    ) -> None:
        if self.motion_gate_audit_path is None:
            return
        record = {
            "schema_version": 1,
            "event": str(event),
            "token": str(token),
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "payload": payload,
            "wall_time_unix": time.time(),
        }
        with self._motion_gate_record_lock:
            with self.motion_gate_audit_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_replan(
        self,
        event: str,
        request_index: int,
        sequence_id: int | None,
        **extra: object,
    ) -> None:
        if self.replan_audit_path is None:
            return
        record = {
            "schema_version": 1,
            "event": event,
            "request_index": request_index,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sequence_id": sequence_id,
            **extra,
            "wall_time_unix": time.time(),
        }
        with self.replan_audit_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _on_typed_replan_request(
        self,
        request: RecoveryControl.Request,
        response: RecoveryControl.Response,
    ) -> RecoveryControl.Response:
        initialize_response(response, request)
        try:
            with self._t4_pose_lock:
                identity = validate_request(
                    request,
                    expected_operation=OP_REQUEST_REPLAN,
                    active_episode_id=self.episode_id,
                    active_reset_generation=self.reset_generation,
                    active_goal_id=goal_identity(
                        self.episode_id,
                        self.reset_generation,
                        self.last_committed_sequence,
                    ),
                    now_ns=_contract_now_ns(self),
                )
                identity_key: tuple[object, ...] = (
                    identity.episode_id,
                    identity.reset_generation,
                    identity.goal_id,
                    identity.recovery_id,
                    identity.operation,
                )
                previous = self._typed_replan_results.get(identity.operation_id)
                if previous is not None:
                    if previous[0] != identity_key:
                        response.status_code = STATUS_CONFLICT
                        response.status_message = "operation_id was already consumed"
                        return response
                    restore_response(response, previous[1])
                    return response
                if self._pending_typed_replan is not None:
                    response.status_code = STATUS_CONFLICT
                    response.status_message = "another recovery replan is pending"
                    return response
                self._replan_request_index += 1
                index = self._replan_request_index
                self._pending_typed_replan = {
                    "request_index": index,
                    "episode_id": identity.episode_id,
                    "reset_generation": identity.reset_generation,
                    "goal_id": identity.goal_id,
                    "recovery_id": identity.recovery_id,
                    "operation_id": identity.operation_id,
                    "deadline_ns": identity.deadline_ns,
                    "active_sequence_id": int(self.last_committed_sequence),
                    "excluded_absolute": set(request.excluded_absolute_sha256),
                    "excluded_shape": set(request.excluded_shape_sha256),
                    "rejection_count": 0,
                }
                response.success = True
                response.status_code = 0
                response.status_message = "fresh trajectory request registered"
                response.motion_disabled = False
                self._record_replan(
                    "requested",
                    index,
                    None,
                    recovery_id=identity.recovery_id,
                    operation_id=identity.operation_id,
                    excluded_absolute_sha256=sorted(
                        self._pending_typed_replan["excluded_absolute"]
                    ),
                    excluded_shape_sha256=sorted(
                        self._pending_typed_replan["excluded_shape"]
                    ),
                )
                if len(self._typed_replan_results) >= 256:
                    self._typed_replan_results.pop(next(iter(self._typed_replan_results)))
                self._typed_replan_results[identity.operation_id] = (
                    identity_key,
                    response_snapshot(response),
                )
        except RecoveryContractError as exc:
            reject_response(response, exc)
        except BaseException as exc:
            response.success = False
            response.status_code = STATUS_INTERNAL
            response.status_message = f"replan registration failed: {exc!r}"[:512]
        return response

    @staticmethod
    def _stand_resolution(
        command: Any, status_message: str
    ) -> ResolveCommand.Response:
        response = ResolveCommand.Response()
        response.status_code = 0
        response.status_message = status_message
        response.episode_id = str(command.episode_id)
        response.reset_generation = int(command.reset_generation)
        response.sequence_id = int(command.sequence_id)
        response.request_id = str(command.request_id)
        response.discrete_action = -1
        response.stop = False
        response.nav2_goal_sent = False
        response.nav2_plan_valid = False
        response.resolution_latency_sec = 0.0
        return response

    def _resolve_nav2_with_terminal_fallback(
        self, command: Any
    ) -> ResolveCommand.Response:
        """Keep a consumed System1 queue miss as a non-terminal safe hold.

        The model owns a finite queue and decrements it before each command is
        sent.  When a missed motion deadline has correctly discarded the
        frozen absolute target, allowing those remaining queue identities to
        drain requests a fresh model trajectory without reviving the timed-out
        target.  Only the exact completion_sim failure is downgraded.
        """

        try:
            response = super()._resolve_nav2(command)
        except ClientFailure as exc:
            queue_target_missing = bool(
                self._t5_sim_time_semantics
                and int(exc.status_code) == STATUS_INVALID_REQUEST
                and str(exc) == self.SYSTEM1_QUEUE_TARGET_MISSING
                and int(command.action_source) == self.SYSTEM1_QUEUE_SOURCE
                and not bool(command.stop)
            )
            if not queue_target_missing:
                raise
            identity = (
                str(command.episode_id),
                int(command.reset_generation),
            )
            with self._t4_pose_lock:
                if self._system1_queue_hold_identity != identity:
                    self._system1_queue_hold_identity = identity
                    self._system1_queue_hold_count = 0
                self._system1_queue_hold_count += 1
                hold_count = self._system1_queue_hold_count
            if hold_count > self.SYSTEM1_QUEUE_SAFE_HOLD_LIMIT:
                self._record_replan(
                    "system1_queue_target_missing_hold_limit_exceeded",
                    0,
                    int(command.sequence_id),
                    action_source=int(command.action_source),
                    hold_count=hold_count,
                    reason=self.SYSTEM1_QUEUE_TARGET_MISSING,
                )
                # super()._resolve_nav2 already issued the physical safe-stop.
                # Keep it asserted when an allegedly finite model queue exceeds
                # its registered 32-action bound.
                raise
            self._record_replan(
                "system1_queue_target_missing_safe_hold",
                0,
                int(command.sequence_id),
                action_source=int(command.action_source),
                hold_count=hold_count,
                reason=self.SYSTEM1_QUEUE_TARGET_MISSING,
            )
            # The resolver already published the physical safe-stop before it
            # raised.  Clear only the evaluator-facing terminal state; motion
            # remains disabled until a later accepted, identity-bound command.
            self._clear_safe_stop(
                "System1 queue target missing; safe hold while requesting "
                "a fresh same-reset trajectory"
            )
            return self._stand_resolution(
                command,
                "System1 queue target missing; awaiting fresh same-reset trajectory",
            )
        with self._t4_pose_lock:
            self._system1_queue_hold_identity = None
            self._system1_queue_hold_count = 0
        return response

    def _resolve_nav2(self, command: Any) -> ResolveCommand.Response:
        self._apply_step3_timeout_advice(command)
        with self._t4_pose_lock:
            pending = self._pending_typed_replan
        if pending is None:
            return self._resolve_nav2_with_terminal_fallback(command)
        identity_changed = False
        deadline_expired = False
        rejected = False
        rejection_limit_exceeded = False
        signature = None
        fresh_system2_action = False
        odometry_is_post_reset = False
        raw_wire_numeric_safe = False
        raw_wire_bypass = False
        with self._t4_pose_lock:
            if (
                str(command.episode_id) != pending["episode_id"]
                or int(command.reset_generation) != pending["reset_generation"]
            ):
                self._pending_typed_replan = None
                identity_changed = True
            else:
                request_index = int(pending["request_index"])
                contract_now_ns = _contract_now_ns(self)
                if contract_now_ns >= int(pending["deadline_ns"]):
                    deadline_expired = True
                    self._record_replan(
                        "deadline_expired",
                        request_index,
                        int(command.sequence_id),
                        recovery_id=pending["recovery_id"],
                    )
                points = [
                    (float(pose.pose.position.x), float(pose.pose.position.y))
                    for pose in command.local_path.poses
                ]
                try:
                    signature = trajectory_signature(points)
                except (TypeError, ValueError):
                    signature = None
                path_signature = signature
                fresh_system2_action = bool(
                    self.allow_system2_recovery_replan
                    and not bool(command.stop)
                    and int(command.action_source) == 1
                    and int(command.discrete_action) in {1, 2, 3}
                    and int(command.sequence_id)
                    > int(pending["active_sequence_id"])
                )
                if fresh_system2_action:
                    odometry = self._t4_odometry
                    odometry_is_post_reset = bool(
                        odometry is not None
                        and self._t4_odometry_serial
                        > self._t4_odometry_barrier_serial
                        and self._t4_odometry_stamp_ns
                        > self._reset_sim_barrier_ns
                        and self._t4_odometry_stamp_ns - contract_now_ns
                        <= self._sensor_future_tolerance_ns
                        and contract_now_ns - self._t4_odometry_stamp_ns
                        <= int(self.navigation_odometry_timeout * 1_000_000_000)
                    )
                    if odometry_is_post_reset:
                        try:
                            pose = _pose2d(odometry)
                            raw_wire_numeric_safe = all(
                                math.isfinite(value)
                                for value in (pose.x, pose.y, pose.yaw_rad)
                            )
                            signature = system2_primitive_signature(
                                action=int(command.discrete_action),
                                episode_id=str(command.episode_id),
                                reset_generation=int(command.reset_generation),
                                sequence_id=int(command.sequence_id),
                                observation_digest=str(command.observation_digest),
                                x=pose.x,
                                y=pose.y,
                                yaw_rad=pose.yaw_rad,
                            )
                        except (AttributeError, TypeError, ValueError):
                            signature = None
                policy = getattr(
                    self, "system2_replan_policy", "observation_bound"
                )
                candidates = (
                    (path_signature, signature)
                    if fresh_system2_action and policy == "strict"
                    else (signature,)
                )
                would_reject = any(
                    candidate is None
                    or candidate.absolute_sha256 in pending["excluded_absolute"]
                    or candidate.shape_sha256 in pending["excluded_shape"]
                    for candidate in candidates
                )
                raw_wire_bypass = bool(
                    policy == "raw_wire_warn"
                    and fresh_system2_action
                    and odometry_is_post_reset
                    and raw_wire_numeric_safe
                    and would_reject
                )
                rejected = bool(would_reject and not raw_wire_bypass)

        response = self._resolve_nav2_with_terminal_fallback(command)
        if identity_changed:
            return response
        bounded_fallback = bool(
            int(response.status_code) == 0
            and not bool(response.nav2_goal_sent)
            and not bool(response.nav2_plan_valid)
            and response.status_message == self.RECOVERY_BOUNDED_FALLBACK
        )
        if deadline_expired and (
            bool(response.nav2_goal_sent) or bool(response.nav2_plan_valid)
        ):
            message = "adapter executed a recovery trajectory after its sim-time deadline"
            self._safe_stop(STATUS_CONFLICT, message)
            raise ClientFailure(STATUS_CONFLICT, message)
        if deadline_expired or bounded_fallback:
            with self._t4_pose_lock:
                current = self._pending_typed_replan
                if (
                    current is not None
                    and current["operation_id"] == pending["operation_id"]
                ):
                    self._pending_typed_replan = None
            self._record_replan(
                (
                    "deadline_expired_bounded_fallback"
                    if deadline_expired
                    else "adapter_bounded_fallback"
                ),
                request_index,
                int(command.sequence_id),
                recovery_id=pending["recovery_id"],
                adapter_status_message=str(response.status_message),
                motion_disabled=True,
            )
            return response
        pre_arm_hold = bool(
            int(response.status_code) == 0
            and not bool(response.nav2_goal_sent)
            and not bool(response.nav2_plan_valid)
            and response.status_message
            == "recovery latch holding until replan gate is armed"
        )
        if pre_arm_hold:
            self._record_replan(
                "pre_arm_trajectory_suppressed",
                request_index,
                int(command.sequence_id),
                recovery_id=pending["recovery_id"],
                absolute_sha256=(
                    signature.absolute_sha256 if signature is not None else None
                ),
                shape_sha256=(
                    signature.shape_sha256 if signature is not None else None
                ),
                trajectory_source=(
                    "system2_primitive" if fresh_system2_action else "local_path"
                ),
            )
            return response
        if raw_wire_bypass:
            self._record_replan(
                "raw_wire_semantic_bypass_warn",
                request_index,
                int(command.sequence_id),
                recovery_id=pending["recovery_id"],
                absolute_sha256=(
                    signature.absolute_sha256 if signature is not None else None
                ),
                shape_sha256=(
                    signature.shape_sha256 if signature is not None else None
                ),
                trajectory_source=(
                    "system2_primitive" if fresh_system2_action else "local_path"
                ),
                deviation="signature_and_semantic_rejection_warn_only",
            )
        with self._t4_pose_lock:
            current = self._pending_typed_replan
            if (
                rejected
                and current is not None
                and current["operation_id"] == pending["operation_id"]
            ):
                current["rejection_count"] = int(current["rejection_count"]) + 1
                self._record_replan(
                    "old_trajectory_rejected",
                    request_index,
                    int(command.sequence_id),
                    recovery_id=pending["recovery_id"],
                    rejection_count=current["rejection_count"],
                    absolute_sha256=(
                        signature.absolute_sha256 if signature is not None else None
                    ),
                    shape_sha256=(
                        signature.shape_sha256 if signature is not None else None
                    ),
                )
                rejection_limit_exceeded = int(current["rejection_count"]) > 2
        if rejected and (
            bool(response.nav2_goal_sent) or bool(response.nav2_plan_valid)
        ):
            message = "adapter released an excluded recovery trajectory"
            self._safe_stop(STATUS_CONFLICT, message)
            raise ClientFailure(STATUS_CONFLICT, message)
        with self._t4_pose_lock:
            current = self._pending_typed_replan
            if (
                rejection_limit_exceeded
                and current is not None
                and current["operation_id"] == pending["operation_id"]
            ):
                self._pending_typed_replan = None
            if (
                current is not None
                and current["operation_id"] == pending["operation_id"]
                and int(response.status_code) == 0
                and not (
                    bool(response.nav2_goal_sent)
                    and bool(response.nav2_plan_valid)
                )
            ):
                if signature is not None:
                    current["excluded_absolute"].add(signature.absolute_sha256)
                    current["excluded_shape"].add(signature.shape_sha256)
                self._record_replan(
                    "pre_release_trajectory_quarantined",
                    request_index,
                    int(command.sequence_id),
                    recovery_id=pending["recovery_id"],
                    absolute_sha256=(
                        signature.absolute_sha256 if signature is not None else None
                    ),
                    shape_sha256=(
                        signature.shape_sha256 if signature is not None else None
                    ),
                    trajectory_source=(
                        "system2_primitive" if fresh_system2_action else "local_path"
                    ),
                )
            if (
                current is not None
                and current["operation_id"] == pending["operation_id"]
                and int(response.status_code) == 0
                and bool(response.nav2_goal_sent)
                and bool(response.nav2_plan_valid)
            ):
                self._pending_typed_replan = None
                self._last_typed_replan_sequence = int(command.sequence_id)
                self._record_replan(
                    "consumed_by_fresh_model_step",
                    request_index,
                    int(command.sequence_id),
                    recovery_id=pending["recovery_id"],
                    operation_id=pending["operation_id"],
                    absolute_sha256=(
                        signature.absolute_sha256 if signature is not None else None
                    ),
                    shape_sha256=(
                        signature.shape_sha256 if signature is not None else None
                    ),
                    trajectory_source=(
                        "system2_primitive" if fresh_system2_action else "local_path"
                    ),
                )
        if rejection_limit_exceeded:
            raise ClientFailure(
                STATUS_TIMEOUT,
                "model repeated an excluded recovery trajectory",
            )
        return response

    def _apply_step3_timeout_advice(self, command: Any) -> None:
        """Map one Step3 choice onto the existing bounded System2 primitive path."""

        if not getattr(self, "_step3_timeout_enabled", False):
            return
        with self._step3_timeout_condition:
            pending = self._step3_timeout_pending
            advice = self._step3_timeout_advice
            if (
                pending is None
                or advice is None
                or pending.get("kind")
                != "motion_timeout_after_confirmed_safe_stop"
            ):
                return
            identity = (
                str(command.episode_id),
                int(command.reset_generation),
                int(command.sequence_id),
            )
            expected = (
                pending["episode_id"],
                pending["reset_generation"],
                pending["expected_sequence_id"],
            )
            self._step3_timeout_pending = None
            self._step3_timeout_advice = None
        model_motion = bool(
            not bool(command.stop)
            and int(command.discrete_action)
            in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}
        )
        if (
            identity != expected
            or advice.get("status") != "ADVISE"
            or not model_motion
        ):
            self._record_motion_gate_event(
                "step3_timeout_fallback",
                str(pending.get("stop_token", "")),
                {
                    "reason": (
                        "sequence_identity_mismatch"
                        if identity != expected
                        else "internvla_terminal_or_hold_retained"
                        if not model_motion
                        else str(advice.get("reason", "advisor_fallback"))
                    ),
                    "expected_identity": list(expected),
                    "observed_identity": list(identity),
                },
            )
            return
        advised_action = int(advice["advised_action"])
        original_action = int(command.discrete_action)
        command.discrete_action = advised_action
        command.stop = False
        command.action_source = 1
        command.trajectory_source = 0
        command.trajectory_valid = False
        command.local_path.poses = []
        self._step3_timeout_interventions += 1
        self._step3_timeout_override = {
            "episode_id": str(command.episode_id),
            "reset_generation": int(command.reset_generation),
            "sequence_id": int(command.sequence_id),
            "trigger_sequence_id": int(pending["trigger_sequence_id"]),
            "trigger_request_id": str(pending["trigger_request_id"]),
            "original_model_action": original_action,
            "advised_action": advised_action,
            "confidence": float(advice["confidence"]),
            "snapshot_id": str(advice.get("snapshot_id", "")),
            "service_wall_latency_sec": float(
                advice.get("service_wall_latency_sec", 0.0)
            ),
            "camera_count": int(advice.get("camera_count", 0)),
        }
        self._record_motion_gate_event(
            "step3_timeout_advice_applied",
            str(pending.get("stop_token", "")),
            self._step3_timeout_override,
        )

    def _step3_timeout_wait_response(
        self,
        *,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        age: float,
    ) -> dict[str, Any] | None:
        """Hold the confirmed stop until Step3 replies or its wall watchdog fires."""

        if not self._step3_timeout_enabled:
            return None
        arrival_followup: tuple[str, dict[str, Any], int] | None = None
        arrival_terminal: tuple[dict[str, Any], dict[str, Any]] | None = None
        arrival_fallback: tuple[dict[str, Any], dict[str, Any]] | None = None
        with self._step3_timeout_condition:
            pending = self._step3_timeout_pending
            advice = self._step3_timeout_advice
            if pending is None:
                return None
            if time.monotonic_ns() >= int(pending["wall_deadline_monotonic_ns"]):
                advice = {
                    "schema_version": 1,
                    "status": "FALLBACK",
                    "episode_id": pending["episode_id"],
                    "reset_generation": pending["reset_generation"],
                    "trigger_sequence_id": pending["trigger_sequence_id"],
                    "trigger_request_id": pending["trigger_request_id"],
                    "advisor_round": pending["advisor_round"],
                    "confidence": 0.0,
                    "reason": "step3_service_wall_timeout",
                }
                self._step3_timeout_advice = advice
            kind = str(pending.get("kind", ""))
            if kind == "motion_timeout_after_confirmed_safe_stop" and advice is not None:
                return None
            if kind == "arrival_check_after_completed_motion" and advice is not None:
                transition = arrival_transition(
                    pending,
                    advice,
                    required_confirmations=(
                        self.STEP3_ARRIVAL_REQUIRED_CONFIRMATIONS
                    ),
                )
                if transition.action == REQUEST_CONFIRMATION:
                    arrival_followup = (
                        str(pending.get("stop_token", "")),
                        dict(pending),
                        int(transition.snapshot_sim_stamp_ns or 0),
                    )
                elif transition.action == TERMINATE_ASSISTED:
                    arrival_terminal = (dict(pending), dict(advice))
                elif transition.action == CONTINUE_NAVIGATION:
                    arrival_fallback = (dict(pending), dict(advice))
                self._step3_timeout_pending = None
                self._step3_timeout_advice = None
            if arrival_terminal is None and arrival_fallback is None:
                remaining = max(
                    0.0,
                    (
                        int(pending["wall_deadline_monotonic_ns"])
                        - time.monotonic_ns()
                    )
                    / 1_000_000_000,
                )
            else:
                remaining = 0.0
        if arrival_followup is not None:
            token, previous, first_snapshot_stamp_ns = arrival_followup
            published = self._publish_step3_arrival_context(
                token=token,
                pending=previous,
                advisor_round=2,
                minimum_snapshot_sim_stamp_ns=first_snapshot_stamp_ns,
            )
            if not published:
                self._record_motion_gate_event(
                    "step3_arrival_fallback",
                    token,
                    {"reason": "arrival_confirmation_budget_exhausted"},
                )
                return None
            pending = self._step3_timeout_pending or previous
            remaining = 12.0
        if arrival_fallback is not None:
            fallback_pending, fallback_advice = arrival_fallback
            self._record_motion_gate_event(
                "step3_arrival_not_confirmed",
                str(fallback_pending.get("stop_token", "")),
                {
                    "status": fallback_advice.get("status"),
                    "reason": fallback_advice.get("reason"),
                    "confidence": fallback_advice.get("confidence"),
                    "advisor_round": fallback_pending.get("advisor_round"),
                },
            )
            return None
        if arrival_terminal is not None:
            terminal_pending, terminal_advice = arrival_terminal
            terminal = {
                "status_code": STATUS_OK,
                "status_message": (
                    "Step3 arrival advisor confirmed destination on two "
                    "fresh safe-hold snapshots"
                ),
                "episode_id": self.episode_id,
                "reset_generation": self.reset_generation,
                "sequence_id": int(terminal_pending["trigger_sequence_id"]),
                "request_id": str(terminal_pending["trigger_request_id"]),
                "sim_stamp_ns": int(sim_stamp_ns),
                "discrete_action": ACTION_STAND_STILL,
                "model_discrete_action": int(
                    terminal_pending.get("completed_action", ACTION_STAND_STILL)
                ),
                "stop": True,
                "model_stop": False,
                "step3_assisted_stop": True,
                "termination_source": "step3_assisted_arrival",
                "step3_arrival_confidence": float(terminal_advice["confidence"]),
                "step3_arrival_confirmations": (
                    self.STEP3_ARRIVAL_REQUIRED_CONFIRMATIONS
                ),
                "step3_arrival_snapshot_id": str(
                    terminal_advice.get("snapshot_id", "")
                ),
                "control_mode": self.control_mode,
                "action_source": 1,
                "trajectory_source": 0,
                "trajectory_valid": False,
                "local_path": [],
                "nav2_goal_sent": False,
                "nav2_plan_valid": False,
                "motion_observation_gate_only": False,
                "observation_pose_source": "navigation_odometry",
                "evaluator_ground_truth_pose_discarded": True,
                "navigation_odometry_age_sec": age,
                "navigation_odometry_stamp_ns": int(odom_stamp_ns),
                "navigation_odometry_serial": int(odom_serial),
            }
            self._record_motion_gate_event(
                "step3_assisted_stop",
                str(terminal_pending.get("stop_token", "")),
                {
                    "termination_source": terminal["termination_source"],
                    "model_stop": False,
                    "confidence": terminal["step3_arrival_confidence"],
                    "confirmations": terminal["step3_arrival_confirmations"],
                },
            )
            return terminal
        return {
            "status_code": STATUS_OK,
            "status_message": "safe-stop held while Step3 timeout advice is pending",
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sequence_id": int(pending["trigger_sequence_id"]),
            "request_id": str(pending["trigger_request_id"]),
            "sim_stamp_ns": int(sim_stamp_ns),
            "discrete_action": ACTION_STAND_STILL,
            "model_discrete_action": int(pending["excluded_action"]),
            "stop": True,
            "model_stop": False,
            "control_mode": self.control_mode,
            "nav2_goal_sent": False,
            "nav2_plan_valid": False,
            "motion_observation_gate_only": True,
            "step3_timeout_advisor_pending": True,
            "step3_advisor_kind": str(pending.get("kind", "")),
            "step3_timeout_advisor_remaining_wall_sec": remaining,
            "observation_pose_source": "navigation_odometry",
            "evaluator_ground_truth_pose_discarded": True,
            "navigation_odometry_age_sec": age,
            "navigation_odometry_stamp_ns": int(odom_stamp_ns),
            "navigation_odometry_serial": int(odom_serial),
        }

    def _on_t4_odometry(self, message: Odometry) -> None:
        callback_epoch = self._motion_gate_epoch
        pose = _pose2d(message)
        odom_stamp_ns = _stamp_ns(message.header.stamp)
        semantic_now = _semantic_now(self)
        sim_stamp_ns = int(semantic_now * 1_000_000_000)
        invalid_reason: str | None = None
        if self._motion_gate_enabled:
            if sim_stamp_ns <= 0:
                invalid_reason = "simulation clock unavailable for navigation odometry"
            elif odom_stamp_ns <= 0:
                invalid_reason = "navigation odometry has a zero header stamp"
            elif (
                odom_stamp_ns - sim_stamp_ns
                > self._sensor_future_tolerance_ns
            ):
                invalid_reason = "navigation odometry header stamp is in the future"
            elif (sim_stamp_ns - odom_stamp_ns) / 1_000_000_000 > self.navigation_odometry_timeout:
                invalid_reason = "navigation odometry header stamp is stale"
        duplicate_sample = False
        with self._t4_pose_lock:
            if callback_epoch != self._motion_gate_epoch:
                invalid_reason = "odometry callback crossed the reset epoch"
            elif self._motion_gate_enabled and odom_stamp_ns <= self._reset_sim_barrier_ns:
                invalid_reason = "navigation odometry predates the reset barrier"
            elif self._motion_gate_enabled and invalid_reason is None:
                stamp_order = classify_odometry_stamp(
                    self._t4_odometry_stamp_ns, odom_stamp_ns
                )
                if stamp_order == ODOMETRY_STAMP_REGRESSED:
                    invalid_reason = (
                        "navigation odometry header stamp did not strictly advance"
                    )
                elif stamp_order == ODOMETRY_STAMP_DUPLICATE:
                    duplicate_sample = True
            if invalid_reason is None and not duplicate_sample:
                self._t4_odometry = message
                self._t4_odometry_monotonic = semantic_now
                self._t4_odometry_stamp_ns = odom_stamp_ns
                self._t4_odometry_serial += 1
                odom_serial = self._t4_odometry_serial
                self._t4_pose_lock.notify_all()
            else:
                odom_serial = self._t4_odometry_serial
        if duplicate_sample:
            return
        if invalid_reason is not None:
            if self._motion_gate_enabled and self.initialized:
                self._fail_active_motion_from_callback(
                    invalid_reason,
                    callback_epoch=callback_epoch,
                    sim_stamp_ns=max(0, sim_stamp_ns),
                    odom_stamp_ns=max(0, odom_stamp_ns),
                    odom_serial=max(0, odom_serial),
                    pose=pose,
                )
            return
        if self._motion_gate_enabled and self.initialized:
            # Isaac may publish a bounded sensor burst ahead of the most recent
            # /clock sample.  Advance only this gate observation; never mutate
            # the node's global semantic-clock watermark from a sensor stamp.
            gate_sim_stamp_ns = max(sim_stamp_ns, odom_stamp_ns)
            self._observe_motion_gate(
                sim_stamp_ns=gate_sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                pose=pose,
                callback_epoch=callback_epoch,
            )

    def _fail_active_motion_from_callback(
        self,
        reason: str,
        *,
        callback_epoch: int,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        pose: Pose2D,
    ) -> None:
        with self.operation_lock:
            if callback_epoch != self._motion_gate_epoch:
                return
            with self._motion_gate_lock:
                if self._motion_gate.pending is None:
                    return
                decision = self._motion_gate.fail_active_motion(
                    episode_id=self.episode_id,
                    reset_generation=self.reset_generation,
                    sim_stamp_ns=sim_stamp_ns,
                    odom_stamp_ns=odom_stamp_ns,
                    odom_serial=odom_serial,
                    reason=reason,
                )
            try:
                self._request_motion_stop_and_wait(decision)
            except ClientFailure as exc:
                self._motion_gate_fault = str(exc)
                self.get_logger().error(f"motion-gate fail-closed stop failed: {exc}")

    def _observe_motion_gate(
        self,
        *,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        pose: Pose2D,
        callback_epoch: int | None = None,
    ) -> GateDecision | None:
        with self.operation_lock:
            if callback_epoch is not None and callback_epoch != self._motion_gate_epoch:
                return None
            if self._motion_gate_fault is not None:
                raise ClientFailure(STATUS_STALE, self._motion_gate_fault)
            with self._t4_pose_lock:
                camera_stamp_ns = self._camera_sensor_stamp_ns
                camera_serial = self._camera_sensor_serial
            with self._motion_gate_lock:
                if (
                    self._motion_gate.active_episode_id != self.episode_id
                    or self._motion_gate.active_reset_generation
                    != self.reset_generation
                ):
                    # initialize/reset owns the identity transition.  A queued
                    # callback from an old epoch cannot revive its identity.
                    return None
                decision = self._motion_gate.observe(
                    episode_id=self.episode_id,
                    reset_generation=self.reset_generation,
                    sim_stamp_ns=sim_stamp_ns,
                    odom_stamp_ns=odom_stamp_ns,
                    odom_serial=odom_serial,
                    camera_sensor_stamp_ns=camera_stamp_ns,
                    camera_sensor_serial=camera_serial,
                    pose=pose,
                )
                snapshot = self._motion_gate.snapshot()
            if decision.requires_safe_stop:
                self._request_motion_stop_and_wait(decision)
                self._record_motion_gate(
                    decision,
                    sim_stamp_ns=sim_stamp_ns,
                    odom_stamp_ns=odom_stamp_ns,
                    odom_serial=odom_serial,
                    snapshot=snapshot,
                )
            return decision

    def _record_motion_gate(
        self,
        decision: GateDecision,
        *,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        if self.motion_gate_audit_path is None:
            return
        if snapshot is None:
            with self._motion_gate_lock:
                snapshot = self._motion_gate.snapshot()
        record = {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sim_stamp_ns": int(sim_stamp_ns),
            "odometry_stamp_ns": int(odom_stamp_ns),
            "odometry_serial": int(odom_serial),
            "decision": decision.to_mapping(),
            "gate": snapshot,
            "wall_time_unix": time.time(),
        }
        with self._motion_gate_record_lock:
            with self.motion_gate_audit_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _on_t4_reset_generation(self, message: Int32) -> None:
        with self._t4_pose_lock:
            self._t4_map_reset_generation = max(
                self._t4_map_reset_generation, int(message.data)
            )
            self._t4_pose_lock.notify_all()

    def _publish_episode_prime(self) -> None:
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "episode_id": self.episode_id,
                "reset_generation": self.reset_generation,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.episode_prime_publisher.publish(message)

    def initialize(self, episode_id: str) -> dict[str, Any]:
        with self.operation_lock:
            response = super().initialize(episode_id)
            with self._t4_pose_lock:
                self._system1_queue_hold_identity = None
                self._system1_queue_hold_count = 0
            self._arm_odometry_barrier()
            self._reset_motion_gate("initialize")
            self._publish_episode_prime()
            return response

    def reset(self, next_episode_id: str) -> dict[str, Any]:
        with self.operation_lock:
            confirmed_stop = self._confirm_stop_before_identity_change(
                "episode reset"
            )
            response = super().reset(next_episode_id)
            with self._t4_pose_lock:
                self._system1_queue_hold_identity = None
                self._system1_queue_hold_count = 0
            self._arm_odometry_barrier()
            self._install_post_reset_sensor_barrier(
                "episode reset", confirmed_stop
            )
            self._publish_episode_prime()
            return response

    def _confirm_stop_before_identity_change(self, reason: str) -> dict[str, Any]:
        """Confirm Nav2 cancellation while the old identity is authoritative."""

        if not self._motion_gate_enabled:
            return {}
        sim_stamp_ns = _t5_semantic_now_ns(self)
        if sim_stamp_ns is None:
            self._safe_stop(STATUS_STALE, "simulation clock unavailable before reset")
            raise ClientFailure(
                STATUS_STALE, "simulation clock unavailable before reset"
            )
        with self._t4_pose_lock:
            odom_stamp_ns = self._t4_odometry_stamp_ns
            odom_serial = self._t4_odometry_serial
            camera_stamp_ns = self._camera_sensor_stamp_ns
            camera_serial = self._camera_sensor_serial
        with self._motion_gate_lock:
            snapshot = self._motion_gate.snapshot()
            barrier = snapshot.get("stop_barrier")
            if self._motion_gate.pending is not None:
                decision = self._motion_gate.fail_active_motion(
                    episode_id=self.episode_id,
                    reset_generation=self.reset_generation,
                    sim_stamp_ns=sim_stamp_ns,
                    odom_stamp_ns=odom_stamp_ns,
                    odom_serial=odom_serial,
                    reason=f"pre-reset safe stop: {reason}",
                )
            elif barrier is None:
                decision = self._motion_gate.require_stop_barrier(
                    episode_id=self.episode_id,
                    reset_generation=self.reset_generation,
                    sequence_id=max(-1, self.last_committed_sequence),
                    request_id=(
                        f"pre-reset:{self.episode_id}:{self.reset_generation}"
                    ),
                    sim_stamp_ns=sim_stamp_ns,
                    odom_stamp_ns=odom_stamp_ns,
                    odom_serial=odom_serial,
                    camera_sensor_stamp_ns=camera_stamp_ns,
                    camera_sensor_serial=camera_serial,
                    reason=f"pre-reset safe stop: {reason}",
                )
            elif not bool(barrier.get("cancel_acknowledged", False)):
                decision = GateDecision(
                    kind=DECISION_SAFE_STOP_STALE,
                    reason=f"resume pre-reset cancellation: {reason}",
                    permits_model_step=False,
                    requires_safe_stop=True,
                    keep_safe_stop=True,
                )
            else:
                decision = None
            gate_snapshot = self._motion_gate.snapshot()
        if decision is not None:
            self._request_motion_stop_and_wait(decision)
            self._record_motion_gate(
                decision,
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                snapshot=gate_snapshot,
            )
        else:
            self._safe_stop(STATUS_OK, f"pre-reset cancellation retained: {reason}")
        with self._motion_gate_lock:
            confirmed = self._motion_gate.snapshot().get("stop_barrier")
        if not isinstance(confirmed, dict) or not bool(
            confirmed.get("cancel_acknowledged", False)
        ):
            self._safe_stop(STATUS_STALE, "pre-reset cancellation was not confirmed")
            raise ClientFailure(
                STATUS_STALE, "pre-reset cancellation was not confirmed"
            )
        self._record_motion_gate_event(
            "pre_reset_cancel_confirmed",
            "",
            {
                "reason": reason,
                "old_episode_id": self.episode_id,
                "old_reset_generation": self.reset_generation,
                "barrier": confirmed,
            },
        )
        return dict(confirmed)

    def _install_post_reset_sensor_barrier(
        self, reason: str, confirmed_stop: dict[str, Any]
    ) -> None:
        """Switch identity only after old-goal cancellation, then gate sensors."""

        if not self._motion_gate_enabled:
            return
        if not bool(confirmed_stop.get("cancel_acknowledged", False)):
            raise ClientFailure(
                STATUS_STALE, "post-reset barrier lacks confirmed old cancellation"
            )
        with self._step3_timeout_condition:
            self._step3_timeout_pending = None
            self._step3_timeout_advice = None
            self._step3_timeout_override = None
            self._step3_timeout_interventions = 0
            self._step3_arrival_completed_actions = 0
            self._step3_arrival_checks = 0
        sim_stamp_ns = _t5_semantic_now_ns(self)
        if sim_stamp_ns is None:
            self._safe_stop(STATUS_STALE, "simulation clock unavailable after reset")
            raise ClientFailure(
                STATUS_STALE, "simulation clock unavailable after reset"
            )
        with self._t4_pose_lock:
            odom_serial = max(
                self._t4_odometry_serial,
                int(confirmed_stop.get("cancel_ack_odom_serial", 0)),
            )
            camera_serial = max(
                self._camera_sensor_serial,
                int(confirmed_stop.get("cancel_ack_camera_serial", 0)),
            )
            cutoff_ns = max(
                sim_stamp_ns,
                self._reset_sim_barrier_ns,
                int(confirmed_stop.get("cancel_ack_sim_ns", 0)),
            )
            odom_cutoff_ns = max(
                cutoff_ns,
                int(confirmed_stop.get("cancel_ack_odom_stamp_ns", 0)),
            )
            camera_cutoff_ns = max(
                cutoff_ns,
                int(confirmed_stop.get("cancel_ack_camera_stamp_ns", 0)),
            )
        with self._motion_gate_lock:
            self._motion_gate.reset(self.episode_id, self.reset_generation)
            decision = self._motion_gate.inherit_confirmed_stop_barrier(
                episode_id=self.episode_id,
                reset_generation=self.reset_generation,
                sequence_id=max(-1, self.last_committed_sequence),
                request_id=f"post-reset:{self.episode_id}:{self.reset_generation}",
                sim_stamp_ns=cutoff_ns,
                odom_stamp_ns=odom_cutoff_ns,
                odom_serial=odom_serial,
                camera_sensor_stamp_ns=camera_cutoff_ns,
                camera_sensor_serial=camera_serial,
                reason=(
                    "old-identity Nav2 cancellation confirmed; waiting for "
                    f"post-reset sensors: {reason}"
                ),
            )
            snapshot = self._motion_gate.snapshot()
        self._motion_gate_fault = None
        self._safe_stop(STATUS_OK, decision.reason)
        self._record_motion_gate(
            decision,
            sim_stamp_ns=cutoff_ns,
            odom_stamp_ns=odom_cutoff_ns,
            odom_serial=odom_serial,
            snapshot=snapshot,
        )

    def _reset_motion_gate(self, reason: str) -> None:
        if not self._motion_gate_enabled:
            return
        with self._step3_timeout_condition:
            self._step3_timeout_pending = None
            self._step3_timeout_advice = None
            self._step3_timeout_override = None
            self._step3_timeout_interventions = 0
            self._step3_arrival_completed_actions = 0
            self._step3_arrival_checks = 0
        self._motion_gate_fault = None
        sim_stamp_ns = _t5_semantic_now_ns(self)
        if sim_stamp_ns is None:
            raise ClientFailure(STATUS_STALE, "simulation clock unavailable at reset")
        with self._t4_pose_lock:
            odom_stamp_ns = self._t4_odometry_stamp_ns
            odom_serial = self._t4_odometry_serial
            camera_stamp_ns = self._camera_sensor_stamp_ns
            camera_serial = self._camera_sensor_serial
        with self._motion_gate_lock:
            self._motion_gate.reset(self.episode_id, self.reset_generation)
            decision = self._motion_gate.require_stop_barrier(
                episode_id=self.episode_id,
                reset_generation=self.reset_generation,
                sequence_id=max(-1, self.last_committed_sequence),
                request_id=f"reset:{self.episode_id}:{self.reset_generation}",
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                camera_sensor_stamp_ns=camera_stamp_ns,
                camera_sensor_serial=camera_serial,
                reason=f"motion gate reset: {reason}",
            )
            snapshot = self._motion_gate.snapshot()
        self._request_motion_stop_and_wait(decision)
        self._record_motion_gate(
            decision,
            sim_stamp_ns=sim_stamp_ns,
            odom_stamp_ns=odom_stamp_ns,
            odom_serial=odom_serial,
            snapshot=snapshot,
        )

    def _arm_odometry_barrier(self) -> None:
        reset_sim_ns = (
            _t5_semantic_now_ns(self) if self._motion_gate_enabled else None
        )
        with self._t4_pose_lock:
            self._motion_gate_epoch += 1
            self._reset_sim_barrier_ns = max(0, int(reset_sim_ns or 0))
            self._t4_odometry_barrier_serial = self._t4_odometry_serial
            self._t4_odometry = None
            self._t4_odometry_monotonic = 0.0
            self._t4_odometry_stamp_ns = 0

    def _navigation_pose(
        self,
    ) -> tuple[list[float], list[float], float, int, int, Pose2D]:
        liveness_timeout = (
            max(5.0, self.navigation_odometry_timeout)
            if getattr(self, "_t5_sim_time_semantics", False)
            else self.navigation_odometry_timeout
        )
        deadline = time.monotonic() + liveness_timeout
        while True:
            # Keep clock -> pose as the only lock order; odometry callbacks use
            # the same order.  Calling the semantic clock while holding the
            # pose condition would otherwise deadlock a concurrent callback.
            sim_now_ns = (
                _t5_semantic_now_ns(self) if self._motion_gate_enabled else None
            )
            with self._t4_pose_lock:
                if self._motion_gate_enabled:
                    age = (
                        None
                        if sim_now_ns is None or self._t4_odometry_stamp_ns <= 0
                        else max(0, sim_now_ns - self._t4_odometry_stamp_ns)
                        / 1_000_000_000
                    )
                else:
                    age = _semantic_age(self, self._t4_odometry_monotonic)
                if (
                    self._t4_odometry is not None
                    and self._t4_odometry_serial
                    > self._t4_odometry_barrier_serial
                    and age is not None
                    and age <= self.navigation_odometry_timeout
                    and (
                        not self._t5_sim_time_semantics
                        or self._t4_odometry_stamp_ns > self._reset_sim_barrier_ns
                    )
                    and (
                        self.allow_nearest_navigation_odometry
                        or self._t4_map_reset_generation >= self.reset_generation
                    )
                ):
                    message = self._t4_odometry
                    odom_stamp_ns = self._t4_odometry_stamp_ns
                    odom_serial = self._t4_odometry_serial
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    age_text = "unavailable" if age is None else f"{age:.3f}"
                    raise ClientFailure(
                        STATUS_TIMEOUT,
                        "fresh navigation odometry unavailable: "
                        f"present={self._t4_odometry is not None} "
                        f"receipt_serial={self._t4_odometry_serial} "
                        f"barrier_serial={self._t4_odometry_barrier_serial} "
                        f"age_sec={age_text} timeout_sec={self.navigation_odometry_timeout:.3f} "
                        f"liveness_timeout_sec={liveness_timeout:.3f} "
                        f"observed_map_generation={self._t4_map_reset_generation} "
                        f"required_generation={self.reset_generation} "
                        f"nearest_allowed={self.allow_nearest_navigation_odometry}",
                    )
                self._t4_pose_lock.wait(remaining)
        pose = message.pose.pose
        gps = [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ]
        rotation = [
            float(pose.orientation.w),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
        ]
        norm = math.sqrt(sum(value * value for value in rotation))
        if not all(math.isfinite(value) for value in gps + rotation) or norm < 0.5:
            raise ClientFailure(STATUS_TIMEOUT, "navigation odometry pose is invalid")
        rotation = [value / norm for value in rotation]
        return gps, rotation, age, odom_stamp_ns, odom_serial, _pose2d(message)

    def _motion_start_baseline(self) -> tuple[Pose2D, int, int, int]:
        """Snapshot the latest accepted odometry after command resolution.

        Odometry may continue to arrive while the synchronous model/Nav2 step
        runs.  All of those samples precede this baseline and therefore cannot
        be mistaken for progress produced by the newly resolved command.
        """

        sim_stamp_ns = _t5_semantic_now_ns(self)
        if sim_stamp_ns is None:
            raise RuntimeError("simulation clock unavailable at motion arm")
        with self._t4_pose_lock:
            message = self._t4_odometry
            odom_stamp_ns = self._t4_odometry_stamp_ns
            odom_serial = self._t4_odometry_serial
            odom_barrier_serial = self._t4_odometry_barrier_serial
            reset_sim_barrier_ns = self._reset_sim_barrier_ns
        if message is None:
            raise RuntimeError("navigation odometry unavailable at motion arm")
        if odom_serial <= odom_barrier_serial:
            raise RuntimeError("navigation odometry predates the reset receipt barrier")
        if odom_stamp_ns <= reset_sim_barrier_ns:
            raise RuntimeError("navigation odometry predates the reset time barrier")
        if odom_stamp_ns <= 0:
            raise RuntimeError("navigation odometry has a zero stamp at motion arm")
        if (
            odom_stamp_ns - sim_stamp_ns
            > self._sensor_future_tolerance_ns
        ):
            raise RuntimeError("navigation odometry is in the future at motion arm")
        gate_sim_stamp_ns = max(sim_stamp_ns, odom_stamp_ns)
        age_sec = max(0, sim_stamp_ns - odom_stamp_ns) / 1_000_000_000
        if age_sec > self.navigation_odometry_timeout:
            raise RuntimeError("navigation odometry is stale at motion arm")
        pose = _pose2d(message)
        if not all(
            math.isfinite(value) for value in (pose.x, pose.y, pose.yaw_rad)
        ):
            raise RuntimeError("navigation odometry pose is invalid at motion arm")
        return pose, gate_sim_stamp_ns, odom_stamp_ns, odom_serial

    def _stop_resolved_goal_after_failure(
        self, result: dict[str, Any], reason: str
    ) -> None:
        """Cancel an already-issued Nav2 goal and leave a sensor barrier."""

        self._safe_stop(STATUS_STALE, reason)
        sim_stamp_ns = _t5_semantic_now_ns(self)
        with self._t4_pose_lock:
            odom_stamp_ns = self._t4_odometry_stamp_ns
            odom_serial = self._t4_odometry_serial
            camera_stamp_ns = self._camera_sensor_stamp_ns
            camera_serial = self._camera_sensor_serial
        with self._motion_gate_lock:
            if self._motion_gate.pending is not None:
                decision = self._motion_gate.fail_active_motion(
                    episode_id=self.episode_id,
                    reset_generation=self.reset_generation,
                    sim_stamp_ns=max(0, int(sim_stamp_ns or 0)),
                    odom_stamp_ns=max(0, int(odom_stamp_ns)),
                    odom_serial=max(0, int(odom_serial)),
                    reason=reason,
                )
            elif self._motion_gate.stop_barrier is None:
                decision = self._motion_gate.require_stop_barrier(
                    episode_id=self.episode_id,
                    reset_generation=self.reset_generation,
                    sequence_id=int(
                        result.get("sequence_id", self.last_committed_sequence)
                    ),
                    request_id=str(result.get("request_id", "post-resolution")),
                    sim_stamp_ns=max(0, int(sim_stamp_ns or 0)),
                    odom_stamp_ns=max(0, int(odom_stamp_ns)),
                    odom_serial=max(0, int(odom_serial)),
                    camera_sensor_stamp_ns=max(0, int(camera_stamp_ns)),
                    camera_sensor_serial=max(0, int(camera_serial)),
                    reason=reason,
                )
            elif not self._motion_gate.stop_barrier.cancel_acknowledged:
                decision = GateDecision(
                    kind=DECISION_SAFE_STOP_STALE,
                    reason=reason,
                    permits_model_step=False,
                    requires_safe_stop=True,
                    keep_safe_stop=True,
                )
            else:
                raise RuntimeError(
                    "resolved Nav2 goal failure found an unrelated acknowledged barrier"
                )
            snapshot = self._motion_gate.snapshot()
        self._request_motion_stop_and_wait(decision)
        self._record_motion_gate(
            decision,
            sim_stamp_ns=max(0, int(sim_stamp_ns or 0)),
            odom_stamp_ns=max(0, int(odom_stamp_ns)),
            odom_serial=max(0, int(odom_serial)),
            snapshot=snapshot,
        )

    def step_arrays(self, **kwargs: Any) -> dict[str, Any]:
        # The stop/cancel/sensor barrier and the model action form one critical
        # section.  Odometry callbacks may update their cache concurrently, but
        # cannot transition/cancel the gate in the middle of an inference.
        with self.operation_lock:
            return self._step_arrays_locked(**kwargs)

    def _step_arrays_locked(self, **kwargs: Any) -> dict[str, Any]:
        # These two arrays came from the evaluator and are deliberately ignored.
        kwargs.pop("global_gps", None)
        kwargs.pop("global_rotation", None)
        self._last_navigation_instruction = str(kwargs.get("instruction", ""))
        self._step3_timeout_override = None
        camera_sequence = kwargs.pop("camera_sensor_sequence", None)
        camera_stamp_ns = kwargs.pop("camera_sensor_stamp_ns", None)
        camera_schema_version = kwargs.pop("camera_sensor_schema_version", None)
        camera_source = kwargs.pop("camera_sensor_source", None)
        if self._motion_gate_enabled:
            if (
                camera_sequence is None
                or camera_stamp_ns is None
                or camera_schema_version != 1
                or camera_source != "x86_isaac_pano_camera_0"
            ):
                self._safe_stop(STATUS_STALE, "camera sensor identity is missing")
                raise ClientFailure(STATUS_STALE, "camera sensor identity is missing")
            if (
                isinstance(camera_sequence, bool)
                or not isinstance(camera_sequence, int)
                or isinstance(camera_stamp_ns, bool)
                or not isinstance(camera_stamp_ns, int)
                or camera_sequence <= 0
                or camera_stamp_ns <= 0
            ):
                self._safe_stop(STATUS_STALE, "camera sensor identity is invalid")
                raise ClientFailure(STATUS_STALE, "camera sensor identity is invalid")
            sim_now_ns = _t5_semantic_now_ns(self)
            if sim_now_ns is None:
                self._safe_stop(STATUS_STALE, "camera validation clock is unavailable")
                raise ClientFailure(
                    STATUS_STALE, "camera validation clock is unavailable"
                )
            if (
                camera_stamp_ns - sim_now_ns
                > self._sensor_future_tolerance_ns
            ):
                self._safe_stop(STATUS_STALE, "camera sensor sim stamp is in the future")
                raise ClientFailure(
                    STATUS_STALE, "camera sensor sim stamp is in the future"
                )
            if (
                sim_now_ns - camera_stamp_ns
            ) / 1_000_000_000 > self.navigation_odometry_timeout:
                self._safe_stop(STATUS_STALE, "camera sensor source frame is stale")
                raise ClientFailure(STATUS_STALE, "camera sensor source frame is stale")
            with self._t4_pose_lock:
                if camera_sequence <= self._camera_sensor_serial:
                    self._safe_stop(
                        STATUS_STALE, "camera sensor sequence did not strictly advance"
                    )
                    raise ClientFailure(
                        STATUS_STALE, "camera sensor sequence did not strictly advance"
                    )
                if camera_stamp_ns <= self._camera_sensor_stamp_ns:
                    self._safe_stop(
                        STATUS_STALE,
                        "camera sensor source sim stamp did not strictly advance",
                    )
                    raise ClientFailure(
                        STATUS_STALE,
                        "camera sensor source sim stamp did not strictly advance",
                    )
                self._camera_sensor_serial = camera_sequence
                self._camera_sensor_stamp_ns = camera_stamp_ns
        gps, rotation, age, odom_stamp_ns, odom_serial, pose = (
            self._navigation_pose()
        )
        sim_stamp_ns = int(self.get_clock().now().nanoseconds)
        if self._motion_gate_enabled:
            # This is a local gate time only.  Source stamps within the bounded
            # completion_sim lead do not advance the global /clock watermark.
            sim_stamp_ns = max(sim_stamp_ns, odom_stamp_ns, int(camera_stamp_ns))
            decision = self._observe_motion_gate(
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                pose=pose,
            )
            if decision is None:
                self._safe_stop(
                    STATUS_STALE, "motion gate identity transition is incomplete"
                )
                raise ClientFailure(
                    STATUS_STALE, "motion gate identity transition is incomplete"
                )
            if not decision.permits_model_step:
                return self._motion_gate_response(
                    decision,
                    sim_stamp_ns=sim_stamp_ns,
                    odom_stamp_ns=odom_stamp_ns,
                    odom_serial=odom_serial,
                    age=age,
                )
            step3_wait = self._step3_timeout_wait_response(
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                age=age,
            )
            if step3_wait is not None:
                return step3_wait
        with self._t4_pose_lock:
            replan_requested = self._replan_requested
            replan_index = self._replan_request_index
            if replan_requested:
                self._replan_requested = False
        result = super().step_arrays(
            global_gps=gps,
            global_rotation=rotation,
            **kwargs,
        )
        step3_override = self._step3_timeout_override
        if step3_override is not None:
            observed_identity = (
                str(result.get("episode_id", "")),
                int(result.get("reset_generation", -1)),
                int(result.get("sequence_id", -1)),
            )
            expected_identity = (
                step3_override["episode_id"],
                step3_override["reset_generation"],
                step3_override["sequence_id"],
            )
            if observed_identity != expected_identity:
                raise ClientFailure(
                    STATUS_STALE, "Step3 timeout execution identity drifted"
                )
            advised_action = int(step3_override["advised_action"])
            result["action_source"] = 1
            result["trajectory_source"] = 0
            result["trajectory_valid"] = False
            result["discrete_action"] = advised_action
            result["local_path"] = (
                [[0.0, 0.0], [0.25, 0.0]]
                if advised_action == ACTION_FORWARD
                else []
            )
            result["step3_timeout_advisor_used"] = True
            result["step3_timeout_advisor"] = step3_override
        resolved_goal_sent = bool(result.get("nav2_goal_sent", False))
        try:
            result["observation_pose_source"] = "navigation_odometry"
            result["evaluator_ground_truth_pose_discarded"] = True
            result["navigation_odometry_age_sec"] = age
            result["navigation_odometry_reset_match_required"] = (
                not self.allow_nearest_navigation_odometry
            )
            result["recovery_trajectory_request_consumed"] = replan_requested
            result["typed_recovery_trajectory_accepted"] = (
                int(result.get("sequence_id", -1))
                == self._last_typed_replan_sequence
            )
            if self._motion_gate_enabled:
                model_action = int(result.get("model_discrete_action", -1))
                resolved_action = int(
                    result.get("discrete_action", ACTION_STAND_STILL)
                )
                system1_queue_plan_motion = is_system1_queue_plan_motion(
                    action_source=int(result.get("action_source", -1)),
                    trajectory_valid=bool(result.get("trajectory_valid", False)),
                    nav2_goal_sent=resolved_goal_sent,
                    nav2_plan_valid=bool(result.get("nav2_plan_valid", False)),
                    stop=bool(result.get("stop", False)),
                    model_action=model_action,
                )
                # The continuous Nav2 adapter intentionally reports STAND while
                # it owns cmd_vel.  A newly accepted goal is nevertheless a
                # real model motion and must arm the action/observation gate. A
                # validated System1 queue response may retain that valid plan
                # without returning a new local path.
                if (
                    resolved_action == ACTION_STAND_STILL
                    and resolved_goal_sent
                    and bool(result.get("nav2_plan_valid", False))
                    and model_action
                    in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}
                ):
                    resolved_action = model_action
                should_arm = bool(
                    resolved_action in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}
                    and not bool(result.get("stop", False))
                    and bool(result.get("nav2_plan_valid", False))
                )
                if should_arm:
                    resolved_commanded, resolved_required = (
                        self._resolved_motion_bounds(
                            resolved_action,
                            result,
                            use_preregistered_bounds=system1_queue_plan_motion,
                        )
                    )
                    (
                        start_pose,
                        start_sim_ns,
                        start_odom_stamp_ns,
                        start_odom_serial,
                    ) = self._motion_start_baseline()
                    with self._motion_gate_lock:
                        arm_decision = self._motion_gate.arm(
                            episode_id=self.episode_id,
                            reset_generation=self.reset_generation,
                            sequence_id=int(result["sequence_id"]),
                            request_id=str(result["request_id"]),
                            action=resolved_action,
                            start_pose=start_pose,
                            start_sim_ns=start_sim_ns,
                            start_odom_stamp_ns=start_odom_stamp_ns,
                            start_odom_serial=start_odom_serial,
                            resolved_commanded_progress=resolved_commanded,
                            resolved_required_progress=resolved_required,
                        )
                        gate_snapshot = self._motion_gate.snapshot()
                    self._record_motion_gate(
                        arm_decision,
                        sim_stamp_ns=start_sim_ns,
                        odom_stamp_ns=start_odom_stamp_ns,
                        odom_serial=start_odom_serial,
                        snapshot=gate_snapshot,
                    )
                else:
                    with self._motion_gate_lock:
                        gate_snapshot = self._motion_gate.snapshot()
                result["motion_observation_gate"] = gate_snapshot
                result["motion_gate_model_action"] = model_action
                result["motion_gate_resolved_action"] = resolved_action
            if replan_requested:
                self._record_replan(
                    "consumed_by_fresh_model_step",
                    replan_index,
                    int(result.get("sequence_id", -1)),
                )
            if self.pose_audit_path is not None:
                record = {
                    "schema_version": 1,
                    "episode_id": self.episode_id,
                    "reset_generation": self.reset_generation,
                    "sequence_id": result.get("sequence_id"),
                    "source": "navigation_odometry",
                    "evaluator_ground_truth_pose_discarded": True,
                    "navigation_odometry_age_sec": age,
                    "navigation_odometry_reset_match_required": (
                        not self.allow_nearest_navigation_odometry
                    ),
                    "wall_time_unix": time.time(),
                }
                with self.pose_audit_path.open(
                    "a", encoding="utf-8", newline="\n"
                ) as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            return result
        except BaseException as exc:
            message = f"post-resolution motion gate failure: {exc}"
            if self._motion_gate_enabled and resolved_goal_sent:
                try:
                    self._stop_resolved_goal_after_failure(result, message)
                except BaseException as stop_exc:
                    self._safe_stop(
                        STATUS_STALE,
                        f"{message}; Nav2 cancellation failed: {stop_exc}",
                    )
                    raise ClientFailure(
                        STATUS_STALE,
                        f"{message}; Nav2 cancellation failed: {stop_exc}",
                    ) from stop_exc
            if isinstance(exc, ClientFailure):
                raise
            raise ClientFailure(STATUS_STALE, message) from exc

    def _resolved_motion_bounds(
        self,
        action: int,
        result: dict[str, Any],
        *,
        use_preregistered_bounds: bool = False,
    ) -> tuple[float, float]:
        """Bind the threshold to the motion actually resolved by Nav2."""

        return resolved_motion_bounds(
            action,
            self._motion_gate.config,
            result.get("local_path", []),
            use_preregistered_bounds=use_preregistered_bounds,
        )

    def _motion_gate_response(
        self,
        decision: GateDecision,
        *,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        age: float,
    ) -> dict[str, Any]:
        with self._motion_gate_lock:
            snapshot = self._motion_gate.snapshot()
        self._record_motion_gate(
            decision,
            sim_stamp_ns=sim_stamp_ns,
            odom_stamp_ns=odom_stamp_ns,
            odom_serial=odom_serial,
            snapshot=snapshot,
        )
        pending = snapshot.get("pending") or snapshot.get("stop_barrier") or {}
        return {
            "status_code": STATUS_OK,
            "status_message": decision.reason,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sequence_id": int(pending.get("sequence_id", self.next_sequence_id)),
            "request_id": str(
                pending.get(
                    "request_id",
                    f"motion-gate:{self.episode_id}:{self.reset_generation}",
                )
            ),
            "sim_stamp_ns": int(sim_stamp_ns),
            "discrete_action": ACTION_STAND_STILL,
            "model_discrete_action": int(pending.get("action", ACTION_STAND_STILL)),
            "stop": bool(decision.keep_safe_stop),
            "model_stop": False,
            "control_mode": self.control_mode,
            "nav2_goal_sent": False,
            "nav2_plan_valid": False,
            "motion_observation_gate_only": True,
            "motion_observation_gate_decision": decision.to_mapping(),
            "motion_observation_gate": snapshot,
            "observation_pose_source": "navigation_odometry",
            "evaluator_ground_truth_pose_discarded": True,
            "navigation_odometry_age_sec": age,
            "navigation_odometry_stamp_ns": int(odom_stamp_ns),
            "navigation_odometry_serial": int(odom_serial),
        }

    def record_step(self, result: dict[str, Any]) -> None:
        if not bool(result.get("motion_observation_gate_only", False)):
            super().record_step(result)
            return
        if self.records_path is None:
            return
        record = {"schema_version": 1, "wall_time_unix": time.time(), **result}
        with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = T4OdometryClientNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    bind_host = os.environ.get("INTERNVLA_CLIENT_TCP_BIND_HOST", "")
    if bind_host:
        bind_port = int(os.environ.get("INTERNVLA_CLIENT_TCP_PORT", "25139"))
        server: LocalIPCServer = TCPIPCServer(
            node,
            bind_host,
            bind_port,
            os.environ.get(
                "INTERNVLA_CLIENT_TCP_EXPECTED_PEER", "10.100.120.111"
            ),
        )
        endpoint = f"tcp://{bind_host}:{bind_port}"
    else:
        socket_path = Path(
            os.environ.get("INTERNVLA_CLIENT_SOCKET", "/tmp/internvla_client.sock")
        )
        server = LocalIPCServer(node, socket_path)
        endpoint = str(socket_path)
    try:
        thread.start()
        node.wait_for_graph()
        node.model_health_start = validate_uninitialized_model_health(node.health())
        node._write_client_summary("READY")
        node.get_logger().info(f"T4 odometry evaluator IPC ready at {endpoint}")
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_event.set()
        node.cancel_current()
        node.finish_records()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
