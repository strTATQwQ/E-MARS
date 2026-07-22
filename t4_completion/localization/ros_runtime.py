"""Optional ROS 2 adapter; importing this module does not require ROS."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import LocalizationConfig
from .contracts import ALL_SOURCES, Pose, PoseSample
from .evidence import LocalizationEvidenceWriter
from .selector import LocalizationSelector


HEALTH_ATTESTATION_KEYS = {
    "schema_version",
    "source",
    "reset_generation",
    "sequence_id",
    "sample_stamp_ns",
    "backend_ready",
    "tracking",
    "reason",
}


def _health_attestation(raw: str, expected_source: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != HEALTH_ATTESTATION_KEYS:
        raise ValueError("health_attestation_keys_mismatch")
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != 1
        or not isinstance(payload["source"], str)
        or payload["source"] != expected_source
    ):
        raise ValueError("health_attestation_identity_mismatch")
    if not isinstance(payload["backend_ready"], bool) or not isinstance(
        payload["tracking"], bool
    ):
        raise ValueError("health_attestation_boolean_mismatch")
    if not isinstance(payload["reason"], str) or not payload["reason"]:
        raise ValueError("health_attestation_reason_missing")
    for key in ("reset_generation", "sequence_id", "sample_stamp_ns"):
        if (
            isinstance(payload[key], bool)
            or not isinstance(payload[key], int)
            or payload[key] < 0
        ):
            raise ValueError("health_attestation_integer_invalid")
    if payload["sample_stamp_ns"] <= 0:
        raise ValueError("health_attestation_stamp_invalid")
    return payload


def run_ros(config: LocalizationConfig, result_dir: Path) -> int:
    """Run the ROS boundary.  ROS imports stay inside this explicit entrypoint."""

    try:
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from rclpy.time import Time
        from std_msgs.msg import Bool, Int32, String
        from tf2_ros import TransformBroadcaster
    except ImportError as exc:  # pragma: no cover - exercised only on ROS hosts
        raise RuntimeError("ROS 2 dependencies unavailable; core remains offline-usable") from exc

    selector = LocalizationSelector(policy=config.policy, sources=config.sources)
    writer = LocalizationEvidenceWriter(
        result_dir,
        record_filename=config.evidence_record_filename,
        summary_filename=config.evidence_summary_filename,
    )

    class LocalizationNode(Node):
        def __init__(self) -> None:
            super().__init__(
                "internvla_t4_completion_localization_selector",
                automatically_declare_parameters_from_overrides=True,
                parameter_overrides=[
                    Parameter("use_sim_time", value=config.use_sim_time)
                ],
            )
            if not self.has_parameter("use_sim_time") or not bool(
                self.get_parameter("use_sim_time").value
            ):
                raise RuntimeError("use_sim_time must be explicitly true")
            self.selector = selector
            self.writer = writer
            self._pending_odometry: dict[tuple[str, int], tuple[Any, int]] = {}
            self._pending_health: dict[
                tuple[str, int], tuple[dict[str, Any], int]
            ] = {}
            self._last_no_output_evidence_ns = 0
            self.oracle_complete = False
            ingress_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=20,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            canonical_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=4,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.odom_publisher = self.create_publisher(
                Odometry, "/odom", canonical_qos
            )
            self.health_publisher = self.create_publisher(
                String, config.output_health_topic, 10
            )
            self.switch_publisher = self.create_publisher(
                String, config.output_switch_topic, 10
            )
            self.deviation_publisher = self.create_publisher(
                String, config.output_deviation_topic, 10
            )
            self.tf_broadcaster = TransformBroadcaster(self)
            self.create_subscription(
                Int32,
                "/internvla_t4/map_reset_generation",
                self._on_reset,
                10,
            )
            self.create_subscription(
                Bool,
                config.oracle_completion_topic,
                self._on_oracle_complete,
                10,
            )
            for source in ALL_SOURCES:
                spec = config.sources[source]
                self.create_subscription(
                    Odometry,
                    spec.odometry_topic,
                    lambda message, source=source: self._on_odometry(source, message),
                    ingress_qos,
                )
                self.create_subscription(
                    String,
                    spec.health_topic,
                    lambda message, source=source: self._on_health(source, message),
                    ingress_qos,
                )
            self.create_timer(0.05, self._tick)

        @staticmethod
        def _stamp_ns(message: Any) -> int:
            return int(message.header.stamp.sec) * 1_000_000_000 + int(
                message.header.stamp.nanosec
            )

        def _trim_pending(self, now_ns: int) -> None:
            # At most 64 identities per source; unmatched data remains unhealthy
            # and cannot silently enter the selector.
            expired: set[str] = set()
            for key, (_message, received_ns) in list(
                self._pending_odometry.items()
            ):
                if now_ns - received_ns > config.policy.source_timeout_ns:
                    del self._pending_odometry[key]
                    expired.add(key[0])
            for key, (_health, received_ns) in list(self._pending_health.items()):
                if now_ns - received_ns > config.policy.source_timeout_ns:
                    del self._pending_health[key]
                    expired.add(key[0])
            for source in expired:
                self.selector.invalidate_source(source, "health_odometry_pair_timeout")
            for pending in (self._pending_odometry, self._pending_health):
                while len(pending) > 64 * len(ALL_SOURCES):
                    removed = next(iter(pending))
                    del pending[removed]
                    self.selector.invalidate_source(
                        removed[0], "health_odometry_pair_capacity_exceeded"
                    )

        def _on_reset(self, message: Any) -> None:
            generation = int(message.data)
            if generation > self.selector.generation:
                self.selector.reset_generation(generation)
                self._pending_odometry.clear()
                self._pending_health.clear()

        def _on_oracle_complete(self, message: Any) -> None:
            if bool(message.data):
                # Refresh source freshness at the completion boundary.  A
                # historical HEALTHY tick must not turn a timer-starved or
                # stale runtime into PASS.
                self._tick(force_evidence=True)
                self.oracle_complete = True
                rclpy.shutdown()

        def _on_odometry(self, source: str, message: Any) -> None:
            stamp_ns = self._stamp_ns(message)
            key = (source, stamp_ns)
            if key in self._pending_odometry:
                del self._pending_odometry[key]
                self.selector.invalidate_source(source, "duplicate_odometry_identity")
                return
            now_ns = time.monotonic_ns()
            self._pending_odometry[key] = (
                message,
                now_ns,
            )
            self._match(source, stamp_ns)
            self._trim_pending(now_ns)

        def _on_health(self, source: str, message: Any) -> None:
            try:
                health = _health_attestation(str(message.data), source)
            except (ValueError, json.JSONDecodeError):
                self.selector.invalidate_source(
                    source, "malformed_health_attestation"
                )
                return
            stamp_ns = int(health["sample_stamp_ns"])
            key = (source, stamp_ns)
            if key in self._pending_health:
                del self._pending_health[key]
                self.selector.invalidate_source(source, "duplicate_health_identity")
                return
            now_ns = time.monotonic_ns()
            self._pending_health[key] = (health, now_ns)
            self._match(source, stamp_ns)
            self._trim_pending(now_ns)

        def _match(self, source: str, stamp_ns: int) -> None:
            key = (source, stamp_ns)
            odometry_item = self._pending_odometry.get(key)
            health_item = self._pending_health.get(key)
            if odometry_item is None or health_item is None:
                return
            message, odometry_received_ns = odometry_item
            health, health_received_ns = health_item
            received_ns = min(odometry_received_ns, health_received_ns)
            del self._pending_odometry[key]
            del self._pending_health[key]
            pose = message.pose.pose
            twist = message.twist.twist
            sample = PoseSample(
                source=source,
                generation=int(health["reset_generation"]),
                sequence_id=int(health["sequence_id"]),
                stamp_ns=stamp_ns,
                received_monotonic_ns=received_ns,
                parent_frame=str(message.header.frame_id),
                child_frame=str(message.child_frame_id),
                pose=Pose(
                    (
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    ),
                    (
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ),
                ),
                linear_velocity_xyz=(
                    float(twist.linear.x),
                    float(twist.linear.y),
                    float(twist.linear.z),
                ),
                angular_velocity_xyz=(
                    float(twist.angular.x),
                    float(twist.angular.y),
                    float(twist.angular.z),
                ),
                backend_ready=bool(health["backend_ready"]),
                tracking=bool(health["tracking"]),
                health_reason=str(health["reason"]),
            )
            self.selector.ingest(sample)

        @staticmethod
        def _string(payload: dict[str, Any]) -> Any:
            message = String()
            message.data = json.dumps(payload, sort_keys=True, allow_nan=False)
            return message

        def _publish_output(self, output: Any) -> None:
            stamp = Time(nanoseconds=output.stamp_ns).to_msg()
            odometry = Odometry()
            odometry.header.stamp = stamp
            odometry.header.frame_id = output.parent_frame
            odometry.child_frame_id = output.child_frame
            odometry.pose.pose.position.x = output.pose.translation[0]
            odometry.pose.pose.position.y = output.pose.translation[1]
            odometry.pose.pose.position.z = output.pose.translation[2]
            odometry.pose.pose.orientation.x = output.pose.quaternion_xyzw[0]
            odometry.pose.pose.orientation.y = output.pose.quaternion_xyzw[1]
            odometry.pose.pose.orientation.z = output.pose.quaternion_xyzw[2]
            odometry.pose.pose.orientation.w = output.pose.quaternion_xyzw[3]
            odometry.twist.twist.linear.x = output.linear_velocity_xyz[0]
            odometry.twist.twist.linear.y = output.linear_velocity_xyz[1]
            odometry.twist.twist.linear.z = output.linear_velocity_xyz[2]
            odometry.twist.twist.angular.x = output.angular_velocity_xyz[0]
            odometry.twist.twist.angular.y = output.angular_velocity_xyz[1]
            odometry.twist.twist.angular.z = output.angular_velocity_xyz[2]
            transform = TransformStamped()
            transform.header = odometry.header
            transform.child_frame_id = output.child_frame
            transform.transform.translation.x = output.pose.translation[0]
            transform.transform.translation.y = output.pose.translation[1]
            transform.transform.translation.z = output.pose.translation[2]
            transform.transform.rotation = odometry.pose.pose.orientation
            self.odom_publisher.publish(odometry)
            self.tf_broadcaster.sendTransform(transform)

        def _tick(self, *, force_evidence: bool = False) -> None:
            now_ns = time.monotonic_ns()
            self._trim_pending(now_ns)
            decision = self.selector.decide(
                now_ns, int(self.get_clock().now().nanoseconds)
            )
            payload = decision.to_dict()
            self.health_publisher.publish(
                self._string(
                    {
                        "schema_version": 1,
                        "runtime_policy": decision.runtime_policy,
                        "runtime_target": decision.runtime_target,
                        "generation": decision.generation,
                        "selected_source": decision.selected_source,
                        "source_health": payload["source_health"],
                    }
                )
            )
            if decision.output is not None:
                self._publish_output(decision.output)
            if decision.switch_event:
                self.switch_publisher.publish(self._string(payload))
            if decision.deviation is not None:
                self.deviation_publisher.publish(self._string(decision.deviation))
            if (
                force_evidence
                or decision.output is not None
                or decision.switch_event
                or now_ns - self._last_no_output_evidence_ns >= 1_000_000_000
            ):
                self.writer.append(decision)
                if decision.output is None:
                    self._last_no_output_evidence_ns = now_ns

    node: Any | None = None
    status = "FAIL"
    initialized = False
    try:  # pragma: no cover - online-only boundary
        # Do not inherit arbitrary ROS arguments from the wrapper CLI.  The
        # exact completion-only parameter contract is injected above.
        rclpy.init(args=[])
        initialized = True
        node = LocalizationNode()
        rclpy.spin(node)
        summary = selector.summary()
        status = (
            "PASS_WITH_DEVIATION"
            if node.oracle_complete
            and summary["current_output_available"]
            and summary["deviation_count"]
            else (
                "PASS"
                if node.oracle_complete and summary["current_output_available"]
                else "FAIL"
            )
        )
        return 0 if status != "FAIL" else 2
    finally:
        try:
            writer.close(selector.summary(), status=status)
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                if initialized and rclpy.ok():
                    rclpy.shutdown()
